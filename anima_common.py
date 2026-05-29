"""
============================================================================
  anima_common.py — 四个节点共享的核心逻辑
============================================================================
发布版加载/导出、实验版 LoKr 加载/导出 都从这里 import，保证逻辑单一来源、
改一处四节点同步，避免拷贝多份导致发布版/实验版不一致。

本模块包含：
- block 定位、子模块分类、区间解析
- 三种格式检测与兼容：
    · kohya LoRA   : .lora_down.weight / .lora_up.weight
    · diffusers LoRA: .lora_A.weight / .lora_B.weight   ← 新增兼容（如 anima masterpiece）
    · LoKr         : .lokr_w1 / .lokr_w2(_a/_b) / .lokr_t2
- impact 影响力计算（三格式通吃）
- 分层缩放（三格式通吃）
- auto_segment 自动分段（分位法 + 平滑 + 最小连续段长保护）

★ 四段命名（v 通用版起）：seg_1 / seg_2 / seg_3 / seg_4
  （弱/中/强/峰，按"实测强度档位"命名，不再绑定 Anima 功能假设；
   因为不同 LoRA 强区位置不同——画风在中部、角色可能在尾部，
   旧的 motion/proportion/core/detail 名不副实，故改为中性强度名。）
"""

import re

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


BLOCK_RE = re.compile(r"blocks[_.](\d+)[_.]")
TOTAL_BLOCKS_DEFAULT = 28

# 四段（纯位置编号命名：seg_1=最前段 ... seg_4=最后段，按 block 位置从前到后）
# 用纯编号是因为同一套节点支持两种分段指标（norm/effective_rank），
# 各指标下"第几档"的含义不同（范数下高档=强，有效秩下含义不同甚至相反），
# 任何含"强弱/功能"暗示的名字都会在某个指标下误导，故用中性编号。
SEG_NAMES = ["seg_1", "seg_2", "seg_3", "seg_4"]
# 默认区间：仅作未跑自动分段时的初始占位（按 block 位置均匀四分）；
# 真正使用时建议打开 auto_segment（现已默认开启）让它按实测指标重算。
SEG_DEFAULT_RANGES = {
    "seg_1": "0-6",
    "seg_2": "7-13",
    "seg_3": "14-20",
    "seg_4": "21-27",
}


def classify_submodule(key: str) -> str:
    k = key.lower()
    if "adaln" in k:
        return "adaln"
    if "self_attn" in k:
        return "self_attn"
    if "cross_attn" in k:
        return "cross_attn"
    if "mlp" in k:
        return "mlp"
    return "other"


def parse_block_range(range_str, total_blocks):
    range_str = (range_str or "").strip().lower()
    if range_str in ("", "all"):
        return set(range(total_blocks))
    result = set()
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a, b = int(a), int(b)
                for i in range(min(a, b), max(a, b) + 1):
                    if 0 <= i < total_blocks:
                        result.add(i)
            except ValueError:
                pass
        else:
            try:
                i = int(part)
                if 0 <= i < total_blocks:
                    result.add(i)
            except ValueError:
                pass
    return result if result else set(range(total_blocks))


def detect_total_blocks(raw):
    max_block = -1
    for k in raw:
        m = BLOCK_RE.search(k)
        if m:
            max_block = max(max_block, int(m.group(1)))
    return max_block + 1 if max_block >= 0 else 0


# ---------------------------------------------------------------------------
#  格式检测与兼容
# ---------------------------------------------------------------------------
def detect_format(raw):
    """返回 'lokr' / 'lora_kohya' / 'lora_diffusers' / 'unknown'。"""
    if any(k.endswith(".lokr_w1") for k in raw):
        return "lokr"
    if any(k.endswith(".lora_down.weight") for k in raw):
        return "lora_kohya"
    if any(k.endswith(".lora_A.weight") for k in raw):
        return "lora_diffusers"
    return "unknown"


# 各格式里"该被缩放的那一块"（down 侧 / w1 侧）的后缀
_DOWN_SUFFIXES = (".lora_down.weight", ".lora_down", ".lora_A.weight", ".lora_A")
_UP_SUFFIXES = (".lora_up.weight", ".lora_up", ".lora_B.weight", ".lora_B")
_SKIP_SUFFIXES = (".alpha", ".dora_scale")

_ALL_MODULE_SUFFIXES = (
    ".lokr_w1_a", ".lokr_w1_b", ".lokr_w1",
    ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2", ".lokr_w2",
    ".lora_down.weight", ".lora_up.weight", ".lora_down", ".lora_up",
    ".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B",
    ".alpha", ".dora_scale",
)


def module_key(key):
    for suf in _ALL_MODULE_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


def _norm(tensor):
    if not _HAS_TORCH:
        return None
    try:
        return float(torch.linalg.vector_norm(tensor.float()).item())
    except Exception:
        try:
            return float(tensor.abs().sum().item())
        except Exception:
            return None


def _rebuild_lokr_factor(raw, mk, which):
    """重建 LoKr 的 w1 或 w2（整块 / 低秩分解 a@b / tucker）。不乘 alpha/rank。"""
    if not _HAS_TORCH:
        return None
    full = f"{mk}.lokr_{which}"
    if full in raw:
        try:
            return raw[full].float()
        except Exception:
            return None
    a_key = f"{mk}.lokr_{which}_a"
    b_key = f"{mk}.lokr_{which}_b"
    if a_key in raw and b_key in raw:
        try:
            a = raw[a_key].float()
            b = raw[b_key].float()
            t_key = f"{mk}.lokr_t2"
            if which == "w2" and t_key in raw:
                t2 = raw[t_key].float()
                return torch.einsum("i j k l, j r, i p -> p r k l", t2, b, a)
            return a @ b
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
#  impact 影响力计算（三格式通吃）
# ---------------------------------------------------------------------------
def compute_block_impact(raw, total_blocks, fmt=None):
    """
    每个 block 的影响力分数（0..1），做对比度拉伸（[min,max]->[0,1]）。
    - LoKr     : 等效权重范数 = ‖w1‖·‖w2‖（先重建 w1/w2，含分解/tucker）。
    - LoRA(任一命名): 累加 block 内所有权重张量（down/up 或 A/B）的范数。
    torch 不可用或失败时返回全 0。
    """
    if not _HAS_TORCH:
        return [0.0] * total_blocks
    if fmt is None:
        fmt = detect_format(raw)
    energy = [0.0] * total_blocks
    try:
        if fmt == "lokr":
            mod_blk = {}
            for key in raw:
                if ".lokr_" not in key:
                    continue
                m = BLOCK_RE.search(key)
                if not m:
                    continue
                idx = int(m.group(1))
                if 0 <= idx < total_blocks:
                    mod_blk[module_key(key)] = idx
            for mk, idx in mod_blk.items():
                w1 = _rebuild_lokr_factor(raw, mk, "w1")
                w2 = _rebuild_lokr_factor(raw, mk, "w2")
                n1 = _norm(w1) if w1 is not None else None
                n2 = _norm(w2) if w2 is not None else None
                if n1 is not None and n2 is not None:
                    energy[idx] += n1 * n2
                elif n1 is not None:
                    energy[idx] += n1
                elif n2 is not None:
                    energy[idx] += n2
        else:
            # LoRA：kohya 或 diffusers 都走这里——累加权重张量范数
            for key, tensor in raw.items():
                m = BLOCK_RE.search(key)
                if not m:
                    continue
                idx = int(m.group(1))
                if not (0 <= idx < total_blocks):
                    continue
                if key.endswith(_SKIP_SUFFIXES):
                    continue
                n = _norm(tensor)
                if n is not None:
                    energy[idx] += n

        mx = max(energy) if energy else 0.0
        mn = min(energy) if energy else 0.0
        if mx <= 0:
            return [0.0] * total_blocks
        span = mx - mn
        if span <= 1e-9:
            return [0.5] * total_blocks
        return [(e - mn) / span for e in energy]
    except Exception:
        return [0.0] * total_blocks


# ---------------------------------------------------------------------------
#  有效秩指标（与范数互补的另一个静态指标）
# ---------------------------------------------------------------------------
#  动机：范数只反映"强弱"。实测发现有效秩与范数高度负相关——
#  范数高的 block 往往能量集中在少数方向（强而专一的变换），
#  范数低的 block 往往方向分散（广泛而微弱的全局调整，如色彩/质感）。
#  所以"按有效秩分段"能挑出范数分段漏掉的"弥漫性调整段"，与范数分段互补。
#  到底哪个更对应功能段，需用户出图实测——这正是做双版本的目的。
#
#  性能：用 gram 矩阵 + eigvalsh 代替直接 SVD（奇异值 = sqrt(对称阵特征值)），
#  对 r×in 这类矮胖矩阵快约 80 倍（实测单模块 72ms -> 0.9ms）。
def _effective_rank(W):
    """权重矩阵的有效秩 = exp(奇异值分布的熵)。越大=用的独立方向越多。"""
    if not _HAS_TORCH:
        return 0.0
    try:
        W = W.float()
        if W.dim() > 2:
            W = W.flatten(1)
        # 用较小的那一边做 gram，奇异值平方 = gram 的特征值
        G = (W @ W.t()) if W.shape[0] <= W.shape[1] else (W.t() @ W)
        ev = torch.linalg.eigvalsh(G)
        ev = ev[ev > 1e-8]
        if len(ev) == 0:
            return 0.0
        s = torch.sqrt(ev)
        p = s / s.sum()
        import math
        return float(math.exp(float(-(p * torch.log(p + 1e-12)).sum())))
    except Exception:
        return 0.0


def _effrank_lora_module(raw, mk):
    """LoRA 模块的有效秩：用 down(或 A) 矩阵——它的方向数反映该模块用了多少独立方向。"""
    for dn in (".lora_down.weight", ".lora_down", ".lora_A.weight", ".lora_A"):
        if mk + dn in raw:
            return _effective_rank(raw[mk + dn])
    return None


def compute_block_effrank(raw, total_blocks, fmt=None):
    """每个 block 的有效秩分数（0..1，对比度拉伸）。LoKr/LoRA 通吃。"""
    if not _HAS_TORCH:
        return [0.0] * total_blocks
    if fmt is None:
        fmt = detect_format(raw)
    vals = [0.0] * total_blocks
    cnt = [0] * total_blocks
    try:
        mod_blk = {}
        for key in raw:
            if key.endswith(_SKIP_SUFFIXES):
                continue
            m = BLOCK_RE.search(key)
            if not m:
                continue
            idx = int(m.group(1))
            if 0 <= idx < total_blocks:
                mod_blk[module_key(key)] = idx
        for mk, idx in mod_blk.items():
            if fmt == "lokr":
                # LoKr：用 w1（整块或 a）的有效秩
                w1 = _rebuild_lokr_factor(raw, mk, "w1")
                er = _effective_rank(w1) if w1 is not None else None
            else:
                er = _effrank_lora_module(raw, mk)
            if er is not None:
                vals[idx] += er
                cnt[idx] += 1
        energy = [vals[i] / cnt[i] if cnt[i] else 0.0 for i in range(total_blocks)]
        mx, mn = max(energy), min(energy)
        if mx <= 0:
            return [0.0] * total_blocks
        span = mx - mn
        if span <= 1e-9:
            return [0.5] * total_blocks
        return [(e - mn) / span for e in energy]
    except Exception:
        return [0.0] * total_blocks


def compute_block_metric(raw, total_blocks, metric="norm", fmt=None):
    """按选定指标计算每个 block 的分数。metric: 'norm'（范数，默认）/ 'effective_rank'（有效秩）。"""
    if metric == "effective_rank":
        return compute_block_effrank(raw, total_blocks, fmt=fmt)
    return compute_block_impact(raw, total_blocks, fmt=fmt)


# ---------------------------------------------------------------------------
#  分层缩放（三格式通吃）
# ---------------------------------------------------------------------------
def apply_layered_scaling(raw, block_w, type_weight):
    """
    返回 (新state_dict, 已缩放张量数, 格式)。
    每个模块只缩放"down 侧一块"，等效整体 factor：
      - LoRA kohya    : 缩 .lora_down
      - LoRA diffusers: 缩 .lora_A
      - LoKr          : 缩整块 .lokr_w1，或分解的 .lokr_w1_a（绝不同时缩 _a 和 _b，绝不碰 w2）
    alpha / dora_scale / up 侧 / w2 侧：原样保留。
    """
    fmt = detect_format(raw)
    out = {}
    scaled = 0

    if fmt == "lokr":
        has_full_w1 = {}
        for key in raw:
            if key.endswith(".lokr_w1"):
                has_full_w1[module_key(key)] = key
        scale_target = {}
        for key in raw:
            mk = module_key(key)
            if mk in has_full_w1:
                scale_target[mk] = has_full_w1[mk]
            elif key.endswith(".lokr_w1_a"):
                scale_target.setdefault(mk, key)
        for key, tensor in raw.items():
            m = BLOCK_RE.search(key)
            if not m:
                out[key] = tensor.clone() if _HAS_TORCH else tensor
                continue
            idx = int(m.group(1))
            bw = block_w[idx] if idx < len(block_w) else 1.0
            factor = bw * type_weight[classify_submodule(key)]
            if scale_target.get(module_key(key)) == key:
                out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
                scaled += 1
            else:
                out[key] = tensor.clone()
        return out, scaled, fmt

    # LoRA（kohya / diffusers）
    for key, tensor in raw.items():
        m = BLOCK_RE.search(key)
        if not m:
            out[key] = tensor.clone() if _HAS_TORCH else tensor
            continue
        idx = int(m.group(1))
        bw = block_w[idx] if idx < len(block_w) else 1.0
        factor = bw * type_weight[classify_submodule(key)]
        if key.endswith(_DOWN_SUFFIXES):
            out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
            scaled += 1
        else:
            out[key] = tensor.clone()
    return out, scaled, fmt


# ---------------------------------------------------------------------------
#  按四段构建逐 block 权重
# ---------------------------------------------------------------------------
def build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs):
    """
    grouped：四段后段覆盖前段（seg_specs 顺序 weak→medium→strong→peak）。
    per_block：读 blk00..NN。
    """
    if control_mode == "per_block":
        block_w = []
        for i in range(total_blocks):
            v = block_kwargs.get(f"blk{i:02d}", 1.0)
            try:
                block_w.append(float(v))
            except (ValueError, TypeError):
                block_w.append(1.0)
        return block_w
    segs = [(parse_block_range(rs, total_blocks), w) for rs, w in seg_specs]
    block_w = []
    for i in range(total_blocks):
        val = 1.0
        for sset, sw in segs:
            if i in sset:
                val = sw
        block_w.append(val)
    return block_w


# ---------------------------------------------------------------------------
#  自动分段（分位法 + 平滑 + 最小连续段长保护）
#  已在 6 个真实文件（画风/美学/角色 LoRA + 三种 LoKr）验证：
#  每档平均强度严格 弱<中<强<峰，段内连续、段间可切开。
# ---------------------------------------------------------------------------
def _smooth(imp, win=1):
    n = len(imp)
    out = []
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        out.append(sum(imp[lo:hi]) / (hi - lo))
    return out


def _ranges_from_blocks(blocks):
    if not blocks:
        return ""
    blocks = sorted(blocks)
    parts = []
    s = p = blocks[0]
    for b in blocks[1:]:
        if b == p + 1:
            p = b
        else:
            parts.append((s, p))
            s = p = b
    parts.append((s, p))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in parts)


def _tier_quantile(sm, total_blocks):
    """分位法：按值排序四等分。每档 block 数量均衡（约 total/4）。"""
    order = sorted(range(total_blocks), key=lambda i: (sm[i], i))
    q = total_blocks / 4.0
    tier = [0] * total_blocks
    for rank, blk in enumerate(order):
        tt = int(rank // q)
        if tt > 3:
            tt = 3
        tier[blk] = tt
    return tier


def _tier_jenks(sm, total_blocks, k=4):
    """
    Jenks 自然断点法（Jenks Natural Breaks, George F. Jenks 1967）：
    经典的一维数据聚类/分类方法，在数据"自然的缝隙"处切档，使类内方差最小、类间方差最大。
    相比分位法（机械四等分），它更尊重数据的真实分布——能量集中在何处，就把强档放在何处，
    档大小可不均衡（这是如实反映 LoRA 真实结构，而非缺陷）。
    这里用经典 Fisher-Jenks 动态规划实现。仅保证每档至少 1 个 block（防止空档导致滑块失效）。
    """
    n = total_blocks
    srt = sorted(range(n), key=lambda i: sm[i])
    vals = [sm[i] for i in srt]
    # 动态规划矩阵
    m1 = [[0] * (k + 1) for _ in range(n + 1)]
    m2 = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    for i in range(1, k + 1):
        m1[1][i] = 1
        m2[1][i] = 0.0
    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = vals[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    if m2[l][j] >= (v + m2[i4][j - 1]):
                        m1[l][j] = i3
                        m2[l][j] = v + m2[i4][j - 1]
        m1[l][1] = 1
        m2[l][1] = s2 - (s1 * s1) / w
    # 回溯类边界（排序序列上的切点）
    kc = [0] * (k + 1)
    kc[k] = n
    kk = n
    for j in range(k, 1, -1):
        kc[j - 1] = m1[kk][j] - 1
        kk = m1[kk][j] - 1
    bounds = [0] + kc[1:k] + [n]
    # 防空档保护：若某档为空（low-variance 数据可能出现），退回分位法
    sizes = [bounds[i + 1] - bounds[i] for i in range(k)]
    if any(s <= 0 for s in sizes):
        return None
    tier = [0] * n
    for cls in range(k):
        for rank in range(bounds[cls], bounds[cls + 1]):
            tier[srt[rank]] = cls
    return tier


def auto_segment(impact, total_blocks=28, min_run=2, smooth_win=1, method="quantile"):
    """
    输入 impact（长度 total_blocks 的 0..1 强度），输出四段区间字符串 dict：
      {"seg_1": "...", "seg_2": "...", "seg_3": "...", "seg_4": "..."}

    method:
      - "quantile"（分位法/均分，默认）：按值排序四等分，每档数量均衡、段内连续、手感稳。
        额外做最小段长保护（消除 < min_run 的连续碎片），适合开箱即用。
      - "jenks"（自然断点）：Jenks Natural Breaks，按数据自然缝隙切档，更尊重 LoRA 真实分布，
        档大小可不均衡（强区大、弱区小是如实反映）。段内可不连续。

    两种方式都返回 (四段区间 dict, tier 列表)。
    """
    if not impact or len(impact) < total_blocks or max(impact) <= 0:
        # 没有有效强度（如 torch 不可用 / 非 Anima 文件）：返回空，前端不覆盖现有值
        return {n: "" for n in SEG_NAMES}, None

    if method == "jenks":
        # Jenks 用【原始】值，不做平滑——Jenks 本就按真实数值找缝隙，
        # 平滑会把"高值夹在低值中间"的 block 误拉低（如某 block 有效秩最高却被左右邻居平均下来掉档）。
        # 平滑只服务于分位法（用来抹毛刺保证连续），不该污染 Jenks 的输入。
        tier = _tier_jenks(impact, total_blocks)
        if tier is not None:
            tiers = {0: [], 1: [], 2: [], 3: []}
            for i in range(total_blocks):
                tiers[tier[i]].append(i)
            return {SEG_NAMES[t]: _ranges_from_blocks(tiers[t]) for t in range(4)}, tier
        # Jenks 失败（如出现空档）→ 退回分位法兜底

    # 分位法（默认，或 Jenks 兜底）：用平滑值，抹掉孤立毛刺让分段更连续
    sm = _smooth(impact, smooth_win)
    tier = _tier_quantile(sm, total_blocks)

    def runs(t):
        res = []
        s = 0
        for i in range(1, total_blocks + 1):
            if i == total_blocks or t[i] != t[s]:
                res.append((s, i - 1, t[s]))
                s = i
        return res

    changed = True
    guard = 0
    while changed and guard < 100:
        guard += 1
        changed = False
        rs = runs(tier)
        for idx, (a, b, tval) in enumerate(rs):
            if b - a + 1 < min_run:
                left = rs[idx - 1] if idx > 0 else None
                right = rs[idx + 1] if idx < len(rs) - 1 else None
                cand = []
                if left:
                    cand.append((abs(left[2] - tval), left[2]))
                if right:
                    cand.append((abs(right[2] - tval), right[2]))
                if cand:
                    cand.sort()
                    newt = cand[0][1]
                    for i in range(a, b + 1):
                        tier[i] = newt
                    changed = True
                    break

    tiers = {0: [], 1: [], 2: [], 3: []}
    for i in range(total_blocks):
        tiers[tier[i]].append(i)
    return {SEG_NAMES[t]: _ranges_from_blocks(tiers[t]) for t in range(4)}, tier


def v2_segment_inputs():
    """四段 + 子模块系数 + 自动分段开关的 INPUT_TYPES 公共定义。"""
    d = {
        "control_mode": (["grouped", "per_block"], {"default": "grouped"}),
        "auto_segment": ("BOOLEAN", {"default": True}),
        "segment_metric": (["norm", "effective_rank"], {"default": "norm"}),
        "segment_method": (["jenks", "quantile"], {"default": "jenks"}),
    }
    for name in SEG_NAMES:
        d[f"{name}_blocks"] = ("STRING", {"default": SEG_DEFAULT_RANGES[name]})
        d[f"{name}_weight"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
    d["w_self_attn"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
    d["w_cross_attn"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
    d["w_mlp"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
    d["w_adaln"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
    return d
