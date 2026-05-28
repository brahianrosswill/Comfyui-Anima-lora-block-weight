"""
============================================================================
  实验性分支 (EXPERIMENTAL) — LoKr 分层支持 (V2 参数体系)
============================================================================
⚠️ 这是实验性节点，独立于正式发布的两个节点，不影响它们。
   以 try/except 安全加载；即使出错也不影响发布版。

与普通 LoRA 的区别
------------------------------------------------------------------------
普通 LoRA 每个模块是 lora_down + lora_up；
LoKr 每个模块是 lokr_w1 + lokr_w2，等效贡献 = lokr_w1 ⊗ lokr_w2（克罗内克积）。

缩放原理（已用 torch 数值验证，误差 ~1e-7）
------------------------------------------------------------------------
克罗内克积性质：(c · w1) ⊗ w2 == c · (w1 ⊗ w2)
=> 把 factor 只乘到 lokr_w1 一块，等效于整体缩放 factor。
   绝不能两块都乘——那会变成 factor²。

impact 范数（已用 torch 在真实文件上数值验证）
------------------------------------------------------------------------
关键性质：‖w1 ⊗ w2‖_F == ‖w1‖_F · ‖w2‖_F
=> 不需要真的 torch.kron 重建大矩阵（会爆内存），直接用 ‖w1‖·‖w2‖
   即等于「重建等效权重后再算 L2 范数」，数值完全吻合且省内存。
   ⚠️ 但 w1/w2 可能是「整块」/「低秩分解 a@b」/「tucker」形式——
      必须先按推理逻辑重建出完整 w1 / w2 再算范数。
      不能对 .lokr_w2_a / .lokr_w2_b 单独算（‖a@b‖≠‖a‖‖b‖，且漏项会让跨 block 排序乱）。
      也不能只取 ‖w1‖（漏掉 w2 那一项，跨 block 比较会排错序）。

重要警告（针对本文件作者的特定 LoKr）
------------------------------------------------------------------------
该 LoKr 训练时用了一个 trick：把 dim/alpha 设成超大值(1000000)交给
Prodigy 自由控制。存成 fp16 后这些值溢出为 inf。
=> 本节点绝不读取/使用 alpha 或 lokr_rank 做任何计算，只对 lokr_w1 做乘法、
   对 w1/w2 算范数，从而完全避开 inf/NaN 问题。

本节点同时兼容普通 LoRA（检测到 lora_down 时按 LoRA 方式处理），
是个"通吃"实验版；但请仍把它当实验品，正式用发布版节点。

★ 本版本已对齐发布版 V2 的参数体系（grouped 四段 + per_block 28 滑块 + 子模块系数 +
  impact 染色），并共享发布版的前端面板 web/anima_block_weight.js。
"""

import os
import re
import json
import torch
import folder_paths
import comfy.utils
import comfy.sd
from safetensors.torch import save_file


BLOCK_RE = re.compile(r"blocks[_.](\d+)[_.]")
TOTAL_BLOCKS_DEFAULT = 28


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


def detect_format(raw):
    """返回 'lokr' / 'lora' / 'unknown'。"""
    has_lokr = any(k.endswith(".lokr_w1") for k in raw)
    has_lora = any(k.endswith(".lora_down.weight") for k in raw)
    if has_lokr:
        return "lokr"
    if has_lora:
        return "lora"
    return "unknown"


_LOKR_SUFFIXES = (
    ".lokr_w1_a", ".lokr_w1_b", ".lokr_w1",
    ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2", ".lokr_w2",
)
_GENERIC_SUFFIXES = _LOKR_SUFFIXES + (
    ".lora_down.weight", ".lora_up.weight", ".lora_down", ".lora_up",
    ".alpha", ".dora_scale",
)


def _module_key(key: str) -> str:
    """去掉张量后缀，得到所属"模块"标识，用于把同一模块的各张量归并到一起。
    支持 LoKr 的整块 / 低秩分解 / tucker 形式，以及 LoRA、alpha、dora_scale。"""
    for suf in _GENERIC_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


def _norm(tensor):
    """张量 Frobenius/L2 范数，失败时退化为绝对值和；空/异常返回 None。"""
    try:
        return float(torch.linalg.vector_norm(tensor.float()).item())
    except Exception:
        try:
            return float(tensor.abs().sum().item())
        except Exception:
            return None


def _rebuild_lokr_factor(raw, mk, which):
    """
    重建 LoKr 的 w1 或 w2（which ∈ {"w1","w2"}），返回 torch 张量或 None。
    与 ComfyUI/LyCORIS 推理一致：
      - 整块:        .lokr_wX
      - 低秩分解:    .lokr_wX_a @ .lokr_wX_b
      - w2 的 tucker: einsum(.lokr_t2, .lokr_w2_b, .lokr_w2_a)
    ⚠️ 完全不乘 alpha/rank：本节点的设计原则是不读 alpha（避开 inf 陷阱）。
       实测这些文件 alpha/rank≈1 且各模块一致，不影响相对排序；
       若遇 alpha/rank≠1 且各模块不同的 LoKr，染色相对强度可能偏差，但缩放仍正确。
    """
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
                # tucker: 'i j k l, j r, i p -> p r k l'
                return torch.einsum("i j k l, j r, i p -> p r k l", t2, b, a)
            return a @ b
        except Exception:
            return None
    return None


def compute_block_impact(raw, total_blocks, fmt=None):
    """
    计算每个 block 的 impact 分数（0..1），LoKr / LoRA 通用。

    - LoRA: 沿用发布版逻辑——累加该 block 所有权重张量的 L2 范数。
    - LoKr: 等效权重 = w1 ⊗ w2，其 Frobenius 范数 = ‖w1‖·‖w2‖（已数值验证）。
            因此对每个模块取 ‖w1‖·‖w2‖ 作为该模块能量，再按 block 累加。
            ★ w1/w2 可能是「整块」或「低秩分解(a@b)」或「tucker(w2)」，
              必须先按推理逻辑重建出完整 w1/w2 再算范数；
              绝不能直接对 lokr_w1 / lokr_w2_a / lokr_w2_b 单独算范数
              （那既漏项又因 ‖a@b‖≠‖a‖‖b‖ 而错，且跨 block 排序会乱）。
            完全不读 alpha / lokr_rank（避开 inf 陷阱）。

    返回长度 total_blocks 的 list；torch 不可用或失败时返回全 0。
    再做对比度拉伸（[min,max] -> [0,1]），与发布版一致。
    """
    if fmt is None:
        fmt = detect_format(raw)
    energy = [0.0] * total_blocks
    try:
        if fmt == "lokr":
            # 收集每个模块的 block 号（凭任一带 block 号的 lokr_ 键）
            mod_blk = {}
            for key in raw:
                if ".lokr_" not in key:
                    continue
                m = BLOCK_RE.search(key)
                if not m:
                    continue
                idx = int(m.group(1))
                if 0 <= idx < total_blocks:
                    mod_blk[_module_key(key)] = idx
            # 对每个模块重建 w1/w2，能量 = ‖w1‖·‖w2‖
            for mk, idx in mod_blk.items():
                w1 = _rebuild_lokr_factor(raw, mk, "w1")
                w2 = _rebuild_lokr_factor(raw, mk, "w2")
                n1 = _norm(w1) if w1 is not None else None
                n2 = _norm(w2) if w2 is not None else None
                if n1 is not None and n2 is not None:
                    energy[idx] += n1 * n2          # = ‖w1⊗w2‖_F
                elif n1 is not None:                # 只有一块时退化（不应常见）
                    energy[idx] += n1
                elif n2 is not None:
                    energy[idx] += n2
        else:
            for key, tensor in raw.items():
                m = BLOCK_RE.search(key)
                if not m:
                    continue
                idx = int(m.group(1))
                if not (0 <= idx < total_blocks):
                    continue
                if key.endswith(".alpha"):
                    continue
                try:
                    energy[idx] += float(torch.linalg.vector_norm(tensor.float()).item())
                except Exception:
                    try:
                        energy[idx] += float(tensor.abs().sum().item())
                    except Exception:
                        pass

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


def apply_layered_scaling(raw, block_w, type_weight):
    """
    返回 (新state_dict, 已缩放张量数, 格式)。

    LoKr 缩放原则：每个模块只把 factor 乘到 w1 一侧的「一块」张量，
      等效整体缩放 factor（克罗内克性质），避开 factor² 与 inf alpha：
        - 整块 w1:     缩放 .lokr_w1
        - 分解 w1:     只缩放 .lokr_w1_a（缩 a 等于缩 a@b；绝不同时缩 _a 和 _b）
      w2 的任何形式（.lokr_w2 / _a / _b / t2）、alpha、dora_scale 一律原样保留。
    LoRA: 缩放 lora_down（与发布版一致）。
    """
    fmt = detect_format(raw)
    out = {}
    scaled = 0
    if fmt == "lokr":
        # 先确定每个模块「该缩哪个键」：优先整块 .lokr_w1，否则分解的 .lokr_w1_a
        scale_target = {}  # module_key -> 要缩放的具体 key
        has_full_w1 = {}
        for key in raw:
            if key.endswith(".lokr_w1"):
                has_full_w1[_module_key(key)] = key
        for key in raw:
            mk = _module_key(key)
            if mk in has_full_w1:
                scale_target[mk] = has_full_w1[mk]
            elif key.endswith(".lokr_w1_a"):
                scale_target.setdefault(mk, key)
        for key, tensor in raw.items():
            m = BLOCK_RE.search(key)
            if not m:
                out[key] = tensor.clone()
                continue
            idx = int(m.group(1))
            bw = block_w[idx] if idx < len(block_w) else 1.0
            factor = bw * type_weight[classify_submodule(key)]
            mk = _module_key(key)
            if scale_target.get(mk) == key:
                out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
                scaled += 1
            else:
                out[key] = tensor.clone()
        return out, scaled, fmt

    for key, tensor in raw.items():
        m = BLOCK_RE.search(key)
        if not m:
            out[key] = tensor.clone()
            continue
        idx = int(m.group(1))
        bw = block_w[idx] if idx < len(block_w) else 1.0
        factor = bw * type_weight[classify_submodule(key)]
        if fmt == "lora":
            if key.endswith(".lora_down.weight") or key.endswith(".lora_down"):
                out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
                scaled += 1
            else:
                out[key] = tensor.clone()
        else:
            out[key] = tensor.clone()
    return out, scaled, fmt


def _build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs):
    """与发布版 V2 同款：grouped 四段（后段覆盖前段）/ per_block 28 滑块。"""
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


def _v2_segment_inputs():
    """V2 四段参数（与发布版完全一致），供两个节点复用。"""
    return {
        "control_mode": (["grouped", "per_block"], {"default": "grouped"}),
        "seg_motion_blocks": ("STRING", {"default": "0-11"}),
        "seg_motion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "seg_proportion_blocks": ("STRING", {"default": "12-14"}),
        "seg_proportion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "seg_core_blocks": ("STRING", {"default": "15-18"}),
        "seg_core_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "seg_detail_blocks": ("STRING", {"default": "19-27"}),
        "seg_detail_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "w_self_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "w_cross_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "w_mlp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        "w_adaln": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
    }


def _detect_total_blocks(raw):
    max_block = -1
    for k in raw:
        m = BLOCK_RE.search(k)
        if m:
            max_block = max(max_block, int(m.group(1)))
    return max_block + 1 if max_block >= 0 else 0


class AnimaLoKrBlockWeightExperimental:
    """
    [实验性] Anima LoKr/LoRA 分层加载（V2 参数体系 + impact 染色）。
    自动检测格式：LoKr 只缩放 lokr_w1，LoRA 缩放 lora_down。
    impact：LoKr 用 ‖w1‖·‖w2‖，LoRA 用权重范数；均做对比度拉伸。
    共享发布版前端面板（grouped 四段 / per_block 28 滑块 / impact 染色）。

    ⚠️ LoKr 的功能地图不能假设与标准 LoRA 相同（不同数学分解）。
       下方默认四段区间沿用 LoRA 实测值，仅作起点；LoKr 专属功能地图需重新消融测定。
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        req = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (lora_list,),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        }
        req.update(_v2_segment_inputs())
        req["verbose"] = ("BOOLEAN", {"default": False})
        for i in range(TOTAL_BLOCKS_DEFAULT):
            req[f"blk{i:02d}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return {"required": req}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load"
    CATEGORY = "loaders/experimental"
    DESCRIPTION = ("[Experimental] LoKr-aware layered loader (V2 params + impact). "
                   "LoKr scales only lokr_w1 (avoids factor^2 & inf alpha); "
                   "impact = ||w1||*||w2||. | "
                   "[实验性] 支持 LoKr 的分层加载（V2 四段/逐block + impact 染色）："
                   "LoKr 只缩放 lokr_w1，impact 用 ‖w1‖·‖w2‖；完全不碰 alpha/rank。")

    def __init__(self):
        self.impact_cache = {}

    def load(self, model, clip, lora_name, strength_model, strength_clip,
             control_mode,
             seg_motion_blocks, seg_motion_weight,
             seg_proportion_blocks, seg_proportion_weight,
             seg_core_blocks, seg_core_weight,
             seg_detail_blocks, seg_detail_weight,
             w_self_attn, w_cross_attn, w_mlp, w_adaln, verbose,
             **block_kwargs):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = comfy.utils.load_torch_file(lora_path, safe_load=True)

        total_blocks = _detect_total_blocks(raw)
        if total_blocks == 0:
            info = ("[LoKr-Exp] No blocks_N detected; loaded normally without layering. | "
                    "未检测到 blocks_N 结构，已按普通方式加载，未分层。")
            print(info)
            m_out, c_out = comfy.sd.load_lora_for_models(
                model, clip, raw, strength_model, strength_clip)
            return {"ui": {"block_impact": [[]]},
                    "result": (m_out, c_out, info)}

        fmt = detect_format(raw)

        ck = (lora_path, fmt)
        if ck in self.impact_cache and len(self.impact_cache[ck]) == total_blocks:
            impact = self.impact_cache[ck]
        else:
            impact = compute_block_impact(raw, total_blocks, fmt=fmt)
            self.impact_cache[ck] = impact

        seg_specs = [
            (seg_motion_blocks, seg_motion_weight),
            (seg_proportion_blocks, seg_proportion_weight),
            (seg_core_blocks, seg_core_weight),
            (seg_detail_blocks, seg_detail_weight),
        ]
        block_w = _build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs)

        type_weight = {"self_attn": w_self_attn, "cross_attn": w_cross_attn,
                       "mlp": w_mlp, "adaln": w_adaln, "other": 1.0}

        weighted, scaled, fmt = apply_layered_scaling(raw, block_w, type_weight)
        m_out, c_out = comfy.sd.load_lora_for_models(
            model, clip, weighted, strength_model, strength_clip)

        info_lines = [
            f"[LoKr-Exp] {os.path.basename(lora_path)} | format={fmt} | mode={control_mode}",
            f"total_blocks={total_blocks}, scaled_tensors={scaled}",
            "block_weights=" + ",".join(f"{w:g}" for w in block_w),
            f"types: self_attn={w_self_attn} cross_attn={w_cross_attn} "
            f"mlp={w_mlp} adaln={w_adaln}",
        ]
        if fmt == "lokr":
            info_lines.append("(LoKr: 仅缩放 lokr_w1，等效整体 factor；impact=‖w1‖·‖w2‖；未触碰 alpha/rank)")
        if any(v > 0 for v in impact):
            top = sorted(range(total_blocks), key=lambda i: impact[i], reverse=True)[:5]
            info_lines.append("impact_top5_blocks=" +
                              ", ".join(f"{i}:{impact[i]:.2f}" for i in top))
        if verbose:
            info_lines.append("  per-block factors:")
            for i in range(total_blocks):
                info_lines.append(f"    blk{i:02d}: {block_w[i]:g}")
        info = "\n".join(info_lines)
        print(info)

        return {"ui": {"block_impact": [impact]},
                "result": (m_out, c_out, info)}


class AnimaLoKrBlockWeightExportExperimental:
    """[实验性] 把 LoKr/LoRA 分层烘焙成新文件（V2 参数体系）。"""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        req = {
            "lora_name": (lora_list,),
            "output_name": ("STRING", {"default": "anima_lokr_baked_EXPERIMENTAL"}),
            "save_to": (["output", "loras"], {"default": "loras"}),
        }
        req.update(_v2_segment_inputs())
        req["overwrite"] = ("BOOLEAN", {"default": False})
        for i in range(TOTAL_BLOCKS_DEFAULT):
            req[f"blk{i:02d}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return {"required": req}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "export"
    CATEGORY = "loaders/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = ("[Experimental] Bake LoKr/LoRA layering into a file (LoKr scales only lokr_w1). | "
                   "[实验性] 把 LoKr/LoRA 分层烘焙成文件（V2 四段/逐block；LoKr 只缩放 lokr_w1）。")

    def export(self, lora_name, output_name, save_to,
               control_mode,
               seg_motion_blocks, seg_motion_weight,
               seg_proportion_blocks, seg_proportion_weight,
               seg_core_blocks, seg_core_weight,
               seg_detail_blocks, seg_detail_weight,
               w_self_attn, w_cross_attn, w_mlp, w_adaln, overwrite,
               **block_kwargs):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = comfy.utils.load_torch_file(lora_path, safe_load=True)

        orig_meta = {}
        try:
            import struct
            with open(lora_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                orig_meta = json.loads(f.read(n)).get("__metadata__", {}) or {}
        except Exception as e:
            print(f"[LoKr-Exp] 读 metadata 失败（忽略）: {e}")

        total_blocks = _detect_total_blocks(raw)
        if total_blocks == 0:
            msg = "[LoKr-Exp] 未检测到 blocks_N，已中止。"
            print(msg)
            return (msg,)

        seg_specs = [
            (seg_motion_blocks, seg_motion_weight),
            (seg_proportion_blocks, seg_proportion_weight),
            (seg_core_blocks, seg_core_weight),
            (seg_detail_blocks, seg_detail_weight),
        ]
        block_w = _build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs)

        type_weight = {"self_attn": w_self_attn, "cross_attn": w_cross_attn,
                       "mlp": w_mlp, "adaln": w_adaln, "other": 1.0}

        out, scaled, fmt = apply_layered_scaling(raw, block_w, type_weight)

        base_dir = (folder_paths.get_output_directory() if save_to == "output"
                    else folder_paths.get_folder_paths("loras")[0])
        os.makedirs(base_dir, exist_ok=True)
        fname = (output_name.strip() or "anima_lokr_baked_EXPERIMENTAL")
        if not fname.endswith(".safetensors"):
            fname += ".safetensors"
        save_path = os.path.join(base_dir, fname)
        if os.path.exists(save_path) and not overwrite:
            stem = fname[:-len(".safetensors")]
            i = 1
            while os.path.exists(os.path.join(base_dir, f"{stem}_{i}.safetensors")):
                i += 1
            save_path = os.path.join(base_dir, f"{stem}_{i}.safetensors")

        meta = {k: str(v) for k, v in orig_meta.items()}
        meta["anima_lbw_baked"] = "true"
        meta["anima_lbw_experimental"] = "true"
        meta["anima_lbw_format"] = fmt
        meta["anima_lbw_mode"] = control_mode
        meta["anima_lbw_source"] = os.path.basename(lora_path)
        meta["anima_lbw_block_weights"] = ",".join(f"{w:g}" for w in block_w)

        save_file(out, save_path, metadata=meta)
        msg = (f"[LoKr-Exp] 已导出: {save_path}\n"
               f"  format={fmt} mode={control_mode} scaled_tensors={scaled}\n"
               f"  block_weights={meta['anima_lbw_block_weights']}")
        print(msg)
        return (save_path,)


NODE_CLASS_MAPPINGS = {
    "AnimaLoKrBlockWeightExperimental": AnimaLoKrBlockWeightExperimental,
    "AnimaLoKrBlockWeightExportExperimental": AnimaLoKrBlockWeightExportExperimental,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoKrBlockWeightExperimental": "Anima LoKr Block Weight [实验性]",
    "AnimaLoKrBlockWeightExportExperimental": "Anima LoKr Block Weight Export [实验性]",
}
