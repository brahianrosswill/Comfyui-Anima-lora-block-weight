"""
Anima LoRA Block Weight V2 — 逐层滑块（方案B：原生控件兜底 + 前端美化 + impact 染色）
=====================================================================================
设计哲学（借鉴成熟项目 comfyUI-Realtime-Lora 的稳健做法）：

  「原生控件做底座，前端 JS 做皮肤」
  - 后端为 28 个 block 各提供一个原生 FLOAT 滑块 (blk00..blk27)。
    即使前端 JS 完全没加载，你也有 28 个可用的原生滑块 —— 不会出现空白。
  - 前端 JS（web/anima_block_weight.js）把这些原生 widget 收拢，
    重绘成紧凑的「方块开关 + 滑块 + 数字」逐层 UI，并按 impact 分数染色。
  - JS 只是美化与排版；它读写的就是这些原生 widget 的 value，
    所以 JS 挂掉也不影响出图，底座永远在。

impact 染色：
  - 节点运行时用 torch 计算每个 block 的 LoRA 权重 L2 范数（跨该 block 所有张量），
    归一化成 0..1 的"影响力分数"，作为 UI 输出 ui.block_impact 传给前端。
  - 前端据此给每行染色：蓝(低)→青→黄→红(高)，让你一眼看出哪些 block 最关键。
  - 该分数也写进 info 文本输出，便于无 JS 时查看。

control_mode：
  - "grouped"     : 四段模式（基于实测 motion/proportion/core/detail，区间可改）
  - "per_block"   : 逐 block 模式（28 个原生滑块；JS 会美化成图形界面）
"""

import os
import re
import folder_paths
import comfy.utils
import comfy.sd

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

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


def parse_block_range(range_str: str, total_blocks: int):
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


def compute_block_impact(raw, total_blocks):
    """
    计算每个 block 的 impact 分数（0..1）。
    方法：对每个 block，累加其所有 lora_down/lora_up 权重张量的 L2 范数（Frobenius），
    作为该 block 的"能量"，再用全局最大值归一化。
    返回长度 total_blocks 的 list；torch 不可用或失败时返回全 0。
    """
    if not _HAS_TORCH:
        return [0.0] * total_blocks
    energy = [0.0] * total_blocks
    try:
        for key, tensor in raw.items():
            m = BLOCK_RE.search(key)
            if not m:
                continue
            idx = int(m.group(1))
            if not (0 <= idx < total_blocks):
                continue
            if key.endswith(".alpha"):
                continue
            # 只统计权重张量
            try:
                t = tensor.float()
                energy[idx] += float(torch.linalg.vector_norm(t).item())
            except Exception:
                # 退化：用绝对值和
                try:
                    energy[idx] += float(tensor.abs().sum().item())
                except Exception:
                    pass
        mx = max(energy) if energy else 0.0
        mn = min(energy) if energy else 0.0
        if mx <= 0:
            return [0.0] * total_blocks
        # 对比度拉伸：LoRA 各 block 的绝对范数通常都在同一量级
        # （如 0.93~1.0），若简单除以最大值，差异会被压扁、颜色无区分度。
        # 改为把 [min, max] 线性拉伸到 [0, 1]，让相对差异充分显现。
        span = mx - mn
        if span <= 1e-9:
            return [0.5] * total_blocks
        return [(e - mn) / span for e in energy]
    except Exception:
        return [0.0] * total_blocks


class AnimaLoRABlockWeightV2:
    """
    Anima LoRA 分层加载器 V2（原生滑块兜底 + 前端美化 + impact 染色）。

    实测功能地图（Anima 画风 LoRA，固定种子单变量实验，详见 README 研究记录）：
      block 0-11  动作/体型尺寸的提示词服从度
      block 12-14 体型比例/敦实度的先验服从度
      block 15-18 LoRA 核心表达段（骨相+比例+材质，信息密度最高）
      block 19-27 全局精修（饱和度/纯度等表层属性）
      w_mlp 影响骨相最大(兼带画风) · w_cross_attn 最安全 · w_adaln 整体性影响
    注：各段间存在串扰，无法干净分离（架构级"先验博弈"特性）。
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

            "control_mode": (["grouped", "per_block"], {"default": "grouped"}),

            # --- grouped 四段（区间可自定义）---
            "seg_motion_blocks": ("STRING", {"default": "0-11"}),
            "seg_motion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "seg_proportion_blocks": ("STRING", {"default": "12-14"}),
            "seg_proportion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "seg_core_blocks": ("STRING", {"default": "15-18"}),
            "seg_core_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "seg_detail_blocks": ("STRING", {"default": "19-27"}),
            "seg_detail_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

            # --- 子模块类型系数（两种模式共用）---
            "w_self_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "w_cross_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "w_mlp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "w_adaln": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "verbose": ("BOOLEAN", {"default": False}),
        }
        # per_block 模式：28 个原生 block 滑块。前端 JS 会把它们重绘为紧凑 UI。
        # 即使没有 JS，它们也是可用的原生滑块（方案B的兜底底座）。
        for i in range(TOTAL_BLOCKS_DEFAULT):
            req[f"blk{i:02d}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return {"required": req}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = ("Anima(Cosmos) LoRA layered loader V2: grouped 4-segment / per-block sliders "
                   "(native sliders + JS panel + impact coloring), applied live. | "
                   "Anima(Cosmos) LoRA 分层加载 V2：四段 / 逐 block 滑块（原生滑块+JS美化+impact染色），即时生效。")

    def __init__(self):
        self.cache = {}
        self.impact_cache = {}

    def _load_raw(self, lora_path):
        if lora_path in self.cache:
            return self.cache[lora_path]
        data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.cache = {lora_path: data}
        return data

    def load(self, model, clip, lora_name, strength_model, strength_clip,
             control_mode,
             seg_motion_blocks, seg_motion_weight,
             seg_proportion_blocks, seg_proportion_weight,
             seg_core_blocks, seg_core_weight,
             seg_detail_blocks, seg_detail_weight,
             w_self_attn, w_cross_attn, w_mlp, w_adaln, verbose,
             **block_kwargs):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = self._load_raw(lora_path)

        max_block = -1
        for k in raw.keys():
            m = BLOCK_RE.search(k)
            if m:
                max_block = max(max_block, int(m.group(1)))
        total_blocks = max_block + 1 if max_block >= 0 else 0

        if total_blocks == 0:
            info = ("[AnimaLBW V2] No blocks_N structure detected; may not be an Anima LoRA. "
                    "Loaded normally without layering. | "
                    "未检测到 blocks_N 结构，可能不是 Anima LoRA。已按普通方式加载，未做分层。")
            print(info)
            m_out, c_out = comfy.sd.load_lora_for_models(
                model, clip, raw, strength_model, strength_clip)
            return {"ui": {"block_impact": [[]]},
                    "result": (m_out, c_out, info)}

        # impact 分数（缓存，避免每次重算）
        if lora_path in self.impact_cache and len(self.impact_cache[lora_path]) == total_blocks:
            impact = self.impact_cache[lora_path]
        else:
            impact = compute_block_impact(raw, total_blocks)
            self.impact_cache[lora_path] = impact

        # 算每 block 权重
        if control_mode == "per_block":
            block_w = []
            for i in range(total_blocks):
                v = block_kwargs.get(f"blk{i:02d}", 1.0)
                try:
                    block_w.append(float(v))
                except (ValueError, TypeError):
                    block_w.append(1.0)
        else:  # grouped 四段
            segs = [
                (parse_block_range(seg_detail_blocks, total_blocks), seg_detail_weight),
                (parse_block_range(seg_core_blocks, total_blocks), seg_core_weight),
                (parse_block_range(seg_proportion_blocks, total_blocks), seg_proportion_weight),
                (parse_block_range(seg_motion_blocks, total_blocks), seg_motion_weight),
            ]
            block_w = []
            for i in range(total_blocks):
                val = 1.0
                for sset, sw in segs:
                    if i in sset:
                        val = sw
                block_w.append(val)

        type_weight = {
            "self_attn": w_self_attn,
            "cross_attn": w_cross_attn,
            "mlp": w_mlp,
            "adaln": w_adaln,
            "other": 1.0,
        }

        weighted = {}
        applied = 0
        stats = {}
        for key, tensor in raw.items():
            m = BLOCK_RE.search(key)
            if m:
                idx = int(m.group(1))
                bw = block_w[idx] if idx < len(block_w) else 1.0
                tw = type_weight[classify_submodule(key)]
                factor = bw * tw
                if key.endswith(".alpha"):
                    weighted[key] = tensor
                else:
                    weighted[key] = tensor * factor
                    applied += 1
                    if verbose:
                        stats.setdefault(idx, set()).add(round(factor, 3))
            else:
                weighted[key] = tensor

        m_out, c_out = comfy.sd.load_lora_for_models(
            model, clip, weighted, strength_model, strength_clip)

        info_lines = [
            f"[AnimaLBW V2] {os.path.basename(lora_path)} | mode={control_mode}",
            f"total_blocks={total_blocks}, scaled_tensors={applied}",
            "block_weights=" + ",".join(f"{w:g}" for w in block_w),
            f"types: self_attn={w_self_attn} cross_attn={w_cross_attn} "
            f"mlp={w_mlp} adaln={w_adaln}",
        ]
        if any(v > 0 for v in impact):
            top = sorted(range(total_blocks), key=lambda i: impact[i], reverse=True)[:5]
            info_lines.append("impact_top5_blocks=" +
                              ", ".join(f"{i}:{impact[i]:.2f}" for i in top))
        if verbose:
            for idx in sorted(stats):
                info_lines.append(f"  block {idx}: factors={sorted(stats[idx])}")
        info = "\n".join(info_lines)
        print(info)

        # ui.block_impact 传给前端染色（长度 = total_blocks 的分数列表）
        return {"ui": {"block_impact": [impact]},
                "result": (m_out, c_out, info)}


NODE_CLASS_MAPPINGS = {"AnimaLoRABlockWeightV2": AnimaLoRABlockWeightV2}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaLoRABlockWeightV2": "Anima LoRA Block Weight V2"}
