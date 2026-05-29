"""
Anima LoRA Block Weight V2 — 逐层滑块（原生控件兜底 + 前端美化 + impact 染色 + 自动分段）
=====================================================================================
设计哲学（借鉴 comfyUI-Realtime-Lora 的稳健做法）：「原生控件做底座，前端 JS 做皮肤」
- 后端为 28 个 block 各提供一个原生 FLOAT 滑块 (blk00..blk27)，JS 挂了也有兜底。
- 前端 JS 把原生 widget 收拢重绘成紧凑 UI，并按 impact 分数染色。

★ 通用版要点（本版本）：
- 四段改为按"实测强度档位"命名：seg_weak / seg_medium / seg_strong / seg_peak
  （旧版是 motion/proportion/core/detail，绑定了 Anima 功能假设；但不同 LoRA 强区
   位置不同——画风在中部、角色可能在尾部——旧名名不副实，故改为中性强度名。）
- 新增 auto_segment 开关：打开后，后端用实测 impact 强度按"分位法"自动把 28 个 block
  分成 weak/medium/strong/peak 四段（段内连续、段间可不连续），并通过 ui 回传，
  让前端四个区间框显示出来。开关打开时手填的四段区间会被忽略（以实测为准）。
- 三格式通吃：kohya LoRA(.lora_down) / diffusers LoRA(.lora_A) / LoKr。

control_mode：
- "grouped"   ：四段模式（weak/medium/strong/peak，区间可改或自动分段）
- "per_block" ：逐 block 模式（28 个原生滑块；JS 美化成图形界面）

⚠️ 旧工作流不兼容：四段参数已从 motion/proportion/core/detail 改名为
   weak/medium/strong/peak。需要旧版请到 GitHub 历史 commit 下载。
"""

import os
import folder_paths
import comfy.utils
import comfy.sd

from .anima_common import (
    TOTAL_BLOCKS_DEFAULT, SEG_NAMES,
    detect_total_blocks, detect_format, compute_block_impact, compute_block_metric,
    apply_layered_scaling, build_block_weights,
    auto_segment as compute_auto_segments, v2_segment_inputs,
    parse_block_range,
)


class AnimaLoRABlockWeightV2:
    """
    Anima LoRA 分层加载器 V2（原生滑块兜底 + 前端美化 + impact 染色 + 自动分段）。

    四段按实测强度档位命名（weak/medium/strong/peak）。可手填区间，
    也可打开 auto_segment 让节点按当前 LoRA 的实测强度自动分段。
    实测背景：Anima 画风 LoRA 的高能量段通常在中部（约 12-18），但这只是
    "权重能量"分布，不等于功能；且不同 LoRA（尤其角色）强区位置差异很大，
    所以建议用 auto_segment 针对每个 LoRA 现算，而非套固定区间。
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
        req.update(v2_segment_inputs())
        req["verbose"] = ("BOOLEAN", {"default": False})
        for i in range(TOTAL_BLOCKS_DEFAULT):
            req[f"blk{i:02d}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return {"required": req}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = ("Anima(Cosmos) LoRA layered loader V2: 4-segment (by measured strength: "
                   "weak/medium/strong/peak) / per-block, with auto-segment & impact coloring. "
                   "Supports kohya/diffusers LoRA & LoKr. | "
                   "Anima(Cosmos) LoRA 分层加载 V2：按实测强度的四段(弱/中/强/峰)或逐block，"
                   "支持自动分段与 impact 染色，通吃 kohya/diffusers LoRA 与 LoKr。")

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
             control_mode, auto_segment, segment_metric, segment_method,
             seg_1_blocks, seg_1_weight,
             seg_2_blocks, seg_2_weight,
             seg_3_blocks, seg_3_weight,
             seg_4_blocks, seg_4_weight,
             w_self_attn, w_cross_attn, w_mlp, w_adaln, verbose,
             **block_kwargs):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = self._load_raw(lora_path)

        total_blocks = detect_total_blocks(raw)
        if total_blocks == 0:
            info = ("[AnimaLBW V2] No blocks_N structure detected; loaded normally without layering. | "
                    "未检测到 blocks_N 结构，已按普通方式加载，未做分层。")
            print(info)
            m_out, c_out = comfy.sd.load_lora_for_models(
                model, clip, raw, strength_model, strength_clip)
            return {"ui": {"block_impact": [[]], "auto_segments": [{}]},
                    "result": (m_out, c_out, info)}

        fmt = detect_format(raw)

        ck = (lora_path, fmt, segment_metric)
        if ck in self.impact_cache and len(self.impact_cache[ck]) == total_blocks:
            impact = self.impact_cache[ck]
        else:
            impact = compute_block_metric(raw, total_blocks, metric=segment_metric, fmt=fmt)
            self.impact_cache[ck] = impact

        # 自动分段：打开开关则用实测强度现算四段区间，覆盖手填值
        auto_seg_ranges = {}
        if auto_segment:
            auto_seg_ranges, _tier = compute_auto_segments(impact, total_blocks, method=segment_method)
            wk = auto_seg_ranges.get("seg_1", seg_1_blocks)
            md = auto_seg_ranges.get("seg_2", seg_2_blocks)
            st = auto_seg_ranges.get("seg_3", seg_3_blocks)
            pk = auto_seg_ranges.get("seg_4", seg_4_blocks)
        else:
            wk, md, st, pk = seg_1_blocks, seg_2_blocks, seg_3_blocks, seg_4_blocks

        seg_specs = [
            (wk, seg_1_weight),
            (md, seg_2_weight),
            (st, seg_3_weight),
            (pk, seg_4_weight),
        ]
        block_w = build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs)

        type_weight = {"self_attn": w_self_attn, "cross_attn": w_cross_attn,
                       "mlp": w_mlp, "adaln": w_adaln, "other": 1.0}

        weighted, applied, fmt = apply_layered_scaling(raw, block_w, type_weight)
        m_out, c_out = comfy.sd.load_lora_for_models(
            model, clip, weighted, strength_model, strength_clip)

        info_lines = [
            f"[AnimaLBW V2] {os.path.basename(lora_path)} | format={fmt} | mode={control_mode}",
            f"total_blocks={total_blocks}, scaled_tensors={applied}",
            "block_weights=" + ",".join(f"{w:g}" for w in block_w),
            f"types: self_attn={w_self_attn} cross_attn={w_cross_attn} "
            f"mlp={w_mlp} adaln={w_adaln}",
        ]
        if auto_segment and auto_seg_ranges:
            info_lines.append("auto_segment: " +
                              " | ".join(f"{n.replace('seg_','')}={auto_seg_ranges.get(n,'')}"
                                         for n in SEG_NAMES))
        if any(v > 0 for v in impact):
            top = sorted(range(total_blocks), key=lambda i: impact[i], reverse=True)[:5]
            info_lines.append("impact_top5_blocks=" +
                              ", ".join(f"{i}:{impact[i]:.2f}" for i in top))
        if verbose:
            for i in range(total_blocks):
                info_lines.append(f"  blk{i:02d}: {block_w[i]:g}")
        info = "\n".join(info_lines)
        print(info)

        return {"ui": {"block_impact": [impact], "auto_segments": [auto_seg_ranges]},
                "result": (m_out, c_out, info)}


NODE_CLASS_MAPPINGS = {"AnimaLoRABlockWeightV2": AnimaLoRABlockWeightV2}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaLoRABlockWeightV2": "Anima LoRA Block Weight V2"}
