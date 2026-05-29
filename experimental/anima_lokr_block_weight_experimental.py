"""
============================================================================
  实验性分支 (EXPERIMENTAL) — LoKr 分层支持
============================================================================
⚠️ 实验性节点，以 try/except 安全加载；即使出错也不影响发布版。

★ 本版本已与发布版共用同一份核心逻辑（../anima_common.py）：
  三格式通吃（kohya/diffusers LoRA + LoKr）、四段强度命名（weak/medium/strong/peak）、
  自动分段、impact 染色全部来自公共模块，保证与发布版同步、不再各写一份。

LoKr 关键点（核心数学在公共模块里，已在真实文件验证）：
- 缩放：只把 factor 乘到 w1 一侧的一块（整块 lokr_w1 或分解的 lokr_w1_a），
  等效整体缩放；绝不两块都乘（否则 factor²），绝不碰 w2 / alpha / dora_scale。
- impact：等效权重范数 = ‖w1‖·‖w2‖（先按推理逻辑重建 w1/w2，含分解/tucker）。
- 完全不读 alpha/rank，天然避开某些 LoKr 的 inf alpha 陷阱（与 alpha 类型无关）。

★ 这个实验版"数学验证通过，但没有做出图实测"，所以保持实验状态，不并入发布版。
"""

import os
import json
import folder_paths
import comfy.utils
import comfy.sd
from safetensors.torch import save_file

from ..anima_common import (
    TOTAL_BLOCKS_DEFAULT, SEG_NAMES,
    detect_total_blocks, detect_format, compute_block_impact, compute_block_metric,
    apply_layered_scaling, build_block_weights,
    auto_segment as compute_auto_segments, v2_segment_inputs,
)


class AnimaLoKrBlockWeightExperimental:
    """[实验性] LoKr/LoRA 分层加载（与发布版同参数体系 + 自动分段 + impact 染色）。"""

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
    CATEGORY = "loaders/experimental"
    DESCRIPTION = ("[Experimental] LoKr-aware layered loader (shared core w/ release: "
                   "auto-segment, impact, kohya/diffusers/LoKr). | "
                   "[实验性] 支持 LoKr 的分层加载（与发布版共用核心：自动分段/impact/三格式通吃）。")

    def __init__(self):
        self.impact_cache = {}

    def load(self, model, clip, lora_name, strength_model, strength_clip,
             control_mode, auto_segment, segment_metric, segment_method,
             seg_1_blocks, seg_1_weight,
             seg_2_blocks, seg_2_weight,
             seg_3_blocks, seg_3_weight,
             seg_4_blocks, seg_4_weight,
             w_self_attn, w_cross_attn, w_mlp, w_adaln, verbose,
             **block_kwargs):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = comfy.utils.load_torch_file(lora_path, safe_load=True)

        total_blocks = detect_total_blocks(raw)
        if total_blocks == 0:
            info = ("[LoKr-Exp] No blocks_N detected; loaded normally. | "
                    "未检测到 blocks_N，已按普通方式加载，未分层。")
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

        auto_seg_ranges = {}
        if auto_segment:
            auto_seg_ranges, _ = compute_auto_segments(impact, total_blocks, method=segment_method)
            wk = auto_seg_ranges.get("seg_1", seg_1_blocks)
            md = auto_seg_ranges.get("seg_2", seg_2_blocks)
            st = auto_seg_ranges.get("seg_3", seg_3_blocks)
            pk = auto_seg_ranges.get("seg_4", seg_4_blocks)
        else:
            wk, md, st, pk = seg_1_blocks, seg_2_blocks, seg_3_blocks, seg_4_blocks

        seg_specs = [(wk, seg_1_weight), (md, seg_2_weight),
                     (st, seg_3_weight), (pk, seg_4_weight)]
        block_w = build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs)
        type_weight = {"self_attn": w_self_attn, "cross_attn": w_cross_attn,
                       "mlp": w_mlp, "adaln": w_adaln, "other": 1.0}

        weighted, scaled, fmt = apply_layered_scaling(raw, block_w, type_weight)
        m_out, c_out = comfy.sd.load_lora_for_models(
            model, clip, weighted, strength_model, strength_clip)

        info_lines = [
            f"[LoKr-Exp] {os.path.basename(lora_path)} | format={fmt} | mode={control_mode}",
            f"total_blocks={total_blocks}, scaled_tensors={scaled}",
            "block_weights=" + ",".join(f"{w:g}" for w in block_w),
            f"types: self_attn={w_self_attn} cross_attn={w_cross_attn} mlp={w_mlp} adaln={w_adaln}",
        ]
        if fmt == "lokr":
            info_lines.append("(LoKr: 仅缩放 w1 一块，等效整体 factor；impact=‖w1‖·‖w2‖；未碰 alpha/rank)")
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


class AnimaLoKrBlockWeightExportExperimental:
    """[实验性] 把 LoKr/LoRA 分层烘焙成新文件（与加载节点同参数）。"""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        req = {
            "lora_name": (lora_list,),
            "output_name": ("STRING", {"default": "anima_lokr_baked_EXPERIMENTAL"}),
            "save_to": (["output", "loras"], {"default": "loras"}),
        }
        req.update(v2_segment_inputs())
        req["overwrite"] = ("BOOLEAN", {"default": False})
        for i in range(TOTAL_BLOCKS_DEFAULT):
            req[f"blk{i:02d}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return {"required": req}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "export"
    CATEGORY = "loaders/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = ("[Experimental] Bake LoKr/LoRA layering into a file (shared core). | "
                   "[实验性] 把 LoKr/LoRA 分层烘焙成文件（与发布版共用核心逻辑）。")

    def export(self, lora_name, output_name, save_to, control_mode, auto_segment, segment_metric, segment_method,
               seg_1_blocks, seg_1_weight,
               seg_2_blocks, seg_2_weight,
               seg_3_blocks, seg_3_weight,
               seg_4_blocks, seg_4_weight,
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

        total_blocks = detect_total_blocks(raw)
        if total_blocks == 0:
            msg = "[LoKr-Exp] 未检测到 blocks_N，已中止。"
            print(msg)
            return (msg,)

        fmt = detect_format(raw)

        auto_seg_ranges = {}
        if auto_segment:
            impact = compute_block_metric(raw, total_blocks, metric=segment_metric, fmt=fmt)
            auto_seg_ranges, _ = compute_auto_segments(impact, total_blocks, method=segment_method)
            wk = auto_seg_ranges.get("seg_1", seg_1_blocks)
            md = auto_seg_ranges.get("seg_2", seg_2_blocks)
            st = auto_seg_ranges.get("seg_3", seg_3_blocks)
            pk = auto_seg_ranges.get("seg_4", seg_4_blocks)
        else:
            wk, md, st, pk = seg_1_blocks, seg_2_blocks, seg_3_blocks, seg_4_blocks

        seg_specs = [(wk, seg_1_weight), (md, seg_2_weight),
                     (st, seg_3_weight), (pk, seg_4_weight)]
        block_w = build_block_weights(control_mode, total_blocks, seg_specs, block_kwargs)
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
    "AnimaLoKrBlockWeightExperimental": "Anima LoKr Block Weight [Experimental]",
    "AnimaLoKrBlockWeightExportExperimental": "Anima LoKr Block Weight Export [Experimental]",
}
