"""
Anima LoRA Block Weight Export V2 — 把分层缩放烘焙进新文件
=====================================================================================
参数与加载节点一致（四段强度命名 weak/medium/strong/peak + 自动分段开关），
便于把调好的系数原样搬过来烘焙。三格式通吃：kohya/diffusers LoRA + LoKr。
导出的文件用任何普通 LoRA Loader 加载即可，效果 = 这组系数 + strength 1.0。

⚠️ 旧工作流不兼容：四段已从 motion/proportion/core/detail 改名为
   weak/medium/strong/peak。需要旧版请到 GitHub 历史 commit 下载。
"""

import os
import json
import folder_paths
import comfy.utils
from safetensors.torch import save_file

from .anima_common import (
    SEG_NAMES,
    detect_total_blocks, detect_format, compute_block_impact, compute_block_metric,
    apply_layered_scaling, build_block_weights,
    auto_segment as compute_auto_segments, v2_segment_inputs,
)


class AnimaLoRABlockWeightExport:
    """把分层缩放烘焙进新的 LoRA 文件。参数与加载节点一致。"""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        req = {
            "lora_name": (lora_list,),
            "output_name": ("STRING", {"default": "anima_lora_baked"}),
            "save_to": (["output", "loras"], {"default": "loras"}),
        }
        req.update(v2_segment_inputs())
        req["overwrite"] = ("BOOLEAN", {"default": False})
        for i in range(28):
            req[f"blk{i:02d}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return {"required": req}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "export"
    CATEGORY = "loaders"
    OUTPUT_NODE = True
    DESCRIPTION = ("Bake layered scaling into a new LoRA file (kohya/diffusers/LoKr), "
                   "with auto-segment support. | "
                   "把分层缩放烘焙进新文件（通吃 kohya/diffusers/LoKr，支持自动分段），供普通 Loader 加载。")

    def export(self, lora_name, output_name, save_to, control_mode, auto_segment, segment_metric,
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
            print(f"[AnimaExport] 读取原 metadata 失败（忽略）: {e}")

        total_blocks = detect_total_blocks(raw)
        if total_blocks == 0:
            msg = "[AnimaExport] 未检测到 blocks_N 结构，已中止。"
            print(msg)
            return (msg,)

        fmt = detect_format(raw)

        # 自动分段（与加载节点一致：开关打开则用实测强度现算区间）
        auto_seg_ranges = {}
        if auto_segment:
            impact = compute_block_metric(raw, total_blocks, metric=segment_metric, fmt=fmt)
            auto_seg_ranges, _ = compute_auto_segments(impact, total_blocks)
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

        out, baked, fmt = apply_layered_scaling(raw, block_w, type_weight)

        base_dir = (folder_paths.get_output_directory() if save_to == "output"
                    else folder_paths.get_folder_paths("loras")[0])
        os.makedirs(base_dir, exist_ok=True)
        fname = output_name.strip() or "anima_lora_baked"
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
        meta["anima_lbw_format"] = fmt
        meta["anima_lbw_source"] = os.path.basename(lora_path)
        meta["anima_lbw_mode"] = control_mode
        meta["anima_lbw_block_weights"] = ",".join(f"{w:g}" for w in block_w)
        meta["anima_lbw_type_weights"] = (
            f"self_attn={w_self_attn},cross_attn={w_cross_attn},"
            f"mlp={w_mlp},adaln={w_adaln}")

        save_file(out, save_path, metadata=meta)

        msg = (f"[AnimaExport] 已导出: {save_path}\n"
               f"  source={os.path.basename(lora_path)} format={fmt} mode={control_mode} "
               f"baked_tensors={baked}\n"
               f"  block_weights={meta['anima_lbw_block_weights']}")
        print(msg)
        return (save_path,)


NODE_CLASS_MAPPINGS = {"AnimaLoRABlockWeightExport": AnimaLoRABlockWeightExport}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaLoRABlockWeightExport": "Anima LoRA Block Weight Export V2"}
