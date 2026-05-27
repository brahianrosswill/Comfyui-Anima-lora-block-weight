"""
Anima LoRA Block Weight Exporter — 分层导出节点
===============================================
把分层缩放「烘焙」进一个新的 .safetensors 文件。

与主加载节点 AnimaLoRABlockWeight 的区别：
  - 主节点：运行时分层，临时缩放后喂给模型，不产生文件（用于调试）
  - 本节点：把同样的分层缩放固化成一个新 LoRA 文件（用于固化/分享）

推荐流程：先用主节点反复试出满意的系数 → 把同一组系数填进本节点导出成品。
导出的文件用任何普通 LoRA Loader 加载即可，效果 = 主节点里那组系数 + strength 1.0。

技术说明：
  缩放系数 factor = block权重 × 子模块类型系数。
  factor 直接乘到每个模块的 lora_down 权重上（单边缩放，
  数学等价于运行时 tensor*factor）；alpha 保持不变。
  保留原 LoRA 的 __metadata__，并追加一条导出记录。
"""

import os
import re
import json
import torch
import folder_paths
import comfy.utils
from safetensors.torch import save_file


BLOCK_RE = re.compile(r"blocks[_.](\d+)[_.]")


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


def parse_per_block_weights(spec: str, total_blocks: int, default: float):
    weights = [default] * total_blocks
    spec = (spec or "").strip()
    if not spec:
        return weights
    tokens = re.split(r"[,\n;]+", spec)
    tokens = [t.strip() for t in tokens if t.strip()]
    has_colon = any(":" in t for t in tokens)
    if not has_colon:
        seq = []
        for t in tokens:
            try:
                seq.append(float(t))
            except ValueError:
                pass
        for i in range(min(len(seq), total_blocks)):
            weights[i] = seq[i]
        return weights
    for t in tokens:
        if ":" not in t:
            continue
        left, right = t.split(":", 1)
        left, right = left.strip(), right.strip()
        try:
            val = float(right)
        except ValueError:
            continue
        if "-" in left:
            a, b = left.split("-", 1)
            try:
                a, b = int(a), int(b)
                for i in range(min(a, b), max(a, b) + 1):
                    if 0 <= i < total_blocks:
                        weights[i] = val
            except ValueError:
                pass
        else:
            try:
                i = int(left)
                if 0 <= i < total_blocks:
                    weights[i] = val
            except ValueError:
                pass
    return weights


class AnimaLoRABlockWeightExport:
    """
    把分层缩放烘焙进新的 LoRA 文件。参数与主节点一致，便于把调好的系数原样搬过来。
    输出文件保存到 ComfyUI/output/<output_name>.safetensors（或 loras 目录，见 save_to）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_list,),
                "output_name": ("STRING", {"default": "anima_lora_baked"}),
                "save_to": (["output", "loras"], {"default": "loras"}),

                "control_mode": (["grouped", "per_block"], {"default": "grouped"}),

                "shallow_blocks": ("STRING", {"default": "0-8"}),
                "shallow_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "middle_blocks": ("STRING", {"default": "9-18"}),
                "middle_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "deep_blocks": ("STRING", {"default": "19-27"}),
                "deep_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

                "block_weights": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "per_block模式用。纯序列/索引:值/区间:值，见说明书",
                }),

                "w_self_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "w_cross_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "w_mlp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "w_adaln": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "default_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "export"
    CATEGORY = "loaders"
    OUTPUT_NODE = True
    DESCRIPTION = "把分层缩放烘焙进新的 Anima LoRA 文件，供普通 Loader 直接加载。"

    def export(self, lora_name, output_name, save_to, control_mode,
               shallow_blocks, shallow_weight, middle_blocks, middle_weight,
               deep_blocks, deep_weight, block_weights, w_self_attn,
               w_cross_attn, w_mlp, w_adaln, default_weight, overwrite):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # 读取原始 metadata（safetensors header 里的 __metadata__）
        orig_meta = {}
        try:
            import struct
            with open(lora_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
                orig_meta = hdr.get("__metadata__", {}) or {}
        except Exception as e:
            print(f"[AnimaExport] 读取原 metadata 失败（忽略）: {e}")

        # 总 block 数
        max_block = -1
        for k in raw.keys():
            m = BLOCK_RE.search(k)
            if m:
                max_block = max(max_block, int(m.group(1)))
        total_blocks = max_block + 1 if max_block >= 0 else 0
        if total_blocks == 0:
            msg = "[AnimaExport] 未检测到 blocks_N 结构，可能不是 Anima LoRA，已中止。"
            print(msg)
            return (msg,)

        # 每 block 权重
        if control_mode == "per_block":
            block_w = parse_per_block_weights(block_weights, total_blocks, default_weight)
        else:
            ss = parse_block_range(shallow_blocks, total_blocks)
            ms = parse_block_range(middle_blocks, total_blocks)
            ds = parse_block_range(deep_blocks, total_blocks)
            block_w = []
            for i in range(total_blocks):
                if i in ds:
                    block_w.append(deep_weight)
                elif i in ms:
                    block_w.append(middle_weight)
                elif i in ss:
                    block_w.append(shallow_weight)
                else:
                    block_w.append(default_weight)

        type_weight = {
            "self_attn": w_self_attn, "cross_attn": w_cross_attn,
            "mlp": w_mlp, "adaln": w_adaln, "other": 1.0,
        }

        # 烘焙：factor 乘进 lora_down，alpha 不动
        out = {}
        baked = 0
        for key, tensor in raw.items():
            m = BLOCK_RE.search(key)
            if not m:
                out[key] = tensor.clone()
                continue
            idx = int(m.group(1))
            factor = block_w[idx] * type_weight[classify_submodule(key)]
            if key.endswith(".lora_down.weight") or key.endswith(".lora_down"):
                out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
                baked += 1
            else:
                # lora_up / alpha / 其它：原样保留
                out[key] = tensor.clone()

        # 输出路径
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

        # metadata：保留原始 + 追加导出记录
        meta = {k: str(v) for k, v in orig_meta.items()}
        meta["anima_lbw_baked"] = "true"
        meta["anima_lbw_source"] = os.path.basename(lora_path)
        meta["anima_lbw_mode"] = control_mode
        meta["anima_lbw_block_weights"] = ",".join(f"{w:g}" for w in block_w)
        meta["anima_lbw_type_weights"] = (
            f"self_attn={w_self_attn},cross_attn={w_cross_attn},"
            f"mlp={w_mlp},adaln={w_adaln}")

        save_file(out, save_path, metadata=meta)

        msg = (f"[AnimaExport] 已导出: {save_path}\n"
               f"  source={os.path.basename(lora_path)} mode={control_mode} "
               f"baked_down_tensors={baked}\n"
               f"  block_weights={meta['anima_lbw_block_weights']}")
        print(msg)
        return (save_path,)


NODE_CLASS_MAPPINGS = {"AnimaLoRABlockWeightExport": AnimaLoRABlockWeightExport}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaLoRABlockWeightExport": "Anima LoRA Block Weight Export"}
