"""
============================================================================
  实验性分支 (EXPERIMENTAL) — LoKr 分层支持
============================================================================
⚠️ 这是一个实验性节点，独立于正式发布的两个节点，不影响它们。
   仅用于测试对 LoKr (LoRA with Kronecker product) 的分层支持。

与普通 LoRA 的区别
------------------------------------------------------------------------
普通 LoRA 每个模块是 lora_down + lora_up；
LoKr 每个模块是 lokr_w1 + lokr_w2，最终贡献 = lokr_w1 ⊗ lokr_w2（克罗内克积）。

缩放原理（已用真实文件数学验证）
------------------------------------------------------------------------
克罗内克积性质：(c · w1) ⊗ w2 == c · (w1 ⊗ w2)
因此把 factor 只乘到 lokr_w1 一块，等效于整体缩放 factor。
绝不能两块都乘——那会变成 factor²。

重要警告（针对本文件作者的特定 LoKr）
------------------------------------------------------------------------
该 LoKr 训练时用了一个训练器 trick：把 dim/alpha 设成超大值(1000000)交给
Prodigy 自由控制。存成 fp16 后这些值溢出为 inf。
=> 本节点绝不读取/使用 alpha 或 lokr_rank 做任何计算，只对 lokr_w1 做乘法，
   从而完全避开 inf/NaN 问题。

本节点同时兼容普通 LoRA（检测到 lora_down 时按 LoRA 方式处理），
所以它其实是个"通吃"实验版；但请仍把它当实验品，正式用发布版节点。
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


def parse_per_block_weights(spec, total_blocks, default):
    weights = [default] * total_blocks
    spec = (spec or "").strip()
    if not spec:
        return weights
    tokens = [t.strip() for t in re.split(r"[,\n;]+", spec) if t.strip()]
    if not any(":" in t for t in tokens):
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
        try:
            val = float(right.strip())
        except ValueError:
            continue
        left = left.strip()
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


def detect_format(raw):
    """返回 'lokr' / 'lora' / 'unknown'。"""
    has_lokr = any(k.endswith(".lokr_w1") for k in raw)
    has_lora = any(k.endswith(".lora_down.weight") for k in raw)
    if has_lokr:
        return "lokr"
    if has_lora:
        return "lora"
    return "unknown"


def apply_layered_scaling(raw, block_w, type_weight, default_weight):
    """
    返回 (新state_dict, 已缩放张量数, 格式)。
    LoKr: 只缩放 lokr_w1（等效整体 factor，避开 factor² 和 inf alpha）。
    LoRA: 缩放 lora_down（与发布版一致）。
    """
    fmt = detect_format(raw)
    out = {}
    scaled = 0
    for key, tensor in raw.items():
        m = BLOCK_RE.search(key)
        if not m:
            out[key] = tensor.clone()
            continue
        idx = int(m.group(1))
        bw = block_w[idx] if idx < len(block_w) else default_weight
        factor = bw * type_weight[classify_submodule(key)]

        if fmt == "lokr":
            # 只缩放 w1；w2 / alpha / lokr_rank 一律原样保留
            if key.endswith(".lokr_w1"):
                out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
                scaled += 1
            else:
                out[key] = tensor.clone()
        elif fmt == "lora":
            if key.endswith(".lora_down.weight") or key.endswith(".lora_down"):
                out[key] = (tensor.to(torch.float32) * factor).to(tensor.dtype)
                scaled += 1
            else:
                out[key] = tensor.clone()
        else:
            out[key] = tensor.clone()
    return out, scaled, fmt


class AnimaLoKrBlockWeightExperimental:
    """
    [实验性] Anima LoKr/LoRA 分层加载。优先用于测试 LoKr 支持。
    自动检测格式：LoKr 只缩放 lokr_w1，LoRA 缩放 lora_down。
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_mode": (["grouped", "per_block"], {"default": "grouped"}),
                "shallow_blocks": ("STRING", {"default": "0-8"}),
                "shallow_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "middle_blocks": ("STRING", {"default": "9-18"}),
                "middle_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "deep_blocks": ("STRING", {"default": "19-27"}),
                "deep_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "block_weights": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "per_block用。纯序列/索引:值/区间:值"}),
                "w_self_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "w_cross_attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "w_mlp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "w_adaln": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "default_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load"
    CATEGORY = "loaders/experimental"
    DESCRIPTION = "[实验性] 支持 LoKr 的分层加载（只缩放 lokr_w1，避开 factor² 与 inf alpha）。"

    def load(self, model, clip, lora_name, strength_model, strength_clip,
             control_mode, shallow_blocks, shallow_weight, middle_blocks,
             middle_weight, deep_blocks, deep_weight, block_weights,
             w_self_attn, w_cross_attn, w_mlp, w_adaln, default_weight, verbose):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = comfy.utils.load_torch_file(lora_path, safe_load=True)

        max_block = -1
        for k in raw:
            m = BLOCK_RE.search(k)
            if m:
                max_block = max(max_block, int(m.group(1)))
        total_blocks = max_block + 1 if max_block >= 0 else 0
        if total_blocks == 0:
            info = "[LoKr-Exp] 未检测到 blocks_N 结构，按普通方式加载，未分层。"
            print(info)
            m_out, c_out = comfy.sd.load_lora_for_models(model, clip, raw, strength_model, strength_clip)
            return (m_out, c_out, info)

        if control_mode == "per_block":
            block_w = parse_per_block_weights(block_weights, total_blocks, default_weight)
        else:
            ss = parse_block_range(shallow_blocks, total_blocks)
            ms = parse_block_range(middle_blocks, total_blocks)
            ds = parse_block_range(deep_blocks, total_blocks)
            block_w = []
            for i in range(total_blocks):
                block_w.append(deep_weight if i in ds else
                               middle_weight if i in ms else
                               shallow_weight if i in ss else default_weight)

        type_weight = {"self_attn": w_self_attn, "cross_attn": w_cross_attn,
                       "mlp": w_mlp, "adaln": w_adaln, "other": 1.0}

        weighted, scaled, fmt = apply_layered_scaling(raw, block_w, type_weight, default_weight)
        m_out, c_out = comfy.sd.load_lora_for_models(model, clip, weighted, strength_model, strength_clip)

        info = (f"[LoKr-Exp] {os.path.basename(lora_path)} | format={fmt} | "
                f"mode={control_mode} | scaled={scaled}\n"
                f"  block_weights=" + ",".join(f"{w:g}" for w in block_w))
        if fmt == "lokr":
            info += "\n  (LoKr: 仅缩放 lokr_w1，等效整体 factor；未触碰 alpha/rank)"
        print(info)
        return (m_out, c_out, info)


class AnimaLoKrBlockWeightExportExperimental:
    """[实验性] 把 LoKr/LoRA 分层烘焙成新文件。"""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_list,),
                "output_name": ("STRING", {"default": "anima_lokr_baked_EXPERIMENTAL"}),
                "save_to": (["output", "loras"], {"default": "loras"}),
                "control_mode": (["grouped", "per_block"], {"default": "grouped"}),
                "shallow_blocks": ("STRING", {"default": "0-8"}),
                "shallow_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "middle_blocks": ("STRING", {"default": "9-18"}),
                "middle_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "deep_blocks": ("STRING", {"default": "19-27"}),
                "deep_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "block_weights": ("STRING", {"multiline": True, "default": ""}),
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
    CATEGORY = "loaders/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = "[实验性] 把 LoKr/LoRA 分层烘焙成文件（LoKr 只缩放 lokr_w1）。"

    def export(self, lora_name, output_name, save_to, control_mode,
               shallow_blocks, shallow_weight, middle_blocks, middle_weight,
               deep_blocks, deep_weight, block_weights, w_self_attn,
               w_cross_attn, w_mlp, w_adaln, default_weight, overwrite):

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

        max_block = -1
        for k in raw:
            m = BLOCK_RE.search(k)
            if m:
                max_block = max(max_block, int(m.group(1)))
        total_blocks = max_block + 1 if max_block >= 0 else 0
        if total_blocks == 0:
            msg = "[LoKr-Exp] 未检测到 blocks_N，已中止。"
            print(msg)
            return (msg,)

        if control_mode == "per_block":
            block_w = parse_per_block_weights(block_weights, total_blocks, default_weight)
        else:
            ss = parse_block_range(shallow_blocks, total_blocks)
            ms = parse_block_range(middle_blocks, total_blocks)
            ds = parse_block_range(deep_blocks, total_blocks)
            block_w = []
            for i in range(total_blocks):
                block_w.append(deep_weight if i in ds else
                               middle_weight if i in ms else
                               shallow_weight if i in ss else default_weight)

        type_weight = {"self_attn": w_self_attn, "cross_attn": w_cross_attn,
                       "mlp": w_mlp, "adaln": w_adaln, "other": 1.0}

        out, scaled, fmt = apply_layered_scaling(raw, block_w, type_weight, default_weight)

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
        meta["anima_lbw_source"] = os.path.basename(lora_path)
        meta["anima_lbw_block_weights"] = ",".join(f"{w:g}" for w in block_w)

        save_file(out, save_path, metadata=meta)
        msg = (f"[LoKr-Exp] 已导出: {save_path}\n"
               f"  format={fmt} scaled_tensors={scaled}\n"
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
