"""
Anima LoRA Block Weight — 分层加载（三档 + 逐 block）
======================================================
在三档版（shallow/middle/deep）基础上，新增「逐 block 模式」：
可以对 0..27 每一个 block 单独设权重，用于精确定位每个 block 的作用、
找出画风/构图/骨相在你这个具体 LoRA 上真正的分界点。

两种模式由 control_mode 切换：
  - "grouped"  : 三档模式（和原节点一致，简单快速）
  - "per_block": 逐 block 模式（用 block_weights 文本框逐个指定）

子模块类型系数 (self_attn / cross_attn / mlp / adaln) 在两种模式下都生效，
与 block 权重相乘。

放置方法：和原节点相同。可与原节点共存（类名不同）。
  ComfyUI/custom_nodes/某文件夹/__init__.py
  重启后在 loaders 分类下出现 "Anima LoRA Block Weight"。
"""

import os
import re
import folder_paths
import comfy.utils
import comfy.sd


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
    """'all' / '0-8' / '0,3,5' / '0-8,19-27' -> set(索引)"""
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
    """
    解析逐 block 权重。返回长度 total_blocks 的列表。
    支持三种写法，可混用（用逗号或换行分隔）：
      1) 纯数值序列：   1,1,0.5,0.5,...    按顺序对应 block 0,1,2,3...
                        不足的用 default 补齐，超出的忽略。
      2) 索引:值：      0:1.0, 5:0.3, 27:1.2
      3) 区间:值：      0-8:0.3, 9-18:0.7, 19-27:1.0
    规则 2/3 可覆盖规则 1。后写的覆盖先写的。
    空字符串 -> 全部 default。
    """
    weights = [default] * total_blocks
    spec = (spec or "").strip()
    if not spec:
        return weights

    # 统一分隔符：换行、分号都当逗号
    tokens = re.split(r"[,\n;]+", spec)
    tokens = [t.strip() for t in tokens if t.strip()]

    # 先判断是不是「纯数值序列」（没有任何冒号）
    has_colon = any(":" in t for t in tokens)

    if not has_colon:
        # 纯序列模式：按顺序填
        seq = []
        for t in tokens:
            try:
                seq.append(float(t))
            except ValueError:
                pass
        for i in range(min(len(seq), total_blocks)):
            weights[i] = seq[i]
        return weights

    # 含冒号：逐条解析 索引:值 或 区间:值
    for t in tokens:
        if ":" not in t:
            # 混写时，无冒号的裸数值忽略（避免歧义）
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


class AnimaLoRABlockWeight:
    """
    Anima LoRA 分层加载器。支持三档与逐 block 两种控制模式。

    最终缩放系数 = block权重 × 类型系数
      - grouped  模式：block权重来自 shallow/middle/deep 三档
      - per_block模式：block权重来自 block_weights 文本框（逐 block）
      - 类型系数：self_attn / cross_attn / mlp / adaln 各一个乘数

    深度-作用对应（经验起点，以实测为准）：
      浅层 block 0-8   -> 构图/姿态/骨相
      深层 block 19-27 -> 画风/纹理/上色
      self_attn -> 空间结构   mlp -> 画风   cross_attn -> 提示词响应
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

                # --- grouped 模式用 ---
                "shallow_blocks": ("STRING", {"default": "0-8"}),
                "shallow_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "middle_blocks": ("STRING", {"default": "9-18"}),
                "middle_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "deep_blocks": ("STRING", {"default": "19-27"}),
                "deep_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

                # --- per_block 模式用 ---
                # 三种写法见 parse_per_block_weights 的说明。留空=全部用 default_weight。
                "block_weights": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "逐block。写法任选:\n纯序列: 1,1,0.5,...(对应block0,1,2...)\n索引: 0:1.0, 5:0.3\n区间: 0-8:0.3, 19-27:1.0",
                }),

                # --- 两种模式共用 ---
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
    CATEGORY = "loaders"
    DESCRIPTION = "Anima(Cosmos) LoRA 分层加载，支持三档或逐block精细控制，即时生效。"

    def __init__(self):
        self.cache = {}

    def _load_raw(self, lora_path):
        if lora_path in self.cache:
            return self.cache[lora_path]
        data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.cache = {lora_path: data}
        return data

    def load(self, model, clip, lora_name, strength_model, strength_clip,
             control_mode, shallow_blocks, shallow_weight, middle_blocks,
             middle_weight, deep_blocks, deep_weight, block_weights,
             w_self_attn, w_cross_attn, w_mlp, w_adaln, default_weight, verbose):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        raw = self._load_raw(lora_path)

        # 总 block 数
        max_block = -1
        for k in raw.keys():
            m = BLOCK_RE.search(k)
            if m:
                max_block = max(max_block, int(m.group(1)))
        total_blocks = max_block + 1 if max_block >= 0 else 0

        if total_blocks == 0:
            info = ("[AnimaLBW] 未检测到 blocks_N 结构，可能不是 Anima LoRA。"
                    "已按普通方式加载，未做分层。")
            print(info)
            m_out, c_out = comfy.sd.load_lora_for_models(
                model, clip, raw, strength_model, strength_clip)
            return (m_out, c_out, info)

        # 根据模式算出每个 block 的权重表
        if control_mode == "per_block":
            block_w = parse_per_block_weights(block_weights, total_blocks, default_weight)
        else:
            shallow_set = parse_block_range(shallow_blocks, total_blocks)
            middle_set = parse_block_range(middle_blocks, total_blocks)
            deep_set = parse_block_range(deep_blocks, total_blocks)
            block_w = []
            for i in range(total_blocks):
                if i in deep_set:
                    block_w.append(deep_weight)
                elif i in middle_set:
                    block_w.append(middle_weight)
                elif i in shallow_set:
                    block_w.append(shallow_weight)
                else:
                    block_w.append(default_weight)

        type_weight = {
            "self_attn": w_self_attn,
            "cross_attn": w_cross_attn,
            "mlp": w_mlp,
            "adaln": w_adaln,
            "other": 1.0,
        }

        # 施加
        weighted = {}
        applied = 0
        stats = {}
        for key, tensor in raw.items():
            m = BLOCK_RE.search(key)
            if m:
                idx = int(m.group(1))
                bw = block_w[idx] if idx < len(block_w) else default_weight
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
            f"[AnimaLBW] {os.path.basename(lora_path)} | mode={control_mode}",
            f"total_blocks={total_blocks}, scaled_tensors={applied}",
            "block_weights=" + ",".join(f"{w:g}" for w in block_w),
            f"types: self_attn={w_self_attn} cross_attn={w_cross_attn} "
            f"mlp={w_mlp} adaln={w_adaln}",
        ]
        if verbose:
            for idx in sorted(stats):
                info_lines.append(f"  block {idx}: factors={sorted(stats[idx])}")
        info = "\n".join(info_lines)
        print(info)
        return (m_out, c_out, info)


NODE_CLASS_MAPPINGS = {"AnimaLoRABlockWeight": AnimaLoRABlockWeight}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaLoRABlockWeight": "Anima LoRA Block Weight"}
