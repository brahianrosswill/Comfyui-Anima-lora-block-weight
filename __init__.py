"""
Anima LoRA Block Weight — ComfyUI custom node
为 Anima (NVIDIA Cosmos 架构) 的 LoRA 提供分层（block weight）控制。
导出两个节点：
  - AnimaLoRABlockWeight        分层加载（运行时缩放，支持调试）
  - AnimaLoRABlockWeightExport  分层导出（烘焙成新的 .safetensors）
"""

from .anima_lora_block_weight import (
    NODE_CLASS_MAPPINGS as _M1,
    NODE_DISPLAY_NAME_MAPPINGS as _D1,
)
from .anima_lora_block_weight_export import (
    NODE_CLASS_MAPPINGS as _M2,
    NODE_DISPLAY_NAME_MAPPINGS as _D2,
)

NODE_CLASS_MAPPINGS = {**_M1, **_M2}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
