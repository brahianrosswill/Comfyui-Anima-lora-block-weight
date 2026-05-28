"""
Anima LoRA Block Weight — ComfyUI custom node
为 Anima (NVIDIA Cosmos 架构) 的 LoRA 提供分层（block weight）控制。

发布版节点（稳定）：
  - AnimaLoRABlockWeightV2        分层加载（四段 / 逐block滑块 + impact染色）
  - AnimaLoRABlockWeightExport    分层导出（烘焙成新的 .safetensors，与 V2 同参数体系）

两个节点共享前端面板（web/anima_block_weight.js），经 WEB_DIRECTORY 自动加载；
未加载 JS 时退化为原生控件，仍可正常使用。

注：旧版 V1 加载节点（AnimaLoRABlockWeight）已从发布版移除。
    如需使用，可下载本仓库的历史版本。

实验性节点（experimental/，可选，支持 LoKr）：
  - AnimaLoKrBlockWeightExperimental
  - AnimaLoKrBlockWeightExportExperimental
  实验分支以 try/except 安全加载；即使其出错也不影响发布版节点。
"""

from .anima_lora_block_weight_v2 import (
    NODE_CLASS_MAPPINGS as _M1,
    NODE_DISPLAY_NAME_MAPPINGS as _D1,
)
from .anima_lora_block_weight_export import (
    NODE_CLASS_MAPPINGS as _M2,
    NODE_DISPLAY_NAME_MAPPINGS as _D2,
)

NODE_CLASS_MAPPINGS = {**_M1, **_M2}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2}

# 前端 JS 目录（共享面板 UI）。ComfyUI 会自动加载其中的 .js。
WEB_DIRECTORY = "./web"

# 实验性分支：安全加载，失败不影响发布版
try:
    from .experimental import (
        NODE_CLASS_MAPPINGS as _ME,
        NODE_DISPLAY_NAME_MAPPINGS as _DE,
    )
    NODE_CLASS_MAPPINGS.update(_ME)
    NODE_DISPLAY_NAME_MAPPINGS.update(_DE)
    print("[Anima LBW] 实验性 LoKr 分支已加载 (loaders/experimental)")
except Exception as e:
    print(f"[Anima LBW] 实验性分支未加载（不影响发布版）: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
