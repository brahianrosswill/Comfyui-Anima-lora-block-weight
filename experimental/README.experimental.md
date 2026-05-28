# 实验性分支：LoKr 支持（EXPERIMENTAL）

> ⚠️ 这是实验性功能，独立于正式发布的两个节点，未经长期验证。正式使用请用发布版节点。

## 这是什么

正式版的两个节点只支持普通 LoRA（`lora_down`/`lora_up` 结构）。
本实验分支额外支持 **LoKr**（LoRA with Kronecker product，权重为 `lokr_w1`/`lokr_w2`）。

包含两个实验节点（菜单分类 `loaders/experimental`）：
- **Anima LoKr Block Weight [实验性]** — 运行时分层加载
- **Anima LoKr Block Weight Export [实验性]** — 烘焙导出

它们也兼容普通 LoRA（自动检测格式），但请仍把它当实验品。

## 为什么 LoKr 需要单独处理

LoKr 每个模块的贡献 = `lokr_w1 ⊗ lokr_w2`（克罗内克积），没有 `lora_down`/`lora_up`。

关键数学性质（已用真实文件验证）：
```
(c · lokr_w1) ⊗ lokr_w2  ==  c · (lokr_w1 ⊗ lokr_w2)
```
因此本节点**只把缩放系数乘到 lokr_w1 一块**，等效于整体缩放 factor。
绝不能两块都乘——那会变成 factor²。

## 已知注意事项

- 某些 LoKr（如用训练器 trick 把 dim/alpha 设为超大值交给 Prodigy 控制的）其
  `alpha`/`lokr_rank` 存成 fp16 后会溢出为 `inf`。本节点**完全不读取/使用 alpha 或 rank**，
  只对 lokr_w1 做乘法，从而避开 `inf/inf = NaN` 的问题。已验证烤出文件无 NaN。
- 分层的"层定位"逻辑与发布版完全一致（同样的 block 正则与子模块分类）。
- 参数与发布版节点相同，调参思路、扫描定位法都通用。

## 安装

experimental 子目录会被主 `__init__.py` 自动尝试加载。无需额外操作；
若不想启用实验功能，删除整个 `experimental/` 目录即可，不影响发布版节点。
