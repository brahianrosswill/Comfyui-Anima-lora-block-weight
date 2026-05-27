# Anima LoRA Block Weight

A ComfyUI node providing block-weight layering for LoRAs of **Anima** (NVIDIA Cosmos architecture, 2B anime model).
**Supports both runtime tuning and bake-to-file export**: adjust per-layer strength live during generation,
or freeze a tuned layering into a new LoRA file.

Existing community LoRA layering nodes are designed for Flux/SDXL and hardcode dot-separated regex
(`blocks.0`). They cannot match Anima's underscore naming (`lora_unet_blocks_0_...`), so layering silently
does nothing on Anima. This node fixes that. Verified against a real Anima LoRA (28 blocks, 896 weight
tensors): 100% matched, zero misses.

Includes two nodes (menu category `loaders`):

| Node | Purpose | Output |
|------|---------|--------|
| **Anima LoRA Block Weight** | Runtime layering, live | Tuning, no file |
| **Anima LoRA Block Weight Export** | Bake layering into a new LoRA | Exports `.safetensors` |

---

## Installation

Place this repo into ComfyUI's custom_nodes directory, with `__init__.py` directly at that level:

```
ComfyUI/custom_nodes/anima-lora-block-weight/
    ├── __init__.py
    ├── anima_lora_block_weight.py            # loader node: runtime layering
    └── anima_lora_block_weight_export.py     # export node: bake to file
```

Restart ComfyUI; find both nodes under `loaders`. No third-party dependencies.

---

## How it works

Final scale factor per weight tensor:

```
factor = block_weight × submodule_type_weight
```

- **block_weight**: from shallow / middle / deep tiers in `grouped` mode; per-block from `block_weights` in `per_block` mode.
- **submodule type weights**: self_attn / cross_attn / mlp / adaln, each a multiplier, active in both modes.
- `.alpha` tensors are not scaled (scaling up/down already changes the contribution).

### Roles of stages and submodules

| Dimension | Value | Role |
|-----------|-------|------|
| Depth · shallow | block 0–8 | Global composition, pose, skeleton, proportions |
| Depth · middle | block 9–18 | Subject form, semantic transition |
| Depth · deep | block 19–27 | Style, brushwork, texture, coloring |
| Type · self_attn | `w_self_attn` | Internal spatial structure, composition, anatomy |
| Type · cross_attn | `w_cross_attn` | Text→image mapping, prompt response |
| Type · mlp | `w_mlp` | Style, texture, look features |
| Type · adaln | `w_adaln` | DiT adaptive norm modulation, overall tone |

> The block ranges are general DiT rules of thumb, not measured for any specific LoRA. Use per_block mode
> to scan and locate your own LoRA's true boundaries (see "Scanning method").

---

## Loader node: Anima LoRA Block Weight

Runtime layering, live, for tuning and everyday use.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model / clip | — | — | Model and CLIP inputs |
| lora_name | dropdown | — | LoRA file to load |
| strength_model | float | 1.0 | Overall LoRA strength on model |
| strength_clip | float | 1.0 | Overall LoRA strength on CLIP |
| control_mode | enum | grouped | `grouped` tiers / `per_block` per-block |
| shallow_blocks / _weight | str / float | "0-8" / 1.0 | Shallow range & factor (grouped) |
| middle_blocks / _weight | str / float | "9-18" / 1.0 | Middle range & factor (grouped) |
| deep_blocks / _weight | str / float | "19-27" / 1.0 | Deep range & factor (grouped) |
| block_weights | multiline | "" | Per-block weights (per_block) |
| w_self_attn / w_cross_attn / w_mlp / w_adaln | float | 1.0 | Submodule type factors |
| default_weight | float | 1.0 | Factor for unspecified blocks |
| verbose | bool | false | Print per-block factors to console |

With all factors at the default 1.0, behavior equals a plain LoRA Loader.

### per_block syntax

Three forms, mixable (comma / newline / semicolon as separators), later overrides earlier:

```
plain sequence:  1,1,0.5,0.5            # maps to block 0,1,2,3...; pad with default
index:value:     5:0.3, 27:1.2          # only blocks 5 and 27
range:value:     0-8:0.3, 19-27:1.0     # set ranges in bulk
mixed:           0-27:1.0, 12:0, 13:0   # all 1.0, then turn off 12 and 13
```

Range syntax (shallow/middle/deep_blocks) accepts `all` / `0-8` / `0,3,5` / `0-8,19-27`.
On overlap, priority is deep > middle > shallow.

### Common recipes

| Goal | shallow | middle | deep | type weights |
|------|---------|--------|------|--------------|
| Keep style, weaken composition/anatomy | 0.3 | 0.7 | 1.0 | all 1.0 |
| Keep composition/anatomy, weaken style | 1.0 | 0.7 | 0.3 | all 1.0 |
| Reduce body type/anatomy only | 1.0 | 1.0 | 1.0 | w_self_attn=0.5 |
| Counter prompt-hijacking/concept pollution | 1.0 | 1.0 | 1.0 | w_cross_attn=0.5 |
| Strengthen style | 1.0 | 1.0 | 1.0 | w_mlp=1.2 |

---

## Export node: Anima LoRA Block Weight Export

Bakes the layering into a new `.safetensors`. Complementary to the loader node: the loader is for tuning and
produces no file; the export node freezes the same scaling into a finished file, loadable by any plain LoRA Loader.

**Recommended flow**: tune coefficients with the loader node until satisfied → put the **same coefficients**
into the export node → get a finished file. Loading it equals your coefficients at strength 1.0, with no
layering node attached — and is easy to share.

### Implementation

The factor (as above) is multiplied into each module's `lora_down` (single-sided scaling, mathematically
equivalent to runtime `tensor*factor`); `lora_up` and `alpha` stay unchanged; the original `__metadata__`
is preserved with an export record appended. Verified on a real file: exact scaling ratios, up/alpha unchanged,
file structure intact.

### Export-only parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| output_name | str | anima_lora_baked | Output filename (`.safetensors` auto-appended) |
| save_to | enum | loras | Save to `loras` (directly loadable) or `output` |
| overwrite | bool | false | Overwrite same name; if false, auto-appends `_1/_2` |

All other parameters (control_mode / tiers / block_weights / four type weights / default_weight) match the loader node.

---

## Scanning method (find your LoRA's true boundaries)

per_block mode can map your LoRA's functions:

1. Fix seed, prompt, sampler, steps, CFG (control variables).
2. With per_block, turn off one small range at a time:
   ```
   0-27:1.0, 0-3:0      # only 0–3 off
   0-27:1.0, 4-7:0      # only 4–7 off
   ...                  # up to 24-27
   ```
3. Compare each against the all-1.0 baseline: if **composition collapses** → that range governs the skeleton;
   if **style changes** → that range governs style.
4. Record results, return to grouped mode with the measured boundaries for efficient daily use.

Tuning tips: change one set of knobs at a time with a fixed seed; explore mainly within 0.0–1.0 (>1.0 risks
artifacts, max 2.0); enable verbose to confirm applied values.

---

## License

MIT
