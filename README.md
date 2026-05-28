# Anima LoRA Block Weight

> **About this project**: This plugin was developed by **Claude (Anthropic's AI)** after simple testing
> and in-depth discussion with me — I did not write it myself. The functional map, layering logic, and
> code were all produced by Claude; I provided real-world testing, experimental data, and feedback.
> The UI design references
> [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora)
> (the per-layer sliders + impact-coloring interaction). Stated here to avoid confusion or ambiguity.
>
> **Interface language**: Node UI text (buttons, tooltips) switches automatically between English and Chinese
> based on ComfyUI's language setting (`Comfy > Locale`); backend descriptions are bilingual (EN | ZH).

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

### Option 1: git clone (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rom0718/Anima-lora-block-weight.git
```

### Option 2: manual download

Download this repo as a zip, extract into `ComfyUI/custom_nodes/`, with `__init__.py` directly at that level:

```
ComfyUI/custom_nodes/Anima-lora-block-weight/
    ├── __init__.py
    ├── anima_lora_block_weight_v2.py         # loader node: four-segment / per-block sliders + impact coloring
    ├── anima_lora_block_weight_export.py     # export node: bake to file
    ├── web/
    │   └── anima_block_weight.js             # frontend panel (sliders / coloring / EN-ZH switch)
    └── experimental/                          # optional: LoKr experimental branch
```

Both options require: **restart ComfyUI**, then **hard-refresh the browser (Ctrl+Shift+R)** to clear the
frontend cache (after updating a node with frontend JS, the cache must be cleared or the old JS may persist).

After restart, find the nodes under `loaders`. No third-party dependencies.

> Update: `cd ComfyUI/custom_nodes/Anima-lora-block-weight && git pull`, then restart + hard-refresh.

---

## How it works

Final scale factor per weight tensor:

```
factor = block_weight × submodule_type_weight
```

- **block_weight**: in `grouped` mode, from the four measured segments (motion / proportion / core / detail);
  in `per_block` mode, from each block's own slider (blk00–blk27).
- **submodule type weights**: self_attn / cross_attn / mlp / adaln, each a multiplier, active in both modes.
- `.alpha` tensors are not scaled (scaling up/down already changes the contribution).

### Measured functional map (this is the V2 default, from real ablation)

Unlike the generic DiT rule of thumb, V2's segments come from fixed-seed single-variable ablation on real
Anima style LoRAs (see the case study below). Note these are **tendencies with cross-talk between segments**,
not clean separations — each segment is really a "prior-obedience knob," not a dedicated brush.

| Dimension | Value | Measured tendency |
|-----------|-------|-------------------|
| seg_motion | block 0–11 | Prompt-obedience of motion / overall body size |
| seg_proportion | block 12–14 | Prior-obedience of body proportion / stockiness |
| seg_core | block 15–18 | LoRA core expression (anatomy + proportion + material; highest info density) |
| seg_detail | block 19–27 | Global refinement (saturation / purity and other surface attributes) |
| Type · self_attn | `w_self_attn` | Detail content + some anatomy + some style |
| Type · cross_attn | `w_cross_attn` | Least effect on style — safest for fine tuning |
| Type · mlp | `w_mlp` | Largest effect on anatomy (also carries style) |
| Type · adaln | `w_adaln` | Holistic effect, no single direction |

> ⚠️ **These tendencies were verified on style LoRAs and appear to be architecture-level (consistent across
> three very different LoRAs), but your specific LoRA may differ.** The built-in **impact coloring** (per_block
> panel) computes each block's weight L2 norm so you can see your LoRA's real hot blocks at a glance; use
> per_block mode to scan and confirm (see "Scanning method"). Full details under "Case study" below.

---

## Loader node: Anima LoRA Block Weight V2

Runtime layering, live. Provides a **measurement-based four-segment split**, a **graphical per-block
slider panel**, and **impact coloring**. (The old V1 loader node has been removed; download a historical
release of this repo if you need it.)

### Three control modes

Switch via `control_mode`:

- **grouped (four segments, default)**: four segments based on the measured functional map, each range
  customizable:
  - `seg_motion` (default 0-11): prompt-obedience of motion / overall body size
  - `seg_proportion` (default 12-14): prior-obedience of body proportion / stockiness
  - `seg_core` (default 15-18): the LoRA's core expression segment (anatomy + proportion + material,
    highest information density)
  - `seg_detail` (default 19-27): global refinement
- **per_block (sliders)**: one slider per block (28 total). With the frontend JS installed, this renders as
  a compact panel (checkbox toggle + slider + number box + impact coloring).

### Impact coloring

In the per_block panel, each row's checkbox and row background are colored by that block's **impact score**
(blue=low → cyan → yellow → red=high). Impact is computed at runtime from each block's LoRA weight L2 norm,
contrast-stretched into a color, so you can see at a glance **which blocks matter most in this LoRA**.

Notes:
- Impact reflects the LoRA's **intrinsic "information density"** — a relative ranking that **does not change
  when you adjust sliders** (sliders are weights you apply; impact is a property of the LoRA; they are
  independent).
- Coloring uses **contrast stretch** (maps this LoRA's [weakest, strongest] to [blue, red]), so relative
  differences show even when absolute norms are close; the trade-off is it expresses "relative importance
  within this LoRA," not absolute strength.
- You must **generate once** before the node sends impact data to the frontend for coloring.

### About the frontend JS and "native fallback"

All controls (strength, the four segments, w_, verbose, per-block) are ComfyUI **native widgets** underneath;
the frontend JS merely hides them and redraws a unified compact panel ("native controls as the base, JS as
the skin"). On a normal install, the JS file (`web/anima_block_weight.js`) loads automatically with the node,
so you see the compact panel.

**If the JS fails to load** (stale browser cache, missing file, incompatible frontend version), the node falls
back to **all native controls laid out directly**: strength / the four segment ranges and weights / w_ /
verbose / the 28 blk sliders all show **at once** (no longer auto-hiding irrelevant items by mode). The node
gets long and unpolished — but it is **fully functional**: all values pass through correctly and generation
works; you only lose the compact layout, coloring, and mode-based hiding.

If you see a pile of scattered native sliders instead of the compact panel:
1. **Hard-refresh the browser (Ctrl+Shift+R)** first (stale frontend cache is the most common cause);
2. If that fails, confirm `custom_nodes/Anima-lora-block-weight/web/anima_block_weight.js` exists;
3. Open the browser console (F12) and check for related errors.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model / clip | — | — | Model and CLIP inputs |
| lora_name | dropdown | — | LoRA file to load |
| strength_model / strength_clip | float | 1.0 | Overall LoRA strength on model / CLIP |
| control_mode | enum | grouped | `grouped` four-segment / `per_block` sliders |
| seg_motion_blocks / _weight | str / float | "0-11" / 1.0 | Motion segment range & factor |
| seg_proportion_blocks / _weight | str / float | "12-14" / 1.0 | Proportion segment range & factor |
| seg_core_blocks / _weight | str / float | "15-18" / 1.0 | Core segment range & factor |
| seg_detail_blocks / _weight | str / float | "19-27" / 1.0 | Detail segment range & factor |
| blk00 … blk27 | float | 1.0 | Per-block factors (per_block; JS beautifies into a panel) |
| w_self_attn / w_cross_attn / w_mlp / w_adaln | float | 1.0 | Submodule type factors |
| verbose | bool | false | Print per-block factors to console |

With all factors at the default 1.0, behavior equals a plain LoRA Loader.

### Common recipes (grouped four segments)

| Goal | motion | proportion | core | detail | type weights |
|------|--------|------------|------|--------|--------------|
| Make body size follow the prompt more | 0.5 | 1.0 | 1.0 | 1.0 | all 1.0 |
| Lock proportion against prompt | 1.0 | tune this (lower = more prior) | 1.0 | 1.0 | all 1.0 |
| Large overhaul, soften the look | 1.0 | 1.0 | 0.5-0.7 | 1.0 | all 1.0 |
| Reduce anatomy only (accept some style loss) | 1.0 | 1.0 | 1.0 | 1.0 | w_mlp=0.7 |
| Counter prompt-hijacking / concept pollution | 1.0 | 1.0 | 1.0 | 1.0 | w_cross_attn=0.5 |

---

## Export node: Anima LoRA Block Weight Export V2

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

## Case study: Real functional map of an Anima LoRA

To illustrate that "the default table is only a starting point," this section records one complete scanning run.
**Results apply only to the LoRA tested**, but they reflect a general pattern: on the Anima/Cosmos architecture,
multiple theoretical defaults may be systematically shifted. Scan your own LoRA to verify.

### Method

Control variables: fixed seed / prompt / sampler / steps / CFG; all other LoRAs disabled. One knob is changed
per shot, others kept at 1.0, each compared against the all-1.0 baseline.

### Results

**By block depth**

| Stage | Observed role | Theoretical default |
|-------|---------------|---------------------|
| shallow 0-8 | Motion consistency: hand gestures, facial expression, toe orientation and other micro-behaviors | Composition / anatomy ❌ |
| **middle 9-18** | **Dense-information layer: anatomy + most of the style + part of the composition** | Transition ❌ |
| deep 19-27 | Minor style details / brushwork | Main style ❌ |

**By submodule type**

| Knob | Observed role | Theoretical default |
|------|---------------|---------------------|
| `w_self_attn` | Detail content + minor anatomy + minor style | Anatomy / pose ❌ |
| **`w_mlp`** | **Largest effect on anatomy; also affects style** (strongest single-dimension knob) | Style ❌ |
| `w_cross_attn` | Smallest effect on style; light detail tweaks (safe micro-adjust) | Prompt response ⚠ |
| `w_adaln` | Global influence: all dimensions shift; no specific direction | Overall tone ✓ |

> Counter-intuitive finding: `w_mlp` is the strongest anatomy knob on this LoRA, while `w_self_attn` has
> only minor effect on anatomy — the opposite of the "self_attn governs spatial structure" rule of thumb.
> This is explainable from training-data distribution: when anatomy and style are jointly trained into the
> same block range and the same submodule type (here, the mlp of the middle stage), adjusting mlp will
> shift both at once.

### Recipes derived from this map

| Goal | Configuration |
|------|---------------|
| Reduce anatomy, accept some style loss | `w_mlp=0.7`, others 1.0 |
| Large overhaul, soften the overall look | `seg_core_weight=0.5-0.7`, others 1.0 |
| Tiny detail tweaks with style nearly intact | `w_cross_attn=0.7`, others 1.0 |
| Preserve the LoRA's signature look | All 1.0; use prompts / negatives to adjust specific elements |
| Reduce motion exaggeration while keeping body type | `seg_motion_weight=0.5-0.7`, others 1.0 |

### Important practical conclusion

On this LoRA, **fully separating style from anatomy is not achievable** — they are physically trained into
the same range (middle) and the same submodule type (mlp). This is a property of the LoRA's training, not a
limitation of the node. The right strategy is not to seek clean separation but to find an **acceptable
trade-off** (e.g. a sweet spot at ~60% anatomy reduction with ~20% style loss).

### Notes for other Anima LoRA users

The map above is **not a universal Anima configuration** — every LoRA differs. But compared to the DiT
defaults, it may be closer to how Anima-architecture LoRAs actually behave, and can serve as a
**priority hypothesis** during scanning:

- Suspect first that middle is the dense-information layer (rather than shallow)
- Suspect first that mlp and self_attn may swap roles vs. the DiT rule of thumb
- `w_adaln` typically has no specific direction; keep at 1.0 first
- `w_cross_attn` is typically the safest light-touch knob

You should still run the scanning method yourself for a precise map of your own LoRA.

---

## License

MIT
