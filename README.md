# Anima LoRA Block Weight

> **About this project**: This plugin was developed by **Claude (Anthropic's AI)** after simple testing
> and in-depth discussion with me — I did not write it myself. The layering logic, algorithms, and code
> were all produced by Claude; I provided real-world testing, experimental data, and feedback. The UI
> design references [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora)
> (its per-layer sliders + impact-coloring interaction). Stated here to avoid confusion.
>
> **Interface language**: Node UI text switches automatically between English and Chinese based on
> ComfyUI's language setting (`Comfy > Locale`). Only the displayed text changes — backend parameter
> names stay fixed, so workflow storage and API calls are unaffected.

A ComfyUI node for block-weight layering of LoRAs. Originally built for **Anima** (NVIDIA Cosmos
architecture, 2B anime model), now grown into a more general "auto-segment by measured metric" layering
tool. Adjust per-layer strength live during generation, or bake a tuned layering into a new LoRA file.

Existing community LoRA layering nodes target Flux/SDXL and hardcode dot-separated regex (`blocks.0`),
which can't match Anima's underscore naming (`lora_unet_blocks_0_...`), so layering silently does nothing
on Anima. This node fixes that.

Two release nodes (menu category `loaders`), plus two LoKr experimental nodes (see end):

| Node | Purpose | Output |
|---|---|---|
| Anima LoRA Block Weight V2 | Runtime layering, live | Tuning, no file |
| Anima LoRA Block Weight Export V2 | Bake layering into a new LoRA | .safetensors |

---

## An important premise (read first)

Many assume "certain DiT layers always control composition, others always control detail." But per
NVIDIA's official docs: **Anima/Cosmos is a uniformly stacked standard DiT — all 28 blocks share the
same structure, with no fixed functional partition in the architecture.**

So "which block segment controls motion vs color" is **not architectural — it is trained into each LoRA
individually and differs per LoRA**. Confirmed in testing: style LoRAs often have their strong region in
the middle, while character LoRAs may have theirs at the tail (e.g. blocks 26-27).

Therefore this node ships **no hardcoded functional presets** — a preset that's right on LoRA A is wrong
on LoRA B. Instead it uses two approaches:

1. **Auto-segment**: split into four segments by the *current* LoRA's measured metric, not fixed ranges.
2. **Dual metrics**: two ways to measure each block, to observe your LoRA from different angles.

What each segment actually controls can only be confirmed by your own generation tests — any static
metric is an aid, not the answer.

---

## Core concept

Final scaling factor per weight tensor:

```
factor = block weight × submodule-type coefficient
```

- **Block weight**: `grouped` from four segments (`seg_1`~`seg_4`); `per_block` from each block's slider.
- **Submodule-type coefficients**: `self_attn` / `cross_attn` / `mlp` / `adaln`, active in both modes.
- `alpha` / `dora_scale` are not scaled (scaling one down-side tensor already changes the contribution).

**Why `seg_1`~`seg_4` (pure numbering)**: earlier versions used `motion/proportion/core/detail`
(function assumptions) and `weak/medium/strong/peak` (strength). But since the node supports two metrics,
"which tier" means different (even opposite) things under each, so any name implying function or strength
misleads under one metric. Hence neutral position numbers.

---

## Auto-segment (on by default)

With `auto_segment` on (default), the node reads the current LoRA's metric and splits the 28 blocks into
four segments by **quantile**, auto-filling the range boxes:

- Quantile: sort by metric value; lowest ~1/4 → `seg_1`, ..., highest ~1/4 → `seg_4`.
- Each segment is internally contiguous; segments may be non-contiguous with each other (e.g. `seg_4`
  could be `16-18,24-27`).
- Includes smoothing (removes isolated spikes) and a minimum-run guard (avoids single-block fragments).

When on, the four range boxes are greyed out (set by the metric); to edit manually, turn it off first.
On by default so newcomers get a sensible split out of the box.

---

## Dual segment metric (segment_metric)

The top `segment_metric` toggles between two metrics (arrows), driving both auto-segment and coloring:

- **norm (default)**: measures each block's "strength."
- **effective_rank**: measures each block's "directional richness."

**Why two**: testing found these two are **strongly negatively correlated** — high-norm layers tend to
concentrate energy in a few directions (strong, specialized transforms), while low-norm layers tend to be
spread across many directions (broad, weak global adjustments, possibly color/texture-like diffuse
content). Norm alone treats "low-norm but directionally rich" layers as weak and ignores them;
effective_rank surfaces them. The two are complementary.

> ⚠ Both are **static metrics** reflecting weight properties, **not function**. Which metric's segments
> better match the functional split you want must be confirmed by your own generation tests — dual metrics
> exist to make that comparison easy, not to decide for you.

### A real test (reference; method is reusable)

Using effective_rank on a style LoRA, single-variable test (fixed seed/prompt/sampler/steps/CFG; lower one
segment's weight to 0.3 at a time). **Results apply only to that one LoRA — illustrative only.**

| Segment (its effective_rank distribution) | Effect of lowering to 0.3 |
|---|---|
| one segment | Overall style switch: linework/coloring/pose/anatomy all change noticeably; biggest impact |
| another | Mainly changes "motion"; the rest barely changes |
| another | Controls small objects/props integrity: held items deform or break, but the subject is unaffected |
| another (highest-frequency) | Controls highest-frequency detail: anatomy barely changes; mainly cleans up the image (removes high-frequency noise) |

The value isn't memorizing "which segment does what" (true only for that LoRA), but showing that
effective_rank's segments do map to distinct functions; the "biggest-impact segment" isn't necessarily the
highest-valued one under effective_rank (which is exactly why neutral numbering replaced strength names);
and you can map your own LoRA by running this method.

---

## impact coloring

In the per-block panel, each row is colored by that block's metric score (blue=low → cyan → yellow → red=high).
- Uses the currently selected `segment_metric`, contrast-stretched.
- It's an intrinsic LoRA property, unaffected by your slider changes.
- Requires one generation first. The coloring data is persisted, so refresh/reload/copy won't lose it.

---

## Loader node: Anima LoRA Block Weight V2

Runtime layering, live — the recommended main node. Two control modes (`control_mode`, top arrows):

- **grouped (four segments, default)**: `seg_1`~`seg_4`, each a range box + a weight. Best with auto-segment.
- **per_block**: 28 sliders, shown as a compact graphical panel when JS is loaded; good for fine scanning.

### Parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| model / clip | — | — | Model & CLIP input |
| lora_name | dropdown | — | LoRA file (click to open list) |
| strength_model / strength_clip | float | 1.0 | Overall LoRA strength on model / CLIP |
| control_mode | enum | grouped | grouped four-segment / per_block |
| auto_segment | bool | true | Auto-segment (range boxes greyed, set by metric) |
| segment_metric | enum | norm | Segment/color metric: norm / effective_rank |
| seg_1_blocks ~ seg_4_blocks | str | auto | Four-segment ranges (filled by metric when auto on) |
| seg_1_weight ~ seg_4_weight | float | 1.0 | Per-segment weights |
| blk00 … blk27 | float | 1.0 | Per-block coefficients (per_block) |
| w_self_attn / w_cross_attn / w_mlp / w_adaln | float | 1.0 | Submodule-type coefficients |
| verbose | bool | false | Print per-block factors to console |

**Native fallback**: all controls are ComfyUI native widgets; JS only hides and redraws them into a
compact panel. If JS fails to load, the node falls back to native widgets laid out plainly — long but
fully functional. If you see scattered native sliders: ① hard-refresh `Ctrl+Shift+R` (usually cache);
② confirm `web/anima_block_weight.js` exists; ③ check the F12 console.

---

## Export node: Anima LoRA Block Weight Export V2

Bakes the layered scaling into a new `.safetensors`. Complementary to the loader: loader is for tuning
(no file), exporter freezes the same scaling into a finished file any plain LoRA Loader can load (result =
your coefficients + strength 1.0).

Implementation: the factor is multiplied into one down-side tensor per module (kohya `lora_down`,
diffusers `lora_A`, LoKr `lokr_w1`); up-side / alpha / dora_scale unchanged; original `__metadata__` kept
with an export record appended.

| Param | Type | Default | Notes |
|---|---|---|---|
| output_name | str | anima_lora_baked | Output filename (.safetensors auto-appended) |
| save_to | enum | loras | Save to loras (directly loadable) or output |
| overwrite | bool | false | Overwrite same name; if false, auto-append _1/_2 |

Other params (control_mode / auto_segment / segment_metric / four segments / submodule coeffs) match the loader.

---

## Three storage formats supported

Auto-detected, no manual selection:
- **kohya**: `.lora_down.weight` / `.lora_up.weight` (most common)
- **diffusers**: `.lora_A.weight` / `.lora_B.weight` (some LoRAs; earlier builds couldn't measure these — now supported)
- **LoKr**: `.lokr_w1` / `.lokr_w2` (experimental nodes)

---

## Scanning method (map your own LoRA's real function distribution)

Since distribution differs per LoRA, scanning yourself is most reliable:

1. Fix seed/prompt/sampler/steps/CFG; disable all other LoRAs.
2. Use auto-segment for four segments, or per_block to move one small range at a time.
3. Lower one segment (or one submodule coefficient) to 0.3~0.5 vs the all-1.0 baseline: composition
   collapses → that segment controls structure; style/linework changes → controls style; color changes → controls color.
4. Record, return to grouped, fine-tune around the measured boundaries.
5. Switch `segment_metric` (norm / effective_rank) and scan each — see which metric's segments map more
   cleanly to a given function. That's the point of dual metrics.

Tips: move one knob group at a time with a fixed seed; explore 0.0~1.0 mostly (>1.0 risks artifacts, cap 2.0);
enable `verbose` to verify applied values.

---

## LoKr experimental (optional, advanced)

The two release nodes are stable on plain LoRAs. For LoKr (a LoRA variant stored differently), try the two
experimental nodes in `experimental/` (menu `loaders/experimental`); same params and UX, and they also
auto-detect plain LoRAs.

**Real status (important)**: "math verified, but no generation testing," so it stays experimental and is
not merged into the release nodes.

- Verified (in code, across multiple real LoKr files): scaling math correct (factor on only one w1-side
  block, equivalent to overall scaling, never squared); impact ordering correct (matches the
  fully-reconstructed reference); all storage forms (full / decomposed / with DoRA) and parameter types
  don't crash (never reads alpha/rank, avoiding the inf-alpha trap some LoKrs have).
- Not verified, and not in this version: actual generation quality; LoKr's functional distribution
  (needs per-variable generation tests — large effort, deferred).
- The experimental branch loads safely; even if it errors it won't affect the release nodes. To remove it,
  delete the whole `experimental/` folder.

---

## License

MIT
