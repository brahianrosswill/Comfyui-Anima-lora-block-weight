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

## Installation

Put the whole folder into ComfyUI's `custom_nodes` directory, then restart ComfyUI:

```
ComfyUI/custom_nodes/Comfyui-Anima-lora-block-weight/
```

Or clone it there:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rom0718/Comfyui-Anima-lora-block-weight.git
```

After restart, the nodes appear under the `loaders` menu category (LoKr experimental nodes under
`loaders/experimental`). Dependencies: only ComfyUI's bundled torch / safetensors — no extra packages.

## File structure

```
Comfyui-Anima-lora-block-weight/
├── __init__.py                              Node registration entry (safe-loads the experimental branch)
├── anima_common.py                          Shared core module (format detect / impact / scaling / auto-segment)
├── anima_lora_block_weight_v2.py            Release · loader node
├── anima_lora_block_weight_export.py        Release · export node
├── web/
│   └── anima_block_weight.js                Shared frontend panel (unified UI / coloring / EN-ZH switch)
├── experimental/                            LoKr experimental branch (safe to delete entirely)
│   ├── __init__.py
│   ├── anima_lokr_block_weight_experimental.py   LoKr loader + export nodes
│   └── README.experimental.md
├── README.md / README.txt                   English docs
└── 使用说明.zh-CN.md / .txt                 Chinese docs
```

All core logic lives in `anima_common.py`; all four nodes import from it, keeping the release and
experimental nodes in sync.

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
four segments, auto-filling the range boxes. How it cuts is set by `segment_method`:

**jenks (natural breaks, default)** — Jenks Natural Breaks, cutting at the data's natural gaps:

- Makes values within each class close and classes far apart (min intra-class, max inter-class variance).
- **Segment sizes may be uneven** — this **faithfully reflects the LoRA's real structure**: e.g. a
  character LoRA with energy piled at the tail gets one large tail segment + a few small weak ones,
  telling you to mainly adjust the tail while the rest are just fine-tuning.
- Verified on 6 real files in-sandbox: jenks's classification quality (GVF) beats quantile on every
  file and every metric.

**quantile** — mechanical four-way split by sorted metric value:

- Even class sizes (~7 each), so the four sliders have similar sensitivity and predictable feel.
- But it ignores actual value gaps and may lump far-apart values together just by adjacent rank.
- Includes smoothing and a minimum-run guard. Serves as the alternative / fallback to jenks.

> About Jenks: a classic 1D classification method by cartographer George F. Jenks (1967, "Jenks Natural
> Breaks"), a public-domain standard algorithm (also the default in software like ArcGIS). This project
> uses that algorithm directly — it is not homegrown.

Both methods keep segments internally contiguous where possible and allow non-contiguous segments (e.g.
`seg_4` could be `16-18,24-27`). When on, the range boxes are greyed out; to edit manually, turn off
`auto_segment` first. On by default with jenks by default, so newcomers get an out-of-the-box split that
stays faithful to the LoRA's own structure.

**About the default (norm + jenks):** we tested all four metric × method combinations on several real
LoRAs (style / aesthetic / character, etc.) and found **norm + jenks to be the most robust default** — on
every LoRA tested, its four segments stayed well-separated in strength with no "dead" segment (one that
does nothing when adjusted). It never broke down on any LoRA, so we settled on it as the default.

That said: **no single combination is optimal for every LoRA.** In testing, the best combination varied
per LoRA — some style LoRAs separated more clearly under effective_rank + quantile, while others were best
under effective_rank + jenks. That's exactly why both switches stay user-configurable: the default gives
you the safest starting point, and to tune a specific LoRA to its best, try all four combinations and see
which makes the four segments most distinct when adjusted.

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

## Submodule-type coefficients (w_self_attn / w_cross_attn / w_mlp / w_adaln)

The four segments (seg_1~4) layer by **block position** — controlling *which layers*. These four `w_`
coefficients are a separate axis, layering by **submodule type inside each block** — controlling *which
part of each layer*. Both axes apply together and stack.

Each Anima/Cosmos block is built from these four submodule types. In plain terms:

- **self_attn**: lets positions within the image "look at" each other and coordinate — it handles the
  image's own internal structure and consistency.
- **cross_attn**: brings in your **prompt text** — the model uses it to keep referencing your prompt
  throughout denoising. It's the channel through which text conditioning enters the image.
- **mlp (feed-forward)**: a per-position non-linear transform of features — think of it as each layer's
  "feature processor."
- **adaln (adaptive layer norm)**: modulates each layer's intensity using the current denoising timestep.
  Not a standalone module — it's a modulation knob attached to the above.

How to use: each defaults to 1.0 (unchanged). **Lowering** one (e.g. 0.7, 0.5) **weakens that submodule
type's contribution across the whole LoRA**; 0 nearly disables it. Raising above 1.0 amplifies but risks
artifacts (cap 2.0, use with care). They multiply with the segments (factor = segment weight × submodule
coefficient), so you can "weaken only a certain submodule type within a certain segment."

> ⚠ Key reminder: **do not assume "self_attn always controls structure, mlp always controls style."**
> Like block segments, what each submodule type carries is **trained into each LoRA and differs per LoRA** —
> there is no fixed architectural division of labor. Testing has even shown counter-intuitive cases: on
> one LoRA, mlp was the *strongest anatomy knob* while self_attn barely affected anatomy — the opposite of
> "common wisdom." So discover what these knobs actually do on *your* LoRA via the scanning method. One
> empirical starting point: adaln tends to have no single clear direction (diffuse effect), so you can
> leave it at 1.0 and start with self_attn / cross_attn / mlp.

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
| segment_method | enum | jenks | Segmenting: jenks (natural breaks, faithful) / quantile (even, steady feel) |
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

Other params (control_mode / auto_segment / segment_metric / segment_method / four segments / submodule coeffs) match the loader.

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
