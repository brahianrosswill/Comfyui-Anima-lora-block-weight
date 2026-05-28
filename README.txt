==============================================================================
Anima LoRA Block Weight
==============================================================================

[About this project]
This plugin was developed by Claude (Anthropic's AI) after simple testing and in-depth discussion with
me -- I did not write it myself. The functional map, layering logic, and code were all produced by Claude;
I provided real-world testing, experimental data, and feedback. The UI design references comfyUI-Realtime-Lora
(https://github.com/shootthesound/comfyUI-Realtime-Lora) -- the per-layer sliders + impact-coloring
interaction. Stated here to avoid confusion or ambiguity.

[Interface language]
Node UI text (buttons, tooltips) switches automatically between English and Chinese based on ComfyUI's
language setting (Comfy > Locale); backend descriptions are bilingual (EN | ZH).

------------------------------------------------------------------------------

A ComfyUI node providing block-weight layering for LoRAs of Anima (NVIDIA Cosmos architecture, 2B anime model).
Supports both runtime tuning and bake-to-file export: adjust per-layer strength live during generation,
or freeze a tuned layering into a new LoRA file.

Existing community LoRA layering nodes are designed for Flux/SDXL and hardcode dot-separated regex
(blocks.0). They cannot match Anima's underscore naming (lora_unet_blocks_0_...), so layering silently
does nothing on Anima. This node fixes that. Verified against a real Anima LoRA (28 blocks, 896 weight
tensors): 100% matched, zero misses.

Includes two nodes (menu category loaders):

  Node  |  Purpose  |  Output
  Anima LoRA Block Weight  |  Runtime layering, live  |  Tuning, no file
  Anima LoRA Block Weight Export  |  Bake layering into a new LoRA  |  Exports .safetensors

------------------------------------------------------------------------------
Installation
------------------------------------------------------------------------------

Place this repo into ComfyUI's custom_nodes directory, with __init__.py directly at that level:

    ComfyUI/custom_nodes/anima-lora-block-weight/
        ├── __init__.py
        ├── anima_lora_block_weight.py            # loader node: runtime layering
        └── anima_lora_block_weight_export.py     # export node: bake to file

Restart ComfyUI; find both nodes under loaders. No third-party dependencies.

------------------------------------------------------------------------------
How it works
------------------------------------------------------------------------------

Final scale factor per weight tensor:

    factor = block_weight × submodule_type_weight

· block_weight: from shallow / middle / deep tiers in grouped mode; per-block from block_weights in per_block mode.
· submodule type weights: self_attn / cross_attn / mlp / adaln, each a multiplier, active in both modes.
· .alpha tensors are not scaled (scaling up/down already changes the contribution).

Roles of stages and submodules

  Dimension  |  Value  |  Role
  Depth · shallow  |  block 0–8  |  Global composition, pose, skeleton, proportions
  Depth · middle  |  block 9–18  |  Subject form, semantic transition
  Depth · deep  |  block 19–27  |  Style, brushwork, texture, coloring
  Type · self_attn  |  w_self_attn  |  Internal spatial structure, composition, anatomy
  Type · cross_attn  |  w_cross_attn  |  Text→image mapping, prompt response
  Type · mlp  |  w_mlp  |  Style, texture, look features
  Type · adaln  |  w_adaln  |  DiT adaptive norm modulation, overall tone

! The mapping above is a general DiT rule of thumb, not universal across LoRAs. Measurements show
that some Anima LoRAs deviate substantially from this default table (e.g. mlp turning out to govern
anatomy, the middle stage becoming the dense-information layer, etc.). Use per_block mode to scan and
locate your own LoRA's true boundaries (see "Scanning method"). A complete case study is included
below under "Case study: Real functional map of an Anima LoRA".

------------------------------------------------------------------------------
Loader node: Anima LoRA Block Weight V2
------------------------------------------------------------------------------

Runtime layering, live. V2 is the recommended main node. Compared to V1 it adds a measurement-based
four-segment split, a graphical per-block slider panel, and impact coloring.

Three control modes (switch via control_mode):

  grouped (four segments, default): based on the measured functional map, each range customizable
    seg_motion     default 0-11   prompt-obedience of motion / overall body size
    seg_proportion default 12-14  prior-obedience of body proportion / stockiness
    seg_core       default 15-18  the LoRA's core expression segment (highest info density)
    seg_detail     default 19-27  global refinement
  per_block (sliders): one slider per block (28). With frontend JS, renders as a compact panel
    (checkbox + slider + number box + impact coloring); without JS, degrades to 28 native sliders
    that still work fully.

Impact coloring:
  In the per_block panel, each row is colored by that block's impact score (blue=low -> cyan -> yellow
  -> red=high). Impact is computed at runtime from each block's LoRA weight L2 norm, contrast-stretched
  into a color, reflecting the block's relative importance within this LoRA. Notes:
  - Impact is an intrinsic property of the LoRA; it does NOT change when you adjust sliders.
  - Uses contrast stretch ([weakest,strongest] -> [blue,red]): expresses relative importance within
    this LoRA, not absolute strength.
  - Generate once before the node sends impact data to the frontend for coloring.

Parameters

  Parameter  |  Type  |  Default  |  Description
  model / clip  |  —  |  —  |  Model and CLIP inputs
  lora_name  |  dropdown  |  —  |  LoRA file to load
  strength_model / strength_clip  |  float  |  1.0  |  Overall LoRA strength on model / CLIP
  control_mode  |  enum  |  grouped  |  grouped four-segment / per_block sliders
  seg_motion_blocks / _weight  |  str / float  |  "0-11" / 1.0  |  Motion segment range & factor
  seg_proportion_blocks / _weight  |  str / float  |  "12-14" / 1.0  |  Proportion segment range & factor
  seg_core_blocks / _weight  |  str / float  |  "15-18" / 1.0  |  Core segment range & factor
  seg_detail_blocks / _weight  |  str / float  |  "19-27" / 1.0  |  Detail segment range & factor
  blk00 … blk27  |  float  |  1.0  |  Per-block factors (per_block; JS beautifies into a panel)
  w_self_attn / w_cross_attn / w_mlp / w_adaln  |  float  |  1.0  |  Submodule type factors
  verbose  |  bool  |  false  |  Print per-block factors to console

Common recipes (grouped four segments)

  Goal  |  motion  |  proportion  |  core  |  detail  |  type weights
  Make body size follow prompt more  |  0.5  |  1.0  |  1.0  |  1.0  |  all 1.0
  Lock proportion against prompt  |  1.0  |  lower=more prior  |  1.0  |  1.0  |  all 1.0
  Large overhaul, soften the look  |  1.0  |  1.0  |  0.5-0.7  |  1.0  |  all 1.0
  Reduce anatomy only (accept some style loss)  |  1.0  |  1.0  |  1.0  |  1.0  |  w_mlp=0.7
  Counter prompt-hijacking / concept pollution  |  1.0  |  1.0  |  1.0  |  1.0  |  w_cross_attn=0.5


------------------------------------------------------------------------------
Export node: Anima LoRA Block Weight Export V2
------------------------------------------------------------------------------

Bakes the layering into a new .safetensors. Complementary to the loader node: the loader is for tuning and
produces no file; the export node freezes the same scaling into a finished file, loadable by any plain LoRA Loader.

Recommended flow: tune coefficients with the loader node until satisfied → put the same coefficients
into the export node → get a finished file. Loading it equals your coefficients at strength 1.0, with no
layering node attached — and is easy to share.

Implementation

The factor (as above) is multiplied into each module's lora_down (single-sided scaling, mathematically
equivalent to runtime tensor*factor); lora_up and alpha stay unchanged; the original __metadata__
is preserved with an export record appended. Verified on a real file: exact scaling ratios, up/alpha unchanged,
file structure intact.

Export-only parameters

  Parameter  |  Type  |  Default  |  Description
  output_name  |  str  |  anima_lora_baked  |  Output filename (.safetensors auto-appended)
  save_to  |  enum  |  loras  |  Save to loras (directly loadable) or output
  overwrite  |  bool  |  false  |  Overwrite same name; if false, auto-appends _1/_2

All other parameters (control_mode / tiers / block_weights / four type weights / default_weight) match the loader node.

------------------------------------------------------------------------------
Scanning method (find your LoRA's true boundaries)
------------------------------------------------------------------------------

per_block mode can map your LoRA's functions:

1. Fix seed, prompt, sampler, steps, CFG (control variables).
2. With per_block, turn off one small range at a time:
       0-27:1.0, 0-3:0      # only 0–3 off
       0-27:1.0, 4-7:0      # only 4–7 off
       ...                  # up to 24-27
3. Compare each against the all-1.0 baseline: if composition collapses → that range governs the skeleton;
   if style changes → that range governs style.
4. Record results, return to grouped mode with the measured boundaries for efficient daily use.

Tuning tips: change one set of knobs at a time with a fixed seed; explore mainly within 0.0–1.0
(>1.0 risks artifacts, max 2.0); enable verbose to confirm applied values.

------------------------------------------------------------------------------
Case study: Real functional map of an Anima LoRA
------------------------------------------------------------------------------

To illustrate that "the default table is only a starting point," this section records one complete
scanning run. Results apply only to the LoRA tested, but they reflect a general pattern: on the
Anima/Cosmos architecture, multiple theoretical defaults may be systematically shifted. Scan your own
LoRA to verify.

Method

Control variables: fixed seed / prompt / sampler / steps / CFG; all other LoRAs disabled. One knob is
changed per shot, others kept at 1.0, each compared against the all-1.0 baseline.

Results

By block depth:

  Stage  |  Observed role  |  Theoretical default
  shallow 0-8  |  Motion consistency: hand gestures, facial expression, toe orientation, etc.  |  Composition / anatomy ✗
  middle 9-18  |  Dense-information layer: anatomy + most of the style + part of the composition  |  Transition ✗
  deep 19-27  |  Minor style details / brushwork  |  Main style ✗

By submodule type:

  Knob  |  Observed role  |  Theoretical default
  w_self_attn  |  Detail content + minor anatomy + minor style  |  Anatomy / pose ✗
  w_mlp  |  Largest effect on anatomy; also affects style (strongest single-dimension knob)  |  Style ✗
  w_cross_attn  |  Smallest effect on style; light detail tweaks (safe micro-adjust)  |  Prompt response ⚠
  w_adaln  |  Global influence: all dimensions shift; no specific direction  |  Overall tone ✓

Counter-intuitive finding: w_mlp is the strongest anatomy knob on this LoRA, while w_self_attn has
only minor effect on anatomy — the opposite of the "self_attn governs spatial structure" rule of
thumb. This is explainable from training-data distribution: when anatomy and style are jointly trained
into the same block range and the same submodule type (here, the mlp of the middle stage), adjusting
mlp will shift both at once.

Recipes derived from this map

  Goal  |  Configuration
  Reduce anatomy, accept some style loss  |  w_mlp=0.7, others 1.0
  Large overhaul, soften the overall look  |  middle_weight=0.5-0.7, others 1.0
  Tiny detail tweaks with style nearly intact  |  w_cross_attn=0.7, others 1.0
  Preserve the LoRA's signature look  |  All 1.0; use prompts / negatives to adjust specific elements
  Reduce motion exaggeration while keeping body type  |  shallow_weight=0.5-0.7, others 1.0

Important practical conclusion

On this LoRA, fully separating style from anatomy is not achievable — they are physically trained into
the same range (middle) and the same submodule type (mlp). This is a property of the LoRA's training,
not a limitation of the node. The right strategy is not to seek clean separation but to find an
acceptable trade-off (e.g. a sweet spot at ~60% anatomy reduction with ~20% style loss).

Notes for other Anima LoRA users

The map above is not a universal Anima configuration — every LoRA differs. But compared to the DiT
defaults, it may be closer to how Anima-architecture LoRAs actually behave, and can serve as a priority
hypothesis during scanning:

· Suspect first that middle is the dense-information layer (rather than shallow)
· Suspect first that mlp and self_attn may swap roles vs. the DiT rule of thumb
· w_adaln typically has no specific direction; keep at 1.0 first
· w_cross_attn is typically the safest light-touch knob

You should still run the scanning method yourself for a precise map of your own LoRA.

------------------------------------------------------------------------------
License
------------------------------------------------------------------------------

MIT
