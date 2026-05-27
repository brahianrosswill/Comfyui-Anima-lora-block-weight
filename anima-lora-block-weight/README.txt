==============================================================================
Anima LoRA Block Weight — User Guide (English)
==============================================================================

A ComfyUI node: Anima LoRA Block Weight (menu category loaders)
For LoRAs of Anima (NVIDIA Cosmos architecture, 2B anime model).
Two modes: grouped (three-tier) and per_block (fine-grained).

This document has two parts:
· Part A · Beginner — plain language, copy-paste recipes.
· Part B · Technical — how it works, full parameters, implementation.

------------------------------------------------------------------------------
Part A · Beginner (start here)
------------------------------------------------------------------------------

1. What this node does for you

A LoRA is like a "spice packet with several flavors mixed in." It carries at once:
· Style (linework, coloring, brushwork)
· Composition (framing, camera angle, layout)
· Anatomy / body type (face shape, build, pose tendencies)

A normal LoRA loader gives you only one master switch, so you can't keep one flavor and
drop another. This node gives you separate small switches — e.g. "keep the style, drop
the body type it bakes in," or the reverse.

It's built specifically for Anima. Existing community nodes are written for other models
(Flux, etc.) and do nothing on Anima — this node fixes that.

2. The one key idea: early stages = skeleton, late stages = skin

Anima has 28 internal "stages" (numbered 0–27, called blocks). Just remember this:

  Stage  |  Index  |  Controls
  Early  |  0–8  |  Skeleton: composition, pose, body type, anatomy
  Middle  |  9–18  |  Transition, in between
  Late  |  19–27  |  Skin: style, coloring, texture

Mnemonic: early stages paint the skeleton, late stages paint the skin. Change body/pose →
touch early stages; change style → touch late stages.

Values: 1.0 = keep as is; 0.5 = half effect; 0.0 = fully off; 1.5 = boost (don't exceed
1.5, artifacts appear).

⚠️ The boundaries (0–8 / 9–18 / 19–27) are rough rules of thumb. Your specific LoRA's true
boundaries may differ slightly — measure them with the "scanning method" in Part B.

3. How to use (grouped mode, simplest)

1. Wire the node in, set control_mode to grouped, pick your LoRA.
2. Keep everything at 1.0 and render once as your baseline.
3. Lower the relevant switch to weaken an aspect:
   · shallow_weight = early stages (composition/anatomy)
   · middle_weight = middle stages
   · deep_weight = late stages (style)
4. Render and compare.

4. Copy-paste recipes

Recipe 1 — keep the style, drop its body/pose
    shallow_weight = 0.3   middle_weight = 0.7   deep_weight = 1.0

Recipe 2 — keep pose/composition, replace the style
    shallow_weight = 1.0   middle_weight = 0.7   deep_weight = 0.3

Recipe 3 — make the style stronger
    shallow_weight = 1.0   middle_weight = 1.0   deep_weight = 1.2

Recipe 4 — weaken overall → no layering needed, just set strength_model to 0.6.

Reduce only body type / anatomy, keep style & composition → all three tiers at 1.0,
set only w_self_attn to 0.5.

5. What those four w_ switches actually change (plain version)

The node has four more switches: w_self_attn, w_cross_attn, w_mlp, w_adaln.
They look cryptic, but think of it this way —

Picture the model as a painter. These four map to four kinds of activity in the painter's head:

· w_self_attn = the painter arranging "how parts of the image relate in space"
  hand below the shoulder, symmetric eyes, where the figure stands, how it's framed.
  → Governs: composition, pose, body structure, anatomy, body type.
  → Overdone anatomy/body and want to rein it in? Lower this. Your most-used one.

· w_cross_attn = the painter "reading your request and painting to it"
  you say "a girl in a red dress," the painter must understand and map it to the canvas.
  → Governs: how the model responds to your prompt.
  → LoRA stealing the show, or a concept (e.g. an untagged blue pants) stuck to your trigger
    word and you want to counter it? Lower this. But it dulls response to ALL your prompts —
    it's a blunt knife, not a scalpel.

· w_mlp = the painter's "skill and style"
  how they color, brush texture, line weight. Independent of *what* is drawn — it's *how*.
  → Governs: style, coloring, texture, brushwork.
  → Want to strengthen or weaken the style itself? Touch this.

· w_adaln = the painter's "overall mood/state"
  heavy or light hand today, cool or warm overall — a subtle global tone.
  → Governs: overall atmosphere, contrast (subtle, often barely perceptible).
  → Keep at 1.0; revisit only after the first three feel familiar.

How does it relate to early/middle/late stages?
Stages are "which pass" (shallow = rough, deep = refine); these four are "which hand in that pass."
They multiply. E.g. deep_weight=1.0 with w_mlp=0.5 = "keep the late pass overall, but halve
the 'style hand' within it."

As a beginner you really only need one regularly: w_self_attn (to rein in anatomy/body).
Of the rest, w_mlp for occasional style tweaks, w_cross_attn for concept pollution, forget w_adaln.

Want to grasp the difference in one shot? Fixed seed, render three: all-1.0; only
w_self_attn at 0.3; only w_mlp at 0.3. You'll see one "figure changed, style didn't" and one
"style changed, figure didn't" — instant clarity.

6. Troubleshooting

· No change after adjusting: make sure you changed the weight value, not just the range
  text; enable verbose to confirm it's actually applied; check it isn't overridden by another
  plain loader.
· Broken image / noise: a switch set too high (>1.5), or one tier zeroed while overall
  strength is high. Reset all to 1.0, then deviate gradually.
· Node not in menu: confirm file placement and restart ComfyUI.

------------------------------------------------------------------------------
Part B · Technical (internals, parameters, implementation)
------------------------------------------------------------------------------

1. Why this node exists

Anima is built on the NVIDIA Cosmos architecture; its LoRA weight keys are named:

    lora_unet_blocks_{N}_<submodule>...    # N = 0..27, underscore-separated

Existing LoRA Block Weight nodes are designed for Flux/SDXL and hardcode dot-separated regex
(blocks\.(\d+) matching blocks.0). They cannot match Anima's underscore naming, so every
tensor falls into the "unmatched" branch and gets multiplied by an average weight — layering
silently does nothing.

This node uses blocks_., matching both dot and underscore. Verified against a real
Anima LoRA (28 blocks, 896 weight tensors): 100% matched, zero misses.

2. Scaling principle

Final scale factor per tensor:

    factor = block_weight × submodule_type_weight

· block_weight:
  · grouped mode: from shallow / middle / deep tiers;
  · per_block mode: per-block from the block_weights text field.
· submodule type weights: self_attn / cross_attn / mlp / adaln, each a multiplier, active in
  both modes.
· .alpha tensors are not scaled (scaling up/down already changes the contribution).
· Weighted tensors are loaded via the standard comfy.sd.load_lora_for_models.

3. Roles of stages and submodules

  Dimension  |  Value  |  Role
  Depth · shallow  |  block 0–8  |  Global composition, pose, skeleton, proportions
  Depth · middle  |  block 9–18  |  Subject form, semantic transition
  Depth · deep  |  block 19–27  |  Style, brushwork, texture, coloring
  Type · self_attn  |  w_self_attn  |  Internal spatial structure, composition, anatomy
  Type · cross_attn  |  w_cross_attn  |  Text→image mapping, prompt response
  Type · mlp  |  w_mlp  |  Style, texture, look features
  Type · adaln  |  w_adaln  |  DiT adaptive norm modulation, overall tone

4. Full parameter table

  Parameter  |  Type  |  Default  |  Description
  model / clip  |  —  |  —  |  Model and CLIP inputs
  lora_name  |  dropdown  |  —  |  LoRA file to load
  strength_model  |  float  |  1.0  |  Overall LoRA strength on model
  strength_clip  |  float  |  1.0  |  Overall LoRA strength on CLIP
  control_mode  |  enum  |  grouped  |  grouped tiers / per_block per-block
  shallow_blocks / _weight  |  str / float  |  "0-8" / 1.0  |  Shallow range & factor (grouped)
  middle_blocks / _weight  |  str / float  |  "9-18" / 1.0  |  Middle range & factor (grouped)
  deep_blocks / _weight  |  str / float  |  "19-27" / 1.0  |  Deep range & factor (grouped)
  block_weights  |  multiline  |  ""  |  Per-block weights (per_block)
  w_self_attn / w_cross_attn / w_mlp / w_adaln  |  float  |  1.0  |  Submodule type factors
  default_weight  |  float  |  1.0  |  Factor for unspecified blocks
  verbose  |  bool  |  false  |  Print per-block factors to console

5. per_block text syntax

Three forms, mixable (comma / newline / semicolon as separators), later overrides earlier:

    plain sequence:  1,1,0.5,0.5            # maps to block 0,1,2,3...; pad with default
    index:value:     5:0.3, 27:1.2          # only blocks 5 and 27
    range:value:     0-8:0.3, 19-27:1.0     # set ranges in bulk
    mixed:           0-27:1.0, 12:0, 13:0   # all 1.0, then turn off 12 and 13

Range syntax (for shallow/middle/deep_blocks) accepts all / 0-8 / 0,3,5 / 0-8,19-27.
On overlap, priority is deep > middle > shallow.

6. Scanning method (find your LoRA's true boundaries)

The biggest value of per_block mode is mapping your LoRA's functions:

1. Fix seed, prompt, sampler, steps, CFG (control variables).
2. With per_block, turn off one small range at a time:
       0-27:1.0, 0-3:0      # only 0–3 off
       0-27:1.0, 4-7:0      # only 4–7 off
       ...                  # up to 24-27
3. Compare each against the all-1.0 baseline: if composition collapses → that range governs
   the skeleton; if style changes → that range governs style.
4. Record results, return to grouped mode with the measured boundaries for efficient daily use.

7. Tuning methodology

· Change one set of knobs at a time, with a fixed seed for comparison.
· Explore mainly within 0.0–1.0; >1.0 amplifies and risks artifacts; max is 2.0.
· Enable verbose to confirm applied values match expectations.

8. Installation

    ComfyUI/custom_nodes/anima-lora-block-weight/
        ├── __init__.py
        └── anima_lora_block_weight_pro.py

Restart ComfyUI; find the node under loaders. No third-party dependencies.

9. Note

The block ranges and values here are general DiT rules of thumb, not measured for any specific
LoRA. The tensor-layering logic is verified against a real Anima LoRA; "renders correctly after
loading into the model" should be confirmed in your own environment.
