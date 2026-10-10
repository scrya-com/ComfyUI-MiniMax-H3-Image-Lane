"""Level-of-Token (LoT) for MiniMax H3 in ComfyUI. Does not patch stock Comfy.

``MiniMax H3 LoT Apply`` adds a diffusion-model wrapper that runs Comfy's own H3
forward with one change: the *target* video rows are embedded as a LoT layout
(fewer, larger tokens where detail is low) and the dense velocity is recovered
afterwards. Text, keyframe/reference rows, audio and the block loop are stock.

The trained adapters sample in LoT's scaled latent space (y = x / s per token
size), so the sampled latent must go through ``MiniMax H3 LoT Unscale Latent``
before VAE Decode. Works for T=1 stills (this pack's image nodes) and for video
clips from the stock H3 nodes (any 17k+5 length; the layout repeats per latent
frame and the time positions follow H3's own clock). Start from empty noise
(t2i / t2v / i2v keyframes); a partial-denoise input latent would need the forward
scaling, which is not wired here.

Adapters and LoRAs: https://huggingface.co/johndpope/MiniMax-H3-LoT
(``adapter.safetensors`` -> ``models/lot/``; ``lora.safetensors`` -> ``models/loras/``,
loaded with the stock LoRA loader). Research weights: they need more training.
"""
from __future__ import annotations

import logging
import os

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import folder_paths
import nodes
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    _mod_row,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from comfy_api.latest import io

from .lot_core.h3 import GRID_LAYOUTS, grid_layout, make_h3_adapter
from .lot_core.h3_splice import LotSplice
from .lot_core import layout_sources as sources

log = logging.getLogger(__name__)

LOT_DIR = os.path.join(folder_paths.models_dir, "lot")
os.makedirs(LOT_DIR, exist_ok=True)
folder_paths.add_model_folder_path("lot", LOT_DIR)
WRAPPER_KEY = "minimax_h3_lot"
LotSpec = io.Custom("MINIMAX_H3_LOT")
LotLayoutSpec = io.Custom("MINIMAX_H3_LOT_LAYOUT")


class _LayoutBuilder:
    """A content-aware layout rule: builds the LoT layout for whatever latent shape the sampler uses."""

    def __init__(self, name: str, build):
        self.name = name
        self._build = build
        self._cache: dict[tuple, object] = {}

    def __call__(self, latent_t: int, token_h: int, token_w: int):
        key = (latent_t, token_h, token_w)
        if key not in self._cache:
            self._cache[key] = self._build(latent_t, token_h, token_w)
        return self._cache[key]


class _Lot:
    """The adapter plus the layout rule; builds one splice per latent shape."""

    def __init__(self, adapter, layout_name: str, builder: _LayoutBuilder | None = None):
        self.adapter = adapter
        self.layout_name = builder.name if builder is not None else layout_name
        self.builder = builder
        self._splices: dict[tuple, LotSplice] = {}

    def splice(self, latent_t: int, lat_h: int, lat_w: int, device) -> LotSplice:
        key = (latent_t, lat_h, lat_w)
        if key not in self._splices:
            if self.builder is not None:
                layout = self.builder(latent_t, lat_h // 2, lat_w // 2)
            else:
                layout = grid_layout(self.layout_name, latent_t, lat_h // 2, lat_w // 2)
            self._splices[key] = LotSplice(self.adapter, layout, y_space=True)
            log.info("MiniMax H3 LoT: layout %s, %d of %d tokens (%.2fx)", self.layout_name, layout.count,
                     layout.dense_count, layout.dense_count / max(layout.count, 1))
        if next(self.adapter.parameters()).device != torch.device(device):
            self.adapter.to(device)
        return self._splices[key]


def _lot_forward(model, lot: _Lot, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
    """Comfy's MiniMaxH3Model._forward with a LoT target-video tail (masks unsupported)."""
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    dense = payload.get("layout")
    if dense is None or dense.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        dense = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                             keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    splice = lot.splice(latent_t, lat_h, lat_w, device)
    n_lot = splice.layout.count

    # The target video is the last segment; shorten it to the LoT token count.
    va, vb, kind = dense.segments[-1]
    if kind != "video":
        raise RuntimeError("expected the target video as the last packed segment")
    segments = dense.segments[:-1] + [(va, va + n_lot, "video")]
    seq_len = va + n_lot
    transformer_options["minimax_h3_layout"] = dense

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in segments})
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    # embed: conditions stock, target video through the LoT extent heads
    video_rows = patchify_video(video_x.to(torch.float32), model.patch_size)
    lot_tokens = splice.video_tokens(video_rows, latent_t, lat_h, lat_w)
    lot_embed = splice.embed_rows(lot_tokens).to(dtype)
    cond_rows = model._cond_video_rows(payload, device)
    cond_embed = model.video_patch_proj(cond_rows).to(dtype) if cond_rows is not None else None
    audio_update = dense.audio_update.to(device)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_audio_rows = model._cond_audio_rows(payload, device)
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows
    audio_embed = model.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != model.hidden_size:
        text_states = model.token_refiner(model.condition_proj(text_states), transformer_options=transformer_options)

    h = torch.empty(seq_len, model.hidden_size, dtype=dtype, device=device)
    coff = aoff = 0
    for a, b, kind in segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind == "video":
            h[a:b] = lot_embed
        elif kind in ("cond", "ref_img"):
            h[a:b] = cond_embed[coff:coff + n]
            coff += n
        else:
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if model.use_adaln_curves:
        table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = model.time_embedder(t_vals).to(dtype)

    positions = splice.replace_video_positions(dense.position_ids, latent_t, lat_h, lat_w)
    rope_freqs = rope_rotation_table(model.rope_freqs(positions, device), dtype)

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(model.blocks), device, transformer_options)
    for i, block in enumerate(model.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, malloc_scope="block")
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args, block=block):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"], attention=args.get("attention"))}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "layout": dense, "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")

    final = model.final_layer
    if final.video_out.weight.shape[0] != final.video_out.out_features:
        raise RuntimeError("LoT does not support PDD multi-head checkpoints")
    shift, scale = final.adaln_proj(t_emb)

    def modulate(a, b, row):
        return (final.norm(h[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype))
                + _mod_row(shift, row, shift.dtype)).to(torch.float32)

    aa, ab, _ = next(s for s in segments if s[2] == "audio")
    v = splice.project_rows(modulate(va, va + n_lot, t_row[seg_t["video"]]), lot_tokens, sigma_v)
    a = final.audio_out(modulate(aa, ab, t_row[seg_t["audio"]]))
    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, model.latents_dim, model.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    return [-video_out.to(video_x.dtype), -unpack_audio(a).to(audio_x.dtype)]


def _make_wrapper(lot: _Lot):
    warned = []

    def wrapper(executor, x, timestep, context, transformer_options={}, minimax_payload=None,
                denoise_mask=None, audio_denoise_mask=None, **kwargs):
        if denoise_mask is not None or audio_denoise_mask is not None:
            if not warned:
                log.warning("MiniMax H3 LoT: denoise masks are not supported; running this step dense")
                warned.append(True)
            return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
                            denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
        return _lot_forward(executor.class_obj, lot, x, timestep, context, transformer_options,
                            minimax_payload=minimax_payload, **kwargs)

    return wrapper


def _load_adapter(name: str):
    from safetensors.torch import load_file

    adapter = make_h3_adapter()
    state = load_file(folder_paths.get_full_path_or_raise("lot", name))
    adapter.load_state_dict(state)
    return adapter.float().eval()


class MiniMaxH3LoTApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTApply",
            display_name="MiniMax H3 LoT Apply (research)",
            category="model/minimax",
            description=("Level-of-Token: run the target (still or video clip) with fewer, larger tokens where "
                         "detail is low. Needs a LoT adapter in models/lot and its LoRA. Research weights; send "
                         "the latent through LoT Unscale Latent before VAE Decode."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("adapter", options=folder_paths.get_filename_list("lot")),
                io.Combo.Input("layout", options=list(GRID_LAYOUTS), default="center",
                               tooltip="center: fine in the middle, coarse at the edges (portraits). bands: fine "
                                       "top third, 2×2 middle, 4×2 bottom. uniform2/4: every token 2×2 / 4×4. "
                                       "dense: no shortening (A/B check). Ignored when layout_override is set."),
                LotLayoutSpec.Input("layout_override", optional=True,
                                    tooltip="Content-aware layout from a LoT Layout from Mask / Boxes / Depth / "
                                            "Texture node: fine tokens where the content needs them."),
            ],
            outputs=[io.Model.Output(), LotSpec.Output(display_name="lot")],
        )

    @classmethod
    def execute(cls, model, adapter, layout, layout_override=None) -> io.NodeOutput:
        lot = _Lot(_load_adapter(adapter), layout, layout_override)
        patched = model.clone()
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                                     _make_wrapper(lot))
        return io.NodeOutput(patched, lot)


class MiniMaxH3LoTUnscaleLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTUnscaleLatent",
            display_name="MiniMax H3 LoT Unscale Latent",
            category="model/latent/minimax",
            description="Still or video: map a LoT-sampled latent back to H3's latent space (multiply each token by its extent "
                        "scale) before VAE Decode. Audio is passed through.",
            inputs=[io.Latent.Input("samples"), LotSpec.Input("lot")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, lot) -> io.NodeOutput:
        latent = samples["samples"]
        nested = getattr(latent, "is_nested", False)
        video = latent.tensors[0] if nested else latent
        _b, _c, latent_t, lat_h, lat_w = video.shape
        splice = lot.splice(latent_t, lat_h, lat_w, "cpu")
        unscaled = splice.to_x(video.float().cpu()).to(video.dtype).to(video.device)
        out = dict(samples)
        if nested:
            import comfy.nested_tensor

            out["samples"] = comfy.nested_tensor.NestedTensor((unscaled, *latent.tensors[1:]))
        else:
            out["samples"] = unscaled
        return io.NodeOutput(out)


_BG_TIP = "Extent for tokens outside the region: 4 (4×4, faster) or 2 (2×2, safer quality)."


def _bg(value: str) -> int:
    return int(value[0])


class MiniMaxH3LoTLayoutFromMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTLayoutFromMask",
            display_name="MiniMax H3 LoT Layout from Mask",
            category="model/minimax/lot-layout",
            description=("Semantic-mask layout (LoT paper supplementary): 1×1 tokens on the mask, 2×2 on the "
                         "optional secondary mask, coarse elsewhere. A batch of masks is per frame (a tracked "
                         "object); a single mask applies to every frame. Any resolution; it is pooled to tokens."),
            inputs=[
                io.Mask.Input("mask", tooltip="primary action region"),
                io.Combo.Input("background", options=["4x4", "2x2"], default="4x4", tooltip=_BG_TIP),
                io.Int.Input("dilate", default=1, min=0, max=8, tooltip="grow the fine region by N tokens (32 px each)"),
                io.Float.Input("coverage", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="share of a token's pixels that must be inside the mask"),
                io.Mask.Input("secondary_mask", optional=True, tooltip="secondary objects -> 2×2"),
            ],
            outputs=[LotLayoutSpec.Output(display_name="layout")],
        )

    @classmethod
    def execute(cls, mask, background, dilate, coverage, secondary_mask=None) -> io.NodeOutput:
        m = mask.detach().float().cpu()
        sm = None if secondary_mask is None else secondary_mask.detach().float().cpu()
        return io.NodeOutput(_LayoutBuilder("mask", lambda t, h, w: sources.mask_layout(
            m, t, h, w, secondary=sm, background=_bg(background), coverage=coverage, dilate=dilate)))


class MiniMaxH3LoTLayoutFromBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTLayoutFromBoxes",
            display_name="MiniMax H3 LoT Layout from Boxes",
            category="model/minimax/lot-layout",
            description=("Bounding-box layout: 1×1 tokens inside the boxes, coarse elsewhere. Boxes are "
                         "normalised x0,y0,x1,y1 in [0, 1], separated by ';' (e.g. a face: 0.3,0.1,0.7,0.5)."),
            inputs=[
                io.String.Input("boxes", default="0.3,0.1,0.7,0.5"),
                io.String.Input("secondary_boxes", default="", tooltip="optional boxes -> 2×2"),
                io.Combo.Input("background", options=["4x4", "2x2"], default="4x4", tooltip=_BG_TIP),
                io.Int.Input("dilate", default=0, min=0, max=8),
            ],
            outputs=[LotLayoutSpec.Output(display_name="layout")],
        )

    @staticmethod
    def parse(text: str) -> list[tuple[float, float, float, float]]:
        boxes = []
        for chunk in text.replace("\n", ";").split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            values = [float(v) for v in chunk.replace(" ", "").split(",")]
            if len(values) != 4:
                raise ValueError(f"box {chunk!r}: need x0,y0,x1,y1")
            boxes.append(tuple(min(max(v, 0.0), 1.0) for v in values))
        return boxes

    @classmethod
    def execute(cls, boxes, secondary_boxes, background, dilate) -> io.NodeOutput:
        primary = cls.parse(boxes)
        secondary = cls.parse(secondary_boxes)
        if not primary:
            raise ValueError("give at least one box")

        def build(t, h, w):
            mask = sources.boxes_to_mask(primary, h * 32, w * 32)
            sec = sources.boxes_to_mask(secondary, h * 32, w * 32) if secondary else None
            return sources.mask_layout(mask, t, h, w, secondary=sec, background=_bg(background), dilate=dilate)

        return io.NodeOutput(_LayoutBuilder("boxes", build))


class MiniMaxH3LoTLayoutFromDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTLayoutFromDepth",
            display_name="MiniMax H3 LoT Layout from Depth",
            category="model/minimax/lot-layout",
            description=("Depth-of-field layout: 1×1 tokens near the focus depth, 2×2 one band away, coarse "
                         "beyond. Feed a depth map (e.g. Depth Anything; brighter = nearer) of the first frame or "
                         "of every frame."),
            inputs=[
                io.Image.Input("depth"),
                io.Float.Input("focus", default=0.8, min=0.0, max=1.0, step=0.01,
                               tooltip="depth value in focus (Depth Anything: 1 = nearest)"),
                io.Float.Input("band", default=0.15, min=0.01, max=1.0, step=0.01),
                io.Combo.Input("background", options=["4x4", "2x2"], default="4x4", tooltip=_BG_TIP),
            ],
            outputs=[LotLayoutSpec.Output(display_name="layout")],
        )

    @classmethod
    def execute(cls, depth, focus, band, background) -> io.NodeOutput:
        d = depth.detach().float().cpu()[..., :3].mean(-1)
        return io.NodeOutput(_LayoutBuilder("depth", lambda t, h, w: sources.depth_layout(
            d, t, h, w, focus=focus, band=band, background=_bg(background))))


class MiniMaxH3LoTLayoutFromTexture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTLayoutFromTexture",
            display_name="MiniMax H3 LoT Layout from Texture",
            category="model/minimax/lot-layout",
            description=("Texture-variance layout (the paper's VRS source): 1×1 tokens where the image is busy, "
                         "coarse where it is flat. Use the i2v first frame, a reference, or a quick low-res dense "
                         "preview (two-pass) as the image."),
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("fine_share", default=0.3, min=0.0, max=1.0, step=0.05,
                               tooltip="share of tokens kept 1×1 (busiest first)"),
                io.Float.Input("mid_share", default=0.3, min=0.0, max=1.0, step=0.05,
                               tooltip="next share kept 2×2"),
                io.Combo.Input("background", options=["4x4", "2x2"], default="4x4", tooltip=_BG_TIP),
            ],
            outputs=[LotLayoutSpec.Output(display_name="layout")],
        )

    @classmethod
    def execute(cls, image, fine_share, mid_share, background) -> io.NodeOutput:
        img = image.detach().float().cpu()
        fine_q = 1.0 - fine_share
        mid_q = max(0.0, fine_q - mid_share)
        return io.NodeOutput(_LayoutBuilder("texture", lambda t, h, w: sources.texture_layout(
            img, t, h, w, fine_q=fine_q, mid_q=mid_q, background=_bg(background))))


class MiniMaxH3LoTLayoutPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoTLayoutPreview",
            display_name="MiniMax H3 LoT Layout Preview",
            category="model/minimax/lot-layout",
            description=("Draw the token layout a LoT layout makes for this canvas: red 1×1, orange 2×2, "
                         "grey 4×4 (outlined), one image per latent frame, with the compression in the log."),
            inputs=[
                LotLayoutSpec.Input("layout"),
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("latent_frames", default=1, min=1, max=64,
                             tooltip="1 for a still; 7 for a 22-frame clip"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, layout, width, height, latent_frames) -> io.NodeOutput:
        lay = layout(latent_frames, height // 32, width // 32)
        colours = {1: (0.93, 0.31, 0.43), 2: (0.98, 0.65, 0.25), 4: (0.85, 0.85, 0.85)}
        out = torch.ones(latent_frames, height, width, 3)
        for r in lay.rects:
            y0, x0, y1, x1 = r.u * 32, r.v * 32, (r.u + r.eh) * 32, (r.v + r.ew) * 32
            out[r.t, y0:y1, x0:x1] = torch.tensor(colours.get(max(r.eh, r.ew), (0.6, 0.6, 0.6)))
            out[r.t, y0:y1, x0] = out[r.t, y0, x0:x1] = 0.2
        log.info("MiniMax H3 LoT layout preview: %d of %d tokens (%.2fx)", lay.count, lay.dense_count,
                 lay.dense_count / max(lay.count, 1))
        return io.NodeOutput(out)


LOT_NODES = [MiniMaxH3LoTApply, MiniMaxH3LoTUnscaleLatent, MiniMaxH3LoTLayoutFromMask,
             MiniMaxH3LoTLayoutFromBoxes, MiniMaxH3LoTLayoutFromDepth, MiniMaxH3LoTLayoutFromTexture,
             MiniMaxH3LoTLayoutPreview]
