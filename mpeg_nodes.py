"""MPEG-Diffusion two-track sampler for MiniMax H3 in ComfyUI.

Does not patch stock Comfy. One node runs the paper's sampler on the loaded DiT:

1. ``switch`` Euler steps denoise the whole frame at half resolution.
2. A region head (or a box) picks the subject on the LR token grid.
3. The clean LR estimate is upscaled with the VAE, re-noised, and the remaining
   steps denoise that region at full resolution. LR queries keep seeing only LR
   keys, so the low-resolution track stays the base model. With no trained
   ``f_up`` maps, an HR query sees a coarse token as four copies.
4. The VAE upscale of the final LR latent is pasted under the HR latent.

The two-track LoRA is not trained yet, so this tests the pass and the region.
It does not sharpen a face beyond what the half-resolution track already drew.
Default switch is 16 of 20: on H3's shift 12 that is noise 0.80, the earliest
step where the v3 head finds an off-centre subject and still leaves HR steps.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.minimax.model import (
    PackedLayout,
    _mod_row,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    unpatchify_video,
)
from comfy_api.latest import io

log = logging.getLogger(__name__)

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
AUDIO_CHANNELS = 32
AUDIO_FRAMES = 2          # one still: (10 * 1 + 3) // 6
MID_BLOCK = 25


def _shift_sigma(sigma, shift: float):
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _unshift_sigma(sigma, shift: float):
    return sigma / (shift + sigma * (1.0 - shift))


def _audio_sigma(video_sigma: float) -> float:
    return float(_shift_sigma(_unshift_sigma(video_sigma, VIDEO_SHIFT), AUDIO_SHIFT))


def _sigma_schedule(steps: int, shift: float = VIDEO_SHIFT) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return _shift_sigma(base, shift)


class _Plan:
    """Complete LR grid plus the HR cells of a mask. ``mask`` is ``(T, h, w)`` on the LR token grid."""

    def __init__(self, mask: torch.Tensor, scale: int = 2):
        mask = mask.bool().cpu()
        time, height, width = mask.shape
        self.mask = mask
        self.scale = int(scale)
        self.lr_count = time * height * width
        up = mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        self.hr_index = torch.nonzero(up.flatten()).squeeze(1)
        t, u, v = torch.meshgrid(torch.arange(time), torch.arange(height), torch.arange(width), indexing="ij")
        du, dv = torch.meshgrid(torch.arange(scale), torch.arange(scale), indexing="ij")
        cells = (t.reshape(-1, 1) * (height * scale) + u.reshape(-1, 1) * scale + du.reshape(1, -1)) * (width * scale) \
            + v.reshape(-1, 1) * scale + dv.reshape(1, -1)
        self.cell_order = torch.argsort(cells.flatten())

    @property
    def hr_count(self) -> int:
        return int(self.hr_index.numel())

    def to(self, device) -> "_Plan":
        self.hr_index = self.hr_index.to(device)
        self.cell_order = self.cell_order.to(device)
        return self


def _smooth(field: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return field
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=field.device)
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    flat = field.reshape(-1, 1, field.shape[-2], field.shape[-1])
    flat = F.pad(flat, (radius, radius, 0, 0), mode="reflect")
    flat = F.conv2d(flat, kernel.view(1, 1, 1, -1))
    flat = F.pad(flat, (0, 0, radius, radius), mode="reflect")
    flat = F.conv2d(flat, kernel.view(1, 1, -1, 1))
    return flat.reshape(field.shape)


def _region_mask(field: torch.Tensor, alpha: float) -> torch.Tensor:
    """``field`` ``(T, H, W)`` logits -> bool mask, ``m + alpha (p - m)``."""
    smoothed = _smooth(field.float(), 1.0)
    flat = smoothed.flatten()
    median, top = float(flat.median()), float(torch.quantile(flat, 0.99))
    return smoothed > median + alpha * (top - median)


def _box_mask(box, frames: int, height: int, width: int) -> torch.Tensor:
    x0, y0, x1, y1 = box
    mask = torch.zeros(frames, height, width, dtype=torch.bool)
    mask[:, round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = True
    return mask


def _rope(states: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE. ``states`` is ``(S, heads, dim)``, ``table`` is Comfy's rotation table."""
    pairs = table.shape[-3]
    packed = states.unsqueeze(0)                                 # (1, S, heads, dim)
    rotary = torch.stack((packed[..., :pairs], packed[..., pairs:2 * pairs]), dim=-1)
    rotary = torch.matmul(table, rotary.unsqueeze(-1)).squeeze(-1)
    rotary = torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)
    return torch.cat((rotary, packed[..., 2 * pairs:]), dim=-1)[0]


def _sdpa(query, key, value) -> torch.Tensor:
    """``(S, heads, dim)`` -> ``(S, heads * dim)``."""
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0))
    return out.squeeze(0).transpose(0, 1).reshape(query.shape[0], -1)


def _load_head(name: str):
    from safetensors import safe_open

    path = folder_paths.get_full_path_or_raise("lot", name)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if "score.weight" not in handle.keys():
            raise ValueError(f"{name} is not a region head (no score.weight)")
        meta = handle.metadata() or {}
        weight = handle.get_tensor("score.weight")
        bias = handle.get_tensor("score.bias")
    block = int(float(meta.get("block", MID_BLOCK)))
    return weight, bias, block


def _score(states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    normed = F.layer_norm(states.float(), (states.shape[-1],))
    return F.linear(normed, weight.float().to(normed.device), bias.float().to(normed.device)).squeeze(-1)


def _video_span(layout: PackedLayout):
    return next(span for span in layout.segments if span[2] == "video")


def _time_embed(model, values, device, dtype):
    t_vals = torch.tensor(values, dtype=torch.float32, device=device)
    if model.use_adaln_curves:
        table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    return model.time_embedder(t_vals).to(dtype)


def _silence(sigma_a: float, device, generator) -> torch.Tensor:
    eps = torch.randn(1, AUDIO_CHANNELS, 2, AUDIO_FRAMES, device=device, generator=generator)
    return sigma_a * eps


def _predict_lr(model, video, sigma: float, context, payload, generator, capture_block=None):
    """One base-model step. Returns the velocity in the musubi sign (Comfy's forward is negated)."""
    device = video.device
    audio = _silence(_audio_sigma(sigma), device, generator)
    timestep = torch.tensor([sigma * 1000.0], device=device, dtype=torch.float32)
    captured = {}
    handle = None
    if capture_block is not None:
        def _hook(_module, _inputs, output):
            captured["h"] = output
        handle = model.blocks[capture_block].register_forward_hook(_hook)
    try:
        pred = model._forward([video, audio], timestep, context, {}, minimax_payload=payload)[0]
    finally:
        if handle is not None:
            handle.remove()
    states = None
    if capture_block is not None:
        text_len = context.shape[1]
        _b, _c, frames, lat_h, lat_w = video.shape
        layout = PackedLayout(text_len, frames, lat_h, lat_w, AUDIO_FRAMES)
        start, stop, _kind = _video_span(layout)
        states = captured["h"][start:stop].detach().float()
    return -pred, states


def _two_track_attention(attn, hidden, rope_freqs, *, plan, video_start, hr_table, transformer_options=None):
    del transformer_options
    length = hidden.shape[0]
    heads, dim = attn.heads, attn.head_dim
    query, key, value = attn.qkv_proj(hidden).split(heads * dim, dim=-1)
    query = attn.q_norm(query.view(length, heads, dim))
    key = attn.k_norm(key.view(length, heads, dim))
    value = value.view(length, heads, dim)
    query_r = _rope(query, rope_freqs).to(value.dtype)
    key_r = _rope(key, rope_freqs).to(value.dtype)
    split = video_start + plan.lr_count
    output = hidden.new_zeros(length, heads * dim)
    output[:split] = _sdpa(query_r[:split], key_r[:split], value[:split])
    if plan.hr_count:
        cells = plan.scale * plan.scale
        # Grid keys are pre-RoPE. The HR cells are rotated at the full-resolution positions afterwards.
        lr_key = key[video_start:split]
        lr_value = value[video_start:split]
        grid_key = lr_key.repeat_interleave(cells, dim=0)[plan.cell_order]
        grid_value = lr_value.repeat_interleave(cells, dim=0)[plan.cell_order]
        grid_key = grid_key.index_copy(0, plan.hr_index, key[split:])
        grid_value = grid_value.index_copy(0, plan.hr_index, value[split:])
        grid_key = _rope(grid_key, hr_table).to(value.dtype)
        output[split:] = _sdpa(query_r[split:],
                               torch.cat([key_r[:video_start], grid_key], dim=0),
                               torch.cat([value[:video_start], grid_value], dim=0))
    return attn.out_proj(output)


def _predict_pair(model, lr, hr, plan, sigma: float, context, payload, generator):
    """Velocities ``(v_lr, v_hr)`` of one two-track step. ``v_hr`` is zero outside the region."""
    device = lr.device
    dtype = context.dtype
    text_len = context.shape[1]
    _b, _c, frames, lat_h, lat_w = lr.shape
    layout = PackedLayout(text_len, frames, lat_h, lat_w, AUDIO_FRAMES)
    video_start, _video_stop, _kind = _video_span(layout)
    if plan.lr_count != _video_stop - video_start:
        raise ValueError(f"plan has {plan.lr_count} LR tokens, the latent has {_video_stop - video_start}")
    hr_layout = PackedLayout(text_len, frames, lat_h * 2, lat_w * 2, AUDIO_FRAMES)
    hr_start, hr_stop, _kind = _video_span(hr_layout)
    hr_positions = hr_layout.position_ids[hr_start:hr_stop]
    positions = torch.cat([layout.position_ids, hr_positions[plan.hr_index.cpu()]])

    audio = _silence(_audio_sigma(sigma), device, generator)
    lr_rows = patchify_video(lr.to(torch.float32), model.patch_size)
    hr_rows = patchify_video(hr.to(torch.float32), model.patch_size)[plan.hr_index]
    video = model.video_patch_proj(torch.cat([lr_rows, hr_rows], dim=0)).to(dtype)
    audio_rows = model.audio_patch_proj(pack_audio(audio.to(torch.float32))).to(dtype)
    text_states = context[0]
    segments = layout.segments[:-1] + [(video_start, video_start + plan.lr_count + plan.hr_count, "video")]

    sigma_v = float(sigma)
    t_v = 1.0 - sigma_v
    t_a = 1.0 - _audio_sigma(sigma_v)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a}
    unique_t = sorted({t_v, t_a})
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2}
    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, stop, kind in segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run = 0
            for i in range(1, stop - start + 1):
                if i == stop - start or tags[i] != tags[run]:
                    mod_segments.append((start + run, start + i, row_base + int(tags[run])))
                    run = i
        else:
            mod_segments.append((start, stop, row_base + seg_tag[kind]))

    audio_start, _audio_stop, _audio_kind = next(span for span in layout.segments if span[2] == "audio")
    seq_len = segments[-1][1]
    hidden = torch.empty(seq_len, model.hidden_size, dtype=dtype, device=device)
    hidden[:text_len] = text_states
    hidden[audio_start:video_start] = audio_rows
    hidden[video_start:] = video

    t_emb = _time_embed(model, unique_t, device, dtype)
    rope = rope_rotation_table(model.rope_freqs(positions, device), dtype)
    hr_table = rope_rotation_table(model.rope_freqs(hr_positions, device), dtype)
    plan = plan.to(device)

    def attention(h, rope_freqs=None, transformer_options=None, attn=None):
        return _two_track_attention(attn, h, rope_freqs, plan=plan, video_start=video_start, hr_table=hr_table,
                                    transformer_options=transformer_options)

    for index, block in enumerate(model.blocks):
        comfy.model_management.throw_exception_if_processing_interrupted()
        hidden = block(hidden, t_emb, mod_segments, rope, transformer_options={"block_index": index},
                       attention=lambda h, rope_freqs=None, transformer_options=None, block=block: attention(
                           h, rope_freqs, transformer_options, block.attn))

    final = model.final_layer
    if final.video_out.weight.shape[0] != final.video_dim:
        raise RuntimeError("MPEG-Diffusion node does not support PDD multi-head checkpoints")
    shift, scale = final.adaln_proj(t_emb)
    row = t_row[t_v]
    modulated = (final.norm(hidden[video_start:]) * (1.0 + _mod_row(scale, row, scale.dtype))
                 + _mod_row(shift, row, shift.dtype)).to(torch.float32)
    out = final.video_out(modulated)
    v_lr = unpatchify_video(out[:plan.lr_count], frames, lat_h // 2, lat_w // 2, model.latents_dim, model.patch_size)
    full = out.new_zeros(4 * plan.lr_count, out.shape[-1])
    full[plan.hr_index] = out[plan.lr_count:]
    v_hr = unpatchify_video(full, frames, lat_h, lat_w, model.latents_dim, model.patch_size)
    return v_lr.to(lr.dtype), v_hr.to(hr.dtype)


def _vae_upscale(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode, bicubic ×2, encode. ``latent`` is ``(1, 24, 1, h, w)``."""
    pixels = vae.decode(latent)
    if pixels.ndim == 5:
        pixels = pixels[:, 0]
    scaled = F.interpolate(pixels.movedim(-1, 1), scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
    encoded = vae.encode(scaled.movedim(1, -1))
    if encoded.ndim == 4:
        encoded = encoded.unsqueeze(2)
    return encoded


def _paste(mask: torch.Tensor, hr: torch.Tensor, lr_up: torch.Tensor, feather: float) -> torch.Tensor:
    weight = mask.float().repeat_interleave(4, -2).repeat_interleave(4, -1)
    if feather > 0:
        radius = int(round(feather))
        eroded = -F.max_pool2d(-weight[:, None], 2 * radius + 1, stride=1, padding=radius)
        kernel = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1) / (2 * radius + 1) ** 2
        weight = F.conv2d(F.pad(eroded, (radius,) * 4, mode="replicate"), kernel)[:, 0]
    weight = weight.view(1, 1, 1, *weight.shape).to(hr.device, hr.dtype)
    return weight * hr + (1.0 - weight) * lr_up


def _context_and_payload(model, positive):
    cond, extra = positive[0]
    if cond.ndim == 2:
        cond = cond.unsqueeze(0)
    device = comfy.model_management.get_torch_device()
    cond = model.diffusion_model.preprocess_text_embeds(cond.to(device=device, dtype=model.get_dtype_inference()))
    payload = {"seed": int(extra.get("seed", 0) or 0)}
    tags = extra.get("minimax_token_tags")
    if tags is not None:
        payload["text_token_tags"] = tags
    if extra.get("minimax_refs") or extra.get("minimax_keyframes"):
        raise ValueError("MPEG-Diffusion node samples a text-only still. Reference and keyframe conds are not wired.")
    return cond, payload


def _euler(x, velocity, sigma, sigma_next):
    x0 = x + sigma * velocity.float()
    return x0 + (sigma_next / sigma) * (x - x0), x0


def sample_mpeg(model_patcher, positive, vae, *, width, height, steps, switch, hr_steps, seed, alpha, feather,
                region_name, box):
    if width % 64 or height % 64:
        raise ValueError("width and height must be multiples of 64")
    if not 1 <= switch <= steps:
        raise ValueError(f"switch must be from 1 to steps ({steps})")
    comfy.model_management.load_models_gpu([model_patcher])
    model = model_patcher.model
    dm = model.diffusion_model
    context, payload = _context_and_payload(model, positive)
    device = context.device
    lat_h, lat_w = height // 32, width // 32
    tok_h, tok_w = lat_h // 2, lat_w // 2
    sigmas = _sigma_schedule(steps, float(getattr(dm, "sigma_shift_video", VIDEO_SHIFT)))
    generator = torch.Generator(device=device).manual_seed(seed)
    audio_gen = torch.Generator(device=device).manual_seed(seed + 1)
    lr = torch.randn(1, 24, 1, lat_h, lat_w, device=device, generator=generator)
    head = None
    block = MID_BLOCK
    if region_name != "box":
        weight, bias, block = _load_head(region_name)
        head = (weight, bias)

    total = switch + (hr_steps if hr_steps else max(0, steps - switch))
    bar = comfy.utils.ProgressBar(max(total, 1))
    x0 = lr
    states = None
    for index in range(switch):
        comfy.model_management.throw_exception_if_processing_interrupted()
        sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
        capture = block if head is not None and index == switch - 1 else None
        velocity, got = _predict_lr(dm, lr, sigma, context, payload, audio_gen, capture)
        if got is not None:
            states = got
        lr, x0 = _euler(lr, velocity, sigma, sigma_next)
        bar.update(1)

    if head is not None:
        logits = _score(states, *head).reshape(1, tok_h, tok_w)
        mask = _region_mask(logits, alpha)[0]
    else:
        mask = _box_mask(box, 1, tok_h, tok_w)[0]
    share = float(mask.float().mean())
    log.info("MPEG-Diffusion region share %.3f (%d of %d tokens) switch=%d", share, int(mask.sum()), tok_h * tok_w, switch)

    if not mask.any() or switch >= steps:
        comfy.model_management.unload_all_models()
        full = _vae_upscale(vae, lr)
        return full, mask, share

    plan = _Plan(mask[None])
    sigma = float(sigmas[switch])
    comfy.model_management.unload_all_models()
    up = _vae_upscale(vae, x0)
    comfy.model_management.load_models_gpu([model_patcher])
    hr_gen = torch.Generator(device=device).manual_seed(seed + 2)
    hr = (1.0 - sigma) * up.to(device) + sigma * torch.randn(1, 24, 1, lat_h * 2, lat_w * 2, device=device, generator=hr_gen)
    tail = sigmas[switch:] if not hr_steps else torch.linspace(sigma, 0.0, hr_steps + 1, dtype=torch.float64)
    for index in range(len(tail) - 1):
        comfy.model_management.throw_exception_if_processing_interrupted()
        sigma, sigma_next = float(tail[index]), float(tail[index + 1])
        v_lr, v_hr = _predict_pair(dm, lr, hr, plan, sigma, context, payload, audio_gen)
        lr, _x0 = _euler(lr, v_lr, sigma, sigma_next)
        hr, _x0 = _euler(hr, v_hr, sigma, sigma_next)
        bar.update(1)
    comfy.model_management.unload_all_models()
    lr_up = _vae_upscale(vae, lr)
    latent = _paste(mask, hr, lr_up.to(hr.device), feather)
    return latent, mask, share


def _region_options():
    names = ["box"]
    try:
        names += [name for name in folder_paths.get_filename_list("lot") if "region" in name.lower()]
    except Exception:
        pass
    return names


class MiniMaxH3MpegDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MpegDiffusion",
            display_name="MiniMax H3 MPEG Diffusion",
            category="model/minimax",
            description=("MPEG-Diffusion pass: half the frame at low resolution, then full resolution on a region. "
                         "Put region_head_v3 in models/lot, or use box. No two-track LoRA yet, so this tests the "
                         "pass. Decode the latent with the same VAE. Research weights."),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive", tooltip="From MiniMax H3 Image to Image. Text only, no references."),
                io.Vae.Input("vae", tooltip="Same VAE you will decode with. It upscales the LR latent at the switch."),
                io.Int.Input("width", default=1344, min=64, max=4096, step=64),
                io.Int.Input("height", default=768, min=64, max=4096, step=64),
                io.Int.Input("steps", default=20, min=2, max=50),
                io.Int.Input("switch", default=16, min=1, max=49,
                             tooltip="LR steps before the region goes full resolution. 16 of 20 is noise 0.80 on shift 12."),
                io.Int.Input("hr_steps", default=6, min=0, max=20,
                             tooltip="Steps after the switch, linear in sigma. 0 keeps the leftover schedule steps."),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1),
                io.Combo.Input("region", options=_region_options(), default="box",
                               tooltip="box uses the coordinates below. A region_head*.safetensors file scores the subject."),
                io.Float.Input("box_x0", default=0.25, min=0.0, max=1.0, step=0.01),
                io.Float.Input("box_y0", default=0.20, min=0.0, max=1.0, step=0.01),
                io.Float.Input("box_x1", default=0.75, min=0.0, max=1.0, step=0.01),
                io.Float.Input("box_y1", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("alpha", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Region threshold. Higher keeps less of the frame."),
                io.Float.Input("feather", default=4.0, min=0.0, max=16.0, step=1.0,
                               tooltip="Paste blend, in full-resolution latent pixels."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="region"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, vae, width, height, steps, switch, hr_steps, seed, region,
                box_x0, box_y0, box_x1, box_y1, alpha, feather) -> io.NodeOutput:
        latent, mask, share = sample_mpeg(
            model, positive, vae, width=width, height=height, steps=steps, switch=switch, hr_steps=hr_steps,
            seed=seed, alpha=alpha, feather=feather, region_name=region, box=(box_x0, box_y0, box_x1, box_y1))
        pixels = mask.float().repeat_interleave(64, -2).repeat_interleave(64, -1)
        image = pixels[None, ..., None].expand(-1, -1, -1, 3).clamp(0, 1).cpu()
        log.info("MPEG-Diffusion done region_share=%.3f latent=%s", share, tuple(latent.shape))
        return io.NodeOutput({"samples": latent.cpu()}, image)


MPEG_NODES = [MiniMaxH3MpegDiffusion]
