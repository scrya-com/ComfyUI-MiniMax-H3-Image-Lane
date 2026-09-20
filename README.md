# ComfyUI MiniMax H3 Image Lane (T=1)

T=1 still nodes for MiniMax H3. **Does not patch** stock
`comfy_extras/nodes_minimax_h3.py`. Video Ref2VA stays on the 17k+5 grid
(`length` 5, 22, 124…).

Based on the Comfy T=1 recipe in
[Mamad8/MiniMax-H3-Image-VAE#4](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/discussions/4)
and the `align_frame_count(1) → 1` helpers from
[discussion #1](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/discussions/1).
Those threads patch core Comfy. This pack is a **sidecar**: new node ids,
stock nodes untouched.

| Lane | Nodes | VAE | Length |
|---|---|---|---|
| **Video** (stock) | Empty MiniMax H3 AV Latent, Reference to Video | `minimax_h3_video_vae_fp16` | 5, 22, 124… |
| **Image** (this pack) | Empty MiniMax H3 Image Latent, Reference to Image, Image to Image | [`minimax_h3_t1_image_vae_step1597`](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE) | always 1 |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/johndpope/ComfyUI-MiniMax-H3-Image-Lane.git
```

Restart Comfy. Search the menu for **`minimax-image`**.

Load `workflows/t1_ref2va_image.json` (or drag it onto the canvas).

## Weights (not in this repo)

| File | Role |
|---|---|
| [minimax_h3_t1_image_vae_step1597.safetensors](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE) | visual encode **and** decode |
| [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3) | audio VAE (Ref2VA still wants it) |
| MiniMax H3 DiT (hybrid ref2va or official) | sampler |
| Qwen3-VL MiniMax CLIP | text / refs |

Use the **Image VAE** on both the Reference-to-Image `vae` input and **VAE Decode**.
The frozen encoder matches the video VAE (`||z_image − z_video|| = 0`). Only the
decoder is image-tuned. Video VAE at T=1 is the fringe / line path (~0.11 pixel
L1 vs ~0.006).

## Nodes

- **Empty MiniMax H3 Image Latent (T=1)** — `z` shape `[B, 24, 1, H/16, W/16]`
- **MiniMax H3 Reference to Image (T=1)** — ref2va, `length` locked to 1
- **MiniMax H3 Image to Image (T=1)** — t2va / first-frame, `length` locked to 1

Each execute temporarily swaps `temporal_shape` on the stock module, then
restores it. Video graphs keep snapping `length=1` → 5 frames / T′=2.

## License

Apache-2.0. MiniMax H3 weights follow MiniMax’s terms. Image VAE: Mamad8 /
MiniMax.
