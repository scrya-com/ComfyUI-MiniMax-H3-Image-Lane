"""T=1 MiniMax H3 *image* lane. Does not patch comfy_extras.nodes_minimax_h3.

Stock Empty / ImageToVideo / ReferenceToVideo stay 17k+5 video.
These nodes temporarily swap temporal_shape for one execute, then restore.
Always length=1, latent T′=1. Load Mamad8 Image VAE on both encode and decode.
"""
from __future__ import annotations

from contextlib import contextmanager

import comfy_extras.nodes_minimax_h3 as h3
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import (
    AUDIO_LATENT_FPS,
    FPS,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    EmptyMiniMaxH3LatentAV,
)
import nodes


def align_frame_count_t1(n):
    if n <= 1:
        return 1
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t_t1(frame_count):
    if frame_count <= 1:
        return 1
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def temporal_shape_t1(length):
    frame_count = align_frame_count_t1(max(1, length))
    duration = frame_count / FPS
    return (
        frame_count,
        video_latent_t_t1(frame_count),
        round(duration * AUDIO_LATENT_FPS),
    )


@contextmanager
def t1_temporal():
    saved = (h3.align_frame_count, h3.video_latent_t, h3.temporal_shape)
    h3.align_frame_count = align_frame_count_t1
    h3.video_latent_t = video_latent_t_t1
    h3.temporal_shape = temporal_shape_t1
    try:
        yield
    finally:
        h3.align_frame_count, h3.video_latent_t, h3.temporal_shape = saved


_VAE_TIP = (
    "IMAGE LANE: load minimax_h3_t1_image_vae_step1597.safetensors. "
    "Do not use the video VAE here (T=1 decode artifacts)."
)


class EmptyMiniMaxH3ImageLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMiniMaxH3ImageLatent",
            display_name="Empty MiniMax H3 Image Latent (T=1)",
            category="model/latent/minimax-image",
            description="T=1 still latent. Stock Empty MiniMax H3 AV Latent is unchanged (video grid).",
            inputs=[
                io.Int.Input("width", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height) -> io.NodeOutput:
        with t1_temporal():
            return EmptyMiniMaxH3LatentAV.execute(width, height, 1)


class MiniMaxH3ImageToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageToImage",
            display_name="MiniMax H3 Image to Image (T=1)",
            category="model/conditioning/minimax-image",
            description="t2va/fl2va at length=1. Stock Image to Video node is unchanged.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip=_VAE_TIP),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height,
                first_frame=None, last_frame=None) -> io.NodeOutput:
        with t1_temporal():
            return MiniMaxH3ImageToVideo.execute(
                clip, vae, prompt, width, height, 1,
                first_frame=first_frame, last_frame=last_frame,
            )


class MiniMaxH3ReferenceToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToImage",
            display_name="MiniMax H3 Reference to Image (T=1)",
            category="model/conditioning/minimax-image",
            description=(
                "ref2va at length=1 for still edits. Stock Reference to Video stays 17k+5. "
                + _VAE_TIP
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip=_VAE_TIP),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Same as stock ref2va.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, ref_image_size="match",
                ref_images=None) -> io.NodeOutput:
        with t1_temporal():
            return MiniMaxH3ReferenceToVideo.execute(
                clip, vae, audio_vae, prompt, width, height, 1, ref_image_size,
                ref_images=ref_images, ref_videos=None, ref_video_audios=None, ref_audios=None,
            )


class MiniMaxH3ImageLaneExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            EmptyMiniMaxH3ImageLatent,
            MiniMaxH3ImageToImage,
            MiniMaxH3ReferenceToImage,
        ]


async def comfy_entrypoint() -> MiniMaxH3ImageLaneExtension:
    return MiniMaxH3ImageLaneExtension()
