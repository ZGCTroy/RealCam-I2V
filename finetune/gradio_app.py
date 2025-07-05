import argparse
import json
import os
from uuid import uuid4

import gradio as gr
import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from image_to_video import Image2Video, _resize_for_rectangle_crop, camera_pose_lerp
from demo.preview import Previewer
from demo.prompt_extend import QwenPromptExpander

torch.backends.cuda.matmul.allow_tf32 = True


def load_model_name():
    with open(args.model_meta_path, "r") as f:
        data = json.load(f)

    return list(filter(lambda x: "interp" not in x, data.keys()))


def load_camera_pose_type():
    with open(args.camera_pose_meta_path, "r") as f:
        data = json.load(f)

    return list(data.keys())


def main(args):
    captioner = QwenPromptExpander(args.caption_model_path, device=args.device)
    previewer = Previewer(args.depth_model_path)
    image2video = Image2Video(
        args.result_dir,
        args.model_meta_path,
        args.camera_pose_meta_path,
        args.save_fps,
        device=args.device,
    )

    with gr.Blocks(analytics_enabled=False, css=r"""
        #input_img img {height: 480px !important;}
        #output_vid video {width: auto !important; margin: auto !important;}
    """) as demo:
        gr.Markdown("""
            <div align='center'>
                <h1> RealCam-I2V (CogVideoX-1.5-5B-I2V) </h1>
            </div>
        """)

        with gr.Row(equal_height=True):
            input_image = gr.Image(label="Input Image")
            output_3d = gr.Model3D(label="Camera Trajectory", clear_color=[1.0, 1.0, 1.0, 1.0], visible=False)
            preview_video = gr.Video(label="Preview Video", interactive=False, autoplay=True, loop=True)
            output_video1 = gr.Video(label="New Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)
            output_video2 = gr.Video(label="Previous Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True, visible=False)

        with gr.Row():
            reload_btn = gr.Button("Reload")
            preview_btn = gr.Button("Preview")
            end_btn = gr.Button("Generate")

        with gr.Row(equal_height=True):
            input_text = gr.Textbox(label='Prompt', scale=4)
            caption_btn = gr.Button("Caption")

        with gr.Row():
            negative_prompt = gr.Textbox(label='Negative Prompt', value="Fast movement, jittery motion, abrupt transitions, distorted body, missing limbs, unnatural posture, blurry, cropped, extra limbs, bad anatomy, deformed, glitchy motion, artifacts.")

        with gr.Row(equal_height=True):
            with gr.Column():
                model_name = gr.Dropdown(label='Model Name', choices=load_model_name())
                camera_pose_type = gr.Dropdown(label='Camera Pose Type', choices=load_camera_pose_type())
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**31, step=1, value=12333)

            with gr.Column(scale=2):
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps (DDPM)", value=25)
                    text_cfg = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Text CFG', value=5)
                    camera_cfg = gr.Slider(minimum=1.0, maximum=5.0, step=0.1, label="Camera CFG", value=1.0, visible=False)
                with gr.Row():
                    trace_extract_ratio = gr.Slider(minimum=0, maximum=1.0, step=0.1, label="Trace Extract Ratio", value=1.0)
                    trace_scale_factor = gr.Slider(minimum=0, maximum=5, step=0.1, label="Camera Trace Scale Factor", value=1.0)
                with gr.Row(equal_height=True):
                    frames = gr.Slider(minimum=17, maximum=161, step=16, label="Video Frames", value=49)
                    height = gr.Slider(minimum=256, maximum=1360, step=16, label="Video Height", value=512)
                    width = gr.Slider(minimum=448, maximum=1360, step=16, label="Video Width", value=896)
                    switch_aspect_ratio = gr.Button("Switch HW")
                with gr.Row(visible=False):
                    noise_shaping = gr.Checkbox(label='Enable Noise Shaping', value=False)
                    noise_shaping_minimum_timesteps = gr.Slider(minimum=0, maximum=1000, step=1, label="Noise Shaping Minimum Timesteps", value=900)

        input_image.upload(fn=lambda : "", outputs=[input_text])

        def caption(*inputs):
            image2video.offload_cpu()
            prompt, image = inputs
            return captioner(prompt, tar_lang="en", image=Image.fromarray(image)).prompt

        caption_btn.click(fn=caption, inputs=[input_text, input_image], outputs=[input_text])

        def preview(input_image, camera_pose_type, trace_extract_ratio, sample_frames, sample_height, sample_width):
            frames = rearrange(torch.from_numpy(input_image), "h w c -> c 1 h w")
            frames, resized_H, resized_W = _resize_for_rectangle_crop(frames, sample_height, sample_width)
            frames = rearrange(frames, "c 1 h w -> 1 h w c").numpy()

            with open(args.camera_pose_meta_path, "r", encoding="utf-8") as f:
                camera_pose_file_path = json.load(f)[camera_pose_type]
            camera_data = torch.from_numpy(np.loadtxt(camera_pose_file_path, comments="https"))  # t, 17

            w2cs_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)
            dummy = torch.tensor([[[0, 0, 0, 1]]] * w2cs_3x4.shape[0])
            w2cs_4x4 = torch.cat([w2cs_3x4, dummy], dim=1)
            c2ws_4x4 = w2cs_4x4.inverse()
            c2ws_lerp_4x4 = camera_pose_lerp(c2ws_4x4, round(sample_frames / trace_extract_ratio))[: sample_frames]
            w2cs_lerp_4x4 = c2ws_lerp_4x4.inverse().numpy()

            fx = fy = 0.5 * max(resized_H, resized_W)
            cx = 0.5 * resized_W
            cy = 0.5 * resized_H
            intrinsics = [fx, fy, cx, cy]
            depths = previewer.estimate_depths(frames, intrinsics)
            previews = previewer.render_previews(frames[0], depths[0], intrinsics, w2cs_lerp_4x4)

            uid = uuid4().fields[0]
            preview_path = f"{args.result_dir}/preview_{uid:08x}.mp4"
            os.makedirs(args.result_dir, exist_ok=True)
            imageio.mimsave(preview_path, previews, fps=args.save_fps)

            return preview_path

        preview_btn.click(
            fn=preview,
            inputs=[input_image, camera_pose_type, trace_extract_ratio, frames, height, width],
            outputs=[preview_video],
        )

        def generate(*inputs):
            *inputs, frames, height, width = inputs
            captioner.offload_cpu()
            return image2video.get_image(*inputs, (frames, height, width))

        end_btn.click(
            fn=generate,
            inputs=[model_name, input_image, input_text, negative_prompt, camera_pose_type, preview_video, steps, trace_extract_ratio, trace_scale_factor, camera_cfg, text_cfg, seed, noise_shaping, noise_shaping_minimum_timesteps, frames, height, width],
            outputs=[output_video1],
        )

        end_btn.click(fn=lambda x: x, inputs=[output_video1], outputs=[output_video2])

        reload_btn.click(
            fn=lambda: (gr.Dropdown(choices=load_model_name()), gr.Dropdown(choices=load_camera_pose_type())),
            outputs=[model_name, camera_pose_type]
        )

        switch_aspect_ratio.click(fn=lambda x: x, inputs=[height], outputs=[width])
        switch_aspect_ratio.click(fn=lambda x: x, inputs=[width], outputs=[height])

    return demo


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_fps", type=int, default=16)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--model_meta_path", type=str, default="demo/models.json")
    parser.add_argument("--example_meta_path", type=str, default="demo/examples.json")
    parser.add_argument("--camera_pose_meta_path", type=str, default="demo/camera_poses.json")
    parser.add_argument("--depth_model_path", type=str, default="pretrained/Metric3D/metric_depth_vit_large_800k.pth")
    parser.add_argument("--caption_model_path", type=str, default="pretrained/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8080)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()

    main(args).launch(server_name=args.server_name, server_port=args.server_port, allowed_paths=["demo"])
