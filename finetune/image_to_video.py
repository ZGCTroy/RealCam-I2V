import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import importlib
import json
import os
from functools import partial
from pprint import pprint
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils.export_utils import export_to_video
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from torch import Tensor
from torchvision import transforms
from transformers import AutoTokenizer, T5EncoderModel
from torchcodec.decoders import VideoDecoder


def relative_pose(rt: Tensor, mode, ref_index) -> Tensor:
    '''
    :param rt: F,4,4
    :param mode: left or right
    :return:
    '''
    if mode == "left":
        rt = rt[ref_index].inverse() @ rt
    elif mode == "right":
        rt = rt @ rt[ref_index].inverse()
    return rt


def camera_pose_lerp(c2w: Tensor, target_frames: int):
    weights = torch.linspace(0, c2w.size(0) - 1, target_frames, dtype=c2w.dtype)
    left_indices = weights.floor().long()
    right_indices = weights.ceil().long()

    return torch.lerp(c2w[left_indices], c2w[right_indices], weights.unsqueeze(-1).unsqueeze(-1).frac())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _resize_for_rectangle_crop(frames, H, W):
    '''
    :param frames: C,F,H,W
    :param image_size: H,W
    :return: frames: C,F,crop_H,crop_W;  camera_intrinsics: F,3,3
    '''
    ori_H, ori_W = frames.shape[-2:]

    # if ori_W / ori_H < 1.0:
    #     tmp_H, tmp_W = int(H), int(W)
    #     H, W = tmp_W, tmp_H

    if ori_W / ori_H > W / H:
        frames = transforms.functional.resize(frames, size=[H, int(ori_W * H / ori_H)])
    else:
        frames = transforms.functional.resize(frames, size=[int(ori_H * W / ori_W), W])

    resized_H, resized_W = frames.shape[2], frames.shape[3]
    frames = frames.squeeze(0)

    delta_H = resized_H - H
    delta_W = resized_W - W

    top, left = delta_H // 2, delta_W // 2
    frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

    return frames, resized_H, resized_W


def _resize(frames, H, W):
    '''
    :param frames: C,F,H,W
    :param image_size: H,W
    :return: frames: C,F,crop_H,crop_W;  camera_intrinsics: F,3,3
    '''
    frames = transforms.functional.resize(frames, size=[H, W])

    resized_H, resized_W = frames.shape[2], frames.shape[3]
    frames = frames.squeeze(0)

    return frames, resized_H, resized_W


class Image2Video:
    def __init__(
        self,
        result_dir: str = "results",
        model_meta_path: str = "models.json",
        camera_pose_meta_path: str = "camera_poses.json",
        save_fps: int = 16,
        device: str = "cuda",
    ):
        self.result_dir = result_dir
        self.model_meta_file = model_meta_path
        self.camera_pose_meta_path = camera_pose_meta_path
        self.save_fps = save_fps
        self.device = torch.device(device)
        self.pipe = None

    def init_model(self, model_name):
        from models.camera_controller.cogvideox_with_controlnetxs import CogVideoXTransformer3DModel
        from models.camera_controller.controlnetxs import ControlnetXs

        with open(self.model_meta_file, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)[model_name]
        pretrained_model_path = model_metadata["pretrained_model_path"]
        controlnetxs_model_path = model_metadata["controlnetxs_model_path"]

        self.transformer = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.controlnetxs = ControlnetXs("models/camera_controller/CogVideoX1.5-5B-I2V", self.transformer.config)
        self.controlnetxs.load_state_dict(torch.load(controlnetxs_model_path)['module'], strict=True)
        self.controlnetxs.to(torch.bfloat16)
        # self.controlnetxs.to(torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.bfloat16)
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

        self.controlnetxs.eval()
        self.text_encoder.eval()
        self.vae.eval()
        self.transformer.eval()

        self.prepare_models()

    def prepare_models(self) -> None:
        if self.vae is not None:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        if self.controlnetxs.vae_encoder is not None:
            self.controlnetxs.vae_encoder.enable_slicing()
            self.controlnetxs.vae_encoder.enable_tiling()

    def init_pipe(self):
        from models.camera_controller.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline

        self.pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.tokenizer,
            text_encoder=None,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler
        )
        self.pipe.scaling_flag = True
        self.pipe.to(self.device)

    def offload_cpu(self):
        if hasattr(self, "transformer"):
            self.transformer.cpu()
        if hasattr(self, "controlnetxs"):
            self.controlnetxs.cpu()
        if hasattr(self, "text_encoder"):
            self.text_encoder.cpu()
        if hasattr(self, "vae"):
            self.vae.cpu()

        torch.cuda.empty_cache()

    def prepare_rotary_positional_embeddings(
            self,
            height: int,
            width: int,
            num_frames: int,
            transformer_config: dict,
            vae_scale_factor_spatial: int,
            device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

    def validation_step(self, input_kwargs: dict[str]) -> torch.Tensor:
        """
        Return the input_kwargs that needs to be saved. For videos, the input_kwargs format is list[PIL],
        and for images, the input_kwargs format is PIL
        image: shape=(1,c,h,w),  value in [0, 1]
        """
        plucker_embedding = input_kwargs["plucker_embedding"]
        image = input_kwargs["image"]
        H, W = image.shape[-2:]

        # camera
        plucker_embedding = plucker_embedding.to(self.controlnetxs.vae_encoder.device, dtype=self.controlnetxs.vae_encoder.dtype)  # [C=6, F, H, W]
        latent_plucker_embedding_dist = self.controlnetxs.vae_encoder.encode(plucker_embedding).latent_dist  # B,C=6,F,H,W --> B,128,(F-1)//4+1,H//4,W//4
        latent_plucker_embedding = latent_plucker_embedding_dist.sample()
        latent_plucker_embedding = latent_plucker_embedding.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] to [B, F, C, H, W]
        latent_plucker_embedding = latent_plucker_embedding.repeat(2, 1, 1, 1, 1)  # cfg

        patch_size_t = self.transformer.config.patch_size_t
        if patch_size_t is not None:
            ncopy = patch_size_t - latent_plucker_embedding.shape[1] % patch_size_t

            if ncopy > 0:
                # Copy the first frame ncopy times to match patch_size_t
                first_frame = latent_plucker_embedding[:, :1, :, :, :]  # Get first frame [B, F, C, H, W]
                latent_plucker_embedding = torch.cat([first_frame.repeat(1, ncopy, 1, 1, 1), latent_plucker_embedding], dim=1)

                if 'latent_scene_frames' in input_kwargs:
                    input_kwargs['latent_scene_frames'] = torch.cat([
                            input_kwargs['latent_scene_frames'][:, :1, :, :, :].repeat(1, ncopy, 1, 1, 1),
                            input_kwargs['latent_scene_frames']
                        ],
                        dim=1
                    )

                    input_kwargs['latent_scene_mask'] = torch.cat([
                            input_kwargs['latent_scene_mask'][:, :1, :, :, :].repeat(1, ncopy, 1, 1, 1),
                            input_kwargs['latent_scene_mask']
                        ],
                        dim=1
                    )

            assert latent_plucker_embedding.shape[1] % patch_size_t == 0

        num_latent_frames = latent_plucker_embedding.shape[1]
        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        rotary_emb_for_controlnetxs = (
            self.prepare_rotary_positional_embeddings(
                height=H,
                width=W,
                num_frames=num_latent_frames,
                transformer_config=self.controlnetxs.transformer.config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.device,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        self.init_pipe()
        original_forward = self.pipe.transformer.forward
        self.pipe.transformer.forward = partial(
            self.pipe.transformer.forward,
            controlnetxs=self.controlnetxs,
            latent_plucker_embedding=latent_plucker_embedding,
            image_rotary_emb_for_controlnetxs=rotary_emb_for_controlnetxs,
        )

        forward_kwargs = dict(
            num_frames=input_kwargs["video_length"],
            height=H,
            width=W,
            prompt_embeds=input_kwargs["prompt_embedding"],
            negative_prompt_embeds=input_kwargs["negative_prompt_embedding"],
            image=image.to(self.device).to_dense(),
            num_inference_steps=input_kwargs['num_inference_steps'],
            guidance_scale=input_kwargs['text_cfg'],
            noise_shaping=input_kwargs['noise_shaping'],
            noise_shaping_minimum_timesteps = input_kwargs['noise_shaping_minimum_timesteps'],
            latent_scene_frames = input_kwargs.get('latent_scene_frames', None),      # B,F,C,H,W
            latent_scene_mask = input_kwargs.get('latent_scene_mask', None),
            generator = input_kwargs['generator'],
        )

        video_generate = self.pipe(**forward_kwargs).frames[0]
        self.pipe.transformer.forward = original_forward
        return video_generate

    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.text_encoder(prompt_token_ids.to(self.device))[0]
        return prompt_embedding.to(torch.bfloat16).to(self.device)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        video = video.to(self.vae.device, dtype=self.vae.dtype)
        latent_dist = self.vae.encode(video).latent_dist
        latent = latent_dist.sample() * self.vae.config.scaling_factor
        return latent.to(torch.bfloat16).to(self.device)

    @torch.inference_mode()
    def get_image(
        self,
        model_name: str,
        ref_img: np.ndarray,
        prompt: str,
        negative_prompt: str,
        camera_pose_type: str,
        preview_video: str = None,
        steps: int = 25,
        trace_extract_ratio: float = 1.0,
        trace_scale_factor: float = 1.0,
        camera_cfg: float = 1.0,
        text_cfg: float = 6.0,
        seed: int = 123,
        noise_shaping: bool = False,
        noise_shaping_minimum_timesteps: int = 800,
        video_shape: tuple[int, int, int] = (81, 768, 1360),
        resize_for_rectangle_crop: bool = True,
    ):
        if self.pipe is None:
            self.init_model(model_name)

        video_length, self.sample_height, self.sample_width = video_shape
        print(video_length, self.sample_height, self.sample_width)

        seed_everything(seed)
        input_kwargs = {
            'video_length': video_length,
            'camera_cfg': camera_cfg,
            'num_inference_steps': steps,
            'text_cfg': text_cfg,
            'noise_shaping': noise_shaping,
            'noise_shaping_minimum_timesteps': noise_shaping_minimum_timesteps,
            'generator': torch.Generator(device=self.device).manual_seed(seed)
        }

        ref_img = rearrange(torch.from_numpy(ref_img), 'h w c -> c 1 h w')
        if resize_for_rectangle_crop:
            ref_img, resized_H, resized_W = _resize_for_rectangle_crop(
                ref_img, self.sample_height, self.sample_width,
            )
        else:
            ref_img, resized_H, resized_W = _resize(
                ref_img, self.sample_height, self.sample_width,
            )
        ref_img = rearrange(ref_img / 255, "c 1 h w -> 1 c h w")
        H, W = ref_img.shape[-2:]

        input_kwargs["image"] = ref_img.to(self.device).to(torch.bfloat16)

        with open(self.camera_pose_meta_path, "r", encoding="utf-8") as f:
            camera_pose_file_path = json.load(f)[camera_pose_type]
        camera_data = torch.from_numpy(np.loadtxt(camera_pose_file_path, comments="https"))  # t, -1

        fx = 0.5 * max(resized_H, resized_W)
        fy = fx
        cx = 0.5 * W
        cy = 0.5 * H
        intrinsics_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1.0]
        ])

        w2cs_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
        dummy = torch.tensor([[[0, 0, 0, 1]]] * w2cs_3x4.shape[0])  # [t, 1, 4]
        w2cs_4x4 = torch.cat([w2cs_3x4, dummy], dim=1)  # [t, 4, 4]
        c2ws_4x4 = w2cs_4x4.inverse()  # [t, 4, 4]
        c2ws_lerp_4x4 = camera_pose_lerp(c2ws_4x4, round(video_length / trace_extract_ratio))[: video_length]

        from utils.camera_utils import get_camera_condition
        plucker_embedding, relative_c2w_RT_4x4 = get_camera_condition(
            H, W, intrinsics_matrix[None, None], c2ws_lerp_4x4[None], mode="c2w",
            cond_frame_index=0, align_factor=trace_scale_factor
        )  # [B=1, C=6, F, H, W]

        input_kwargs["plucker_embedding"] = plucker_embedding.to(self.device).to(torch.bfloat16)

        uid = uuid4().fields[0]
        if noise_shaping:
            scene_frames = VideoDecoder(preview_video, device=str(self.device))[:]
            scene_frames = rearrange(scene_frames / 255 * 2 - 1, "t c h w -> c t h w")  # c,f,h,w, value in [-1, 1]
            latent_scene_frames = self.encode_video(scene_frames.unsqueeze(0))  # b=1,c,f,h,w
            input_kwargs['latent_scene_frames'] = latent_scene_frames.permute(0, 2, 1, 3, 4)    # b=1,c,f,h,w --> b=1,f,c,h,w

            from models.camera_controller.utils import apply_thresholded_conv
            scene_mask = (scene_frames < 1).float().amax(0, keepdim=True)  # c,f,h,w --> 1,f,h,w
            scene_mask = apply_thresholded_conv(scene_mask, kernel_size=5, threshold=1.0)  # 1,f,h,w
            latent_scene_mask = torch.cat([
                F.interpolate(scene_mask[:, :1].unsqueeze(1), (1, H // 8, W // 8), mode="trilinear", align_corners=True),
                F.interpolate(scene_mask[:, 1:].unsqueeze(1), ((video_length - 1) // 4, H // 8, W // 8), mode="trilinear", align_corners=True)
            ], dim=2).bool()
            input_kwargs['latent_scene_mask'] = latent_scene_mask.permute(0, 2, 1, 3, 4)

        self.vae.cpu()
        self.transformer.cpu()
        self.controlnetxs.cpu()
        torch.cuda.empty_cache()

        self.text_encoder.to(self.device)
        input_kwargs |= {
            "prompt_embedding": self.encode_text(prompt),
            "negative_prompt_embedding": self.encode_text(negative_prompt),
        }

        self.text_encoder.cpu()
        torch.cuda.empty_cache()

        self.vae.to(self.device)
        self.transformer.to(self.device)
        self.controlnetxs.to(self.device)

        generated_video = self.validation_step(input_kwargs)
        video_path = f"{self.result_dir}/{model_name}_{uid:08x}.mp4"
        os.makedirs(self.result_dir, exist_ok=True)
        export_to_video(generated_video, video_path, fps=self.save_fps)

        torch.cuda.empty_cache()

        return video_path
