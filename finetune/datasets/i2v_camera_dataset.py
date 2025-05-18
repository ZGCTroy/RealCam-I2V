import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)

if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip
import numpy as np
import os
from finetune.utils.camera_utils import get_camera_condition

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseI2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
            self,
            data_root: str,
            metadata_path: str,
            enable_align_factor: bool,
            predict_disparity: bool = False,
            device: torch.device = torch.device("cpu"),
            trainer: "Trainer" = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__()

        self.trainer = trainer
        self.predict_disparity = predict_disparity
        self.data_root = data_root
        self.enable_align_factor = enable_align_factor

        self.train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        self.cache_dir = self.trainer.args.data_root / "cache"
        self.video_latent_dir = self.cache_dir / "video_latent" / self.trainer.args.model_name / self.train_resolution_str
        self.prompt_embeddings_dir = self.cache_dir / "prompt_embeddings"
        self.video_latent_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.all_metadata = np.load(os.path.join(data_root, metadata_path), allow_pickle=True)["arr_0"].tolist()
        self.all_metadata = list(filter(
            lambda x: x['camera_extrinsics'].shape[0] > self.trainer.args.train_resolution[0], 
            self.all_metadata,
        ))
        logger.info(f"Data Count (num_frames > {self.trainer.args.train_resolution[0]}): {len(self.all_metadata)}", main_process_only=True)
        if self.trainer.args.precompute:
            def check(x):
                video_latent_path = self.video_latent_dir.joinpath(Path(x["video_path"]).stem + ".safetensors")
                prompt_embeddings_path = self.prompt_embeddings_dir.joinpath(str(hashlib.sha256(x['long_caption'].encode()).hexdigest()) + ".safetensors")
                return not video_latent_path.exists() or not prompt_embeddings_path.exists()
            self.all_metadata = list(filter(check, self.all_metadata))
        # self.all_metadata = list(self.all_metadata)
        logger.info(f"Data Count (final): {len(self.all_metadata)}", main_process_only=True)

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

    def __len__(self) -> int:
        return len(self.all_metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        metadata = self.all_metadata[index % len(self.all_metadata)]
        video_path = Path(os.path.join(self.data_root, metadata['video_path']))
        image = load_images_from_videos([video_path])[0]
        prompt: str = metadata['long_caption']
        camera_extrinsics = torch.from_numpy(metadata['camera_extrinsics'])  # [F, 4, 4]
        fx, fy, cx, cy = metadata['camera_intrinsics']
        camera_intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])  # 3x3)  # [3, 3]
        num_frames, *_ = camera_extrinsics.shape
        align_factor = torch.full((num_frames, ), metadata['align_factor'] if self.enable_align_factor else 1.0)  # F,

        if self.predict_disparity:
            raise NotImplementedError("Predict disparity is not implemented yet")
            # disparity_video_path = metadata['disparity_videos']
        else:
            disparity_video_path = None

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = self.prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = self.video_latent_dir / (video_path.stem + ".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            try:
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            except:
                pass
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        (
            frames,
            image,
            camera_extrinsics,
            camera_intrinsics,
            align_factor,
            disparity_video,
        ) = self.preprocess(
            video_path,
            image,
            camera_extrinsics,
            camera_intrinsics,
            align_factor,
            disparity_video_path,
            self.trainer.args.use_precompute_video_latents,
        )
        H, W = frames.shape[-2:]
        image = self.image_transform(image)
        frames = self.video_transform(frames)
        video = frames  # F, C, H, W

        if encoded_video_path.exists() and self.trainer.args.use_precompute_video_latents:
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
        else:
            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [F, C, H, W] -> [B, C, F, H, W], value in [-1,1]
            encoded_video = self.encode_video(frames)
            encoded_video = encoded_video[0].to("cpu")  # [1, C, F, H, W] -> [C, F, H, W]
            if self.trainer.args.precompute:
                try:
                    save_file({"encoded_video": encoded_video}, encoded_video_path)
                except:
                    pass
                logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        if not self.trainer.args.precompute:
            # plucker embedding
            cond_frame_index = torch.zeros(1, device=camera_extrinsics.device, dtype=torch.long)
            plucker_embedding, relative_c2w_RT_4x4 = get_camera_condition(  # B 6 F H W
                camera_intrinsics.unsqueeze(0), camera_extrinsics.unsqueeze(0), cond_frame_index,
                trace_scale_factor=1.0, per_frame_scale=align_factor.unsqueeze(0), H=H, W=W,
            )  # [B=1, C=6, F, H, W]
            plucker_embedding = plucker_embedding[0].contiguous()
        else:
            plucker_embedding = None

        ret = {
            "image": image,
            "prompt_embedding": prompt_embedding,  # [C, H, W]
            "prompt": prompt,
            "video": video,  # F, C, H, W
            "encoded_video": encoded_video,  # [C, F//4, H//8, W//8]
            "plucker_embedding": plucker_embedding,  # [B=1, C=6, F, H, W]
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

        if self.predict_disparity:
            raise NotImplementedError("Predict disparity is not implemented yet")

        return ret

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, keep_aspect_ratio: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.keep_aspect_ratio = keep_aspect_ratio

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    def _resize_for_rectangle_crop(self, frames, H, W):
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
            frames = transforms.functional.resize(
                frames,
                size=[H, int(ori_W * H / ori_H)],
            )
        else:
            frames = transforms.functional.resize(
                frames,
                size=[int(ori_H * W / ori_W), W],
            )

        resized_H, resized_W = frames.shape[2], frames.shape[3]
        frames = frames.squeeze(0)

        delta_H = resized_H - H
        delta_W = resized_W - W

        top, left = delta_H // 2, delta_W // 2
        frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

        return frames, resized_H, resized_W

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None, camera_pose_4x4, camera_intrinsics, per_frame_scale, disparity_video_path=None, use_precompute_video_latents=True):
        if disparity_video_path is not None:
            disparity_video = preprocess_video_with_resize(disparity_video_path, self.max_num_frames, self.height, self.width, use_precompute_video_latents)
        else:
            disparity_video = None

        if video_path is not None:
            video, indices = preprocess_video_with_resize(
                video_path, self.max_num_frames, self.height, self.width,
                keep_aspect_ratio=self.keep_aspect_ratio,
                use_precompute_video_latents=use_precompute_video_latents
            )
            if self.keep_aspect_ratio:
                video, resized_H, resized_W = self._resize_for_rectangle_crop(video, self.height, self.width)
            else:
                resized_H, resized_W = video.shape[-2:]

            camera_pose_4x4 = camera_pose_4x4[indices]
            per_frame_scale = per_frame_scale[indices]
            camera_intrinsics = camera_intrinsics.clone()
            cur_H, cur_W = video.shape[-2:]
            camera_intrinsics[0, 0] *= resized_W
            camera_intrinsics[0, 2] *= cur_W
            camera_intrinsics[1, 1] *= resized_H
            camera_intrinsics[1, 2] *= cur_H
            camera_intrinsics = camera_intrinsics.unsqueeze(0).repeat(camera_pose_4x4.shape[0], 1, 1)  # f,3,3
        else:
            video = None
        if image_path is not None and use_precompute_video_latents:
            image = preprocess_image_with_resize(image_path, self.height, self.width, keep_aspect_ratio=self.keep_aspect_ratio)
            if self.keep_aspect_ratio:
                image, resized_H, resized_W = self._resize_for_rectangle_crop(image.unsqueeze(0), self.height, self.width)
        elif not use_precompute_video_latents:
            image = video[0, :, :, :].clone()
        else:
            image = None
        
        return video, image, camera_pose_4x4, camera_intrinsics, per_frame_scale, disparity_video

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseI2VDataset):
    def __init__(
            self,
            video_resolution_buckets: List[Tuple[int, int, int]],
            vae_temporal_compression_ratio: int,
            vae_height_compression_ratio: int,
            vae_width_compression_ratio: int,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
