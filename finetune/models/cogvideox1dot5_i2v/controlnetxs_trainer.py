import json
import math
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
from functools import partial


from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    # CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer, _DTYPE_MAP
from finetune.utils import unwrap_model

from ..utils import register
import os

from finetune.trainer import logger
from finetune.datasets.i2v_camera_dataset import I2VDatasetWithResize
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from finetune.models.camera_controller.controlnetxs import ControlnetXs
from finetune.models.camera_controller.cogvideox_with_controlnetxs import CogVideoXTransformer3DModel
from diffusers.video_processor import VideoProcessor
from torch.utils.data import DataLoader, Dataset
from finetune.schemas import Args, Components, State



class CogVideoX1dot5I2VControlnetXsTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    def __init__(self, args: Args) -> None:
        super().__init__(args)

        if not args.precompute and args.allow_switch_hw and self.accelerator.process_index % 2 == 0:
            f, h, w = self.args.train_resolution
            self.args.train_resolution = (f, w, h)
            self.state.train_frames = args.train_resolution[0]
            self.state.train_height = args.train_resolution[1]
            self.state.train_width = args.train_resolution[2]


    def get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        components.controlnetxs = ControlnetXs("models/camera_controller/CogVideoX1.5-5B-I2V", components.transformer.config)
        self.state.controlnetxs_transformer_config = components.controlnetxs.transformer.config

        return components

    @override
    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        if self.components.controlnetxs.vae_encoder is not None:
            if self.args.enable_slicing:
                self.components.controlnetxs.vae_encoder.enable_slicing()
            if self.args.enable_tiling:
                self.components.controlnetxs.vae_encoder.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    @override
    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "i2v":
            self.dataset = I2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        elif self.args.model_type == "t2v":
            self.dataset = T2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        if self.args.precompute:
            logger.info("Precomputing latent for video and prompt embedding ...")
            tmp_data_loader = torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=0,
                pin_memory=self.args.pin_memory,
                shuffle=True
            )
            tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
            for _ in tqdm(tmp_data_loader):
                ...
            self.accelerator.wait_for_everyone()
            logger.info("Precomputing latent for video and prompt embedding ... Done")
            return

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        # self.components.transformer.set_controlnetxs_like_adapter(
        #     unwrap_model(self.accelerator, self.components.controlnetxs)
        # )
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=self.components.transformer,
            scheduler=self.components.scheduler,
        )
        return pipe

    def decode_video(self, latent: torch.Tensor, video_processor) -> torch.Tensor:
        # shape of input video: [B, C,F//4, H//8, W//8]
        vae = self.components.vae
        latent = latent.to(vae.device, dtype=vae.dtype)

        video = vae.decode(latent / vae.config.scaling_factor).sample
        video = video_processor.postprocess_video(video=video, output_type="pil")
        return video

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos": [], "prompt_embeddings": [], "images": [], "plucker_embeddings": [], "prompts": [], "videos": []
        }

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            plucker_embedding = sample["plucker_embedding"]
            video = sample["video"]
            prompt = sample["prompt"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embeddings"].append(prompt_embedding)
            if not self.args.precompute:
                ret["plucker_embeddings"].append(plucker_embedding)
            ret["images"].append(image)
            ret["videos"].append(video)
            ret["prompts"].append(prompt)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        ret["images"] = torch.stack(ret["images"])
        if not self.args.precompute:
            ret["plucker_embeddings"] = torch.stack(ret["plucker_embeddings"])
        ret["videos"] = torch.stack(ret["videos"])

        return ret

    @override
    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self._prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["controlnetxs"] + self.UNLOAD_LIST
        self._move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()
            self.components.controlnetxs.enable_gradient_checkpointing()

        self.components.controlnetxs.train()
        self.components.controlnetxs.requires_grad_(True)
        # self.components.controlnetxs.set_main_transformer(self.components.transformer)

    @override
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.controlnetxs], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        # for idx, (n, p) in enumerate(self.components.controlnetxs.named_parameters()):
        #     if p.requires_grad:
        #         print(idx, n)

        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.controlnetxs.parameters()))
        trainable_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [trainable_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
                self.accelerator.state.deepspeed_plugin is not None
                and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
                self.accelerator.state.deepspeed_plugin is not None
                and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps
        num_warmup_steps = self.args.lr_warmup_steps

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def prepare_for_training(self) -> None:
        self.components.controlnetxs, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.controlnetxs, self.optimizer, self.data_loader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    @override
    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
                self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.controlnetxs.train()
            models_to_accumulate = [self.components.controlnetxs]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss, loss_log = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.controlnetxs.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.controlnetxs.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    # import ipdb
                    # ipdb.set_trace()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # print('0-0', self.components.controlnetxs.transformer.patch_embed.proj.weight.data[0][:5])
                    # print('0-1', self.components.controlnetxs.transformer.patch_embed.proj.weight.grad)
                    # print('1', unwrap_model(self.accelerator, self.components.controlnetxs).transformer.patch_embed.proj.weight.data[0][:5])
                    # print('1', unwrap_model(self.accelerator, self.components.controlnetxs).transformer.patch_embed.proj.weight.grad)
                    progress_bar.update(1)
                    global_step += 1
                    self._maybe_save_checkpoint(global_step)

                    for key, value in loss_log.items():
                        logs[key] = value.item()
                    try:
                        logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                        progress_bar.set_postfix(logs)
                    except:
                        pass

                    # Maybe run validation
                    should_run_validation = self.args.do_validation and global_step % self.args.validation_steps == 0
                    if should_run_validation:
                        del loss
                        free_memory()
                        try:
                            self.validate(global_step)
                        except:
                            print('fail to validate')

                    accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self._maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embeddings"]
        latent = batch["encoded_videos"]
        video = batch["videos"]
        prompt = batch["prompts"]
        images = batch["images"]
        plucker_embedding = batch["plucker_embeddings"]  # B, C, F, H, W

        self.state.image = images[0]  # C, H, W;  value in [0, 255]
        self.state.video = video[0]  # F, C, H, W; value in [-1, 1]
        self.state.prompt_embedding = prompt_embedding[0]  # L, D
        self.state.prompt = prompt[0]
        self.state.plucker_embedding = plucker_embedding[0]  # C=6, F, H, W


        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        if self.args.time_sampling_type == "truncated_normal":
            time_sampling_dict = {
                'mean': self.args.time_sampling_mean,
                'std': self.args.time_sampling_std,
                'a': self.args.camera_condition_start_timestep / self.components.scheduler.config.num_train_timesteps,
                'b': 1.0,
            }
            timesteps = torch.nn.init.trunc_normal_(
                torch.empty(batch_size, device=self.accelerator.device), **time_sampling_dict
            ) * self.components.scheduler.config.num_train_timesteps

        timesteps = timesteps.long()

        self.state.timestep = timesteps[0].item()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        controlnetxs_transformer_config = self.state.controlnetxs_transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        rotary_emb_for_controlnetxs = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=controlnetxs_transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        if self.args.enable_gft_training:
            camera_condition_gft_beta = torch.ones(timesteps.shape[0]).uniform_(0.2, 1.0).to(self.accelerator.device)  # [0.2, 1.0]
        else:
            camera_condition_gft_beta = torch.ones(timesteps.shape[0]).to(self.accelerator.device)  # [1.0, 1.0]
        predicted_results = self.components.controlnetxs(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            image_rotary_emb_for_controlnetxs=rotary_emb_for_controlnetxs,
            return_dict=True,
            plucker_embedding=plucker_embedding,  # B,F,C,H,W
            main_transformer=self.components.transformer,
            camera_condition_gft_beta = camera_condition_gft_beta,
            camera_condition_dropout=0.1 if self.args.enable_gft_training else 0.0
        )
        predicted_noise = predicted_results['sample']

        if self.args.enable_gft_training:
            with torch.inference_mode():
                uncond_predicted_results = self.components.controlnetxs(
                    hidden_states=latent_img_noisy,
                    encoder_hidden_states=prompt_embedding,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=rotary_emb,
                    image_rotary_emb_for_controlnetxs=rotary_emb_for_controlnetxs,
                    return_dict=True,
                    plucker_embedding=plucker_embedding,  # B,F,C,H,W
                    main_transformer=self.components.transformer,
                    camera_condition_gft_beta=torch.ones_like(camera_condition_gft_beta),
                    camera_condition_dropout=1.0
                )
                uncond_predicted_noise = uncond_predicted_results['sample']
            predicted_noise = camera_condition_gft_beta[:, None, None, None, None] * predicted_noise \
                                  + (1-camera_condition_gft_beta[:, None, None, None, None]) * uncond_predicted_noise


        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
        loss_log = {
            'diffusion_loss': loss.detach(),
        }

        return loss, loss_log

    @override
    def prepare_for_validation(self):
        pass

    @override
    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        self.components.controlnetxs.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self._move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["controlnetxs", "transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []

        image = self.state.image  # C, H, W;  value in [0, 255]
        video = self.state.video  # F, C, H, W; value in [-1, 1]
        prompt_embeds = self.state.prompt_embedding  # L, D
        prompt = self.state.prompt
        plucker_embedding = self.state.plucker_embedding  # C=6, F, H, W

        if image is not None:
            # Convert image tensor (C, H, W) to PIL images
            image = ((image * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)

        if video is not None:
            # Convert video tensor (F, C, H, W) to list of PIL images
            video = ((video * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
            video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

        validation_artifacts = self.validation_step({"prompt_embeds": prompt_embeds, "image": image, "plucker_embedding": plucker_embedding}, pipe)
        prompt_filename = string_to_filename(prompt)[:25]
        artifacts = {
            "conditioned_image": {"type": "image", "value": image},
            "gt_video": {"type": "video", "value": video},
        }

        for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
            artifacts.update({f"generated_video_{i}": {"type": artifact_type, "value": artifact_value}})

        for key, value in list(artifacts.items()):
            artifact_type = value["type"]
            artifact_value = value["value"]
            if artifact_type not in ["image", "video"] or artifact_value is None:
                continue

            extension = "png" if artifact_type == "image" else "mp4"
            filename = f"validation-{key}-{step}-{accelerator.process_index}-{prompt_filename}.{extension}"
            validation_path = self.args.output_dir / "validation_res"
            validation_path.mkdir(parents=True, exist_ok=True)
            filename = str(validation_path / filename)

            if artifact_type == "image":
                logger.debug(f"Saving image to {filename}")
                artifact_value.save(filename)
                artifact_value = wandb.Image(filename)
            elif artifact_type == "video":
                logger.debug(f"Saving video to {filename}")
                export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                artifact_value = wandb.Video(filename, caption=f"{key}_{prompt}")

            all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self._move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self._move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.controlnetxs.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.controlnetxs], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.controlnetxs.train()

    @override
    def validation_step(
            self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt_embeds, image, plucker_embedding = eval_data["prompt_embeds"], eval_data["image"], eval_data["plucker_embedding"]

        # camera
        plucker_embedding = plucker_embedding.to(self.components.controlnetxs.vae_encoder.device, dtype=self.components.controlnetxs.vae_encoder.dtype) # [C=6, F, H, W]
        latent_plucker_embedding_dist = self.components.controlnetxs.vae_encoder.encode(plucker_embedding.unsqueeze(0)).latent_dist  # B,C=6,F,H,W --> B,128,F//4,H//4,W//4
        latent_plucker_embedding = latent_plucker_embedding_dist.sample()
        patch_size_t = self.components.transformer.config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent_plucker_embedding.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent_plucker_embedding[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent_plucker_embedding = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent_plucker_embedding], dim=2)
            assert latent_plucker_embedding.shape[2] % patch_size_t == 0
        latent_plucker_embedding = latent_plucker_embedding.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] to [B, F, C, H, W]

        latent_plucker_embedding = latent_plucker_embedding.repeat(2,1,1,1,1) # cfg
        num_frames = latent_plucker_embedding.shape[1]
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        controlnetxs_transformer_config = self.state.controlnetxs_transformer_config
        rotary_emb_for_controlnetxs = (
            self.prepare_rotary_positional_embeddings(
                height=latent_plucker_embedding.shape[-2] * vae_scale_factor_spatial,
                width=latent_plucker_embedding.shape[-1] * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=controlnetxs_transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        original_forward = pipe.transformer.forward
        if self.args.enable_gft_training:
            camera_condition_gft_beta = torch.ones((latent_plucker_embedding.shape[0], ), device=self.accelerator.device) * 0.4
        else:
            camera_condition_gft_beta = torch.ones((latent_plucker_embedding.shape[0],), device=self.accelerator.device)
        pipe.transformer.forward = partial(
            pipe.transformer.forward,
            controlnetxs=unwrap_model(self.accelerator, self.components.controlnetxs),
            latent_plucker_embedding=latent_plucker_embedding,
            image_rotary_emb_for_controlnetxs=rotary_emb_for_controlnetxs,
            camera_condition_gft_beta=camera_condition_gft_beta,
            camera_condition_start_timestep=self.args.camera_condition_start_timestep
        )
        #########

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt_embeds=prompt_embeds.unsqueeze(0),
            image=image,
            generator=self.state.generator,
            num_inference_steps=25
        ).frames[0]
        pipe.transformer.forward = original_forward
        return [("video", video_generate)]

    @override
    def _prepare_saving_loading_hooks(self):
        pass

    def prepare_rotary_positional_embeddings(
            self,
            height: int,
            width: int,
            num_frames: int,
            transformer_config: Dict,
            vae_scale_factor_spatial: int,
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


register("cogvideox1.5-i2v", "controlnetxs", CogVideoX1dot5I2VControlnetXsTrainer)
