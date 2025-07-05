from .vae_encoder import ControlnetXsVaeEncoderCogVideoX
from .vae_decoder import ControlnetXsVaeDecoderCogVideoX
from .transformer import CogVideoXTransformer3DControlnetXs
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero, RMSNorm
import math
from .transformer import CogVideoXControlnetXsLayerNormZero

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ZeroLayerNormDownProjector(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            time_embed_dim: int,
            dropout: float = 0.0,
            activation_fn: str = "gelu-approximate",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # dim_in > dim_out
        self.silu = nn.SiLU()
        self.scale_shift_linear = nn.Linear(time_embed_dim, 2 * dim_in, bias=True)
        self.gate_linear = nn.Linear(time_embed_dim, dim_out, bias=True)
        self.norm = nn.LayerNorm(dim_in, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 4 * dim_out),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * dim_out, dim_out),
        )

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, hidden_states2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        silu_temb = self.silu(temb)
        scale, shift = self.scale_shift_linear(silu_temb).chunk(2, dim=1)
        gate = self.gate_linear(silu_temb)

        return hidden_states2 + gate[:, None, :] * \
            self.mlp(self.norm(hidden_states) * (1 + scale[:, None, :]) + shift[:, None, :])


class ZeroLayerNormUpProjector(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            time_embed_dim: int,
            dropout: float = 0.0,
            activation_fn: str = "gelu-approximate",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # dim_in < dim_out
        self.silu = nn.SiLU()
        self.scale_shift_linear = nn.Linear(time_embed_dim, 2 * dim_in, bias=True)
        self.gate_linear = nn.Linear(time_embed_dim, dim_out, bias=True)
        self.norm = nn.LayerNorm(dim_in, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 4 * dim_in),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * dim_in, dim_out),
        )

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, hidden_states2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        silu_temb = self.silu(temb)
        scale, shift = self.scale_shift_linear(silu_temb).chunk(2, dim=1)
        gate = self.gate_linear(silu_temb)

        return hidden_states2 + gate[:, None, :] * \
            self.mlp(self.norm(hidden_states) * (1 + scale[:, None, :]) + shift[:, None, :])


class ControlnetXs(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            model_path: str,
            main_transformer_config=None,
    ):
        super().__init__()
        self.vae_encoder = ControlnetXsVaeEncoderCogVideoX.from_config(os.path.join(model_path, "controlnetxs_vae_encoder"))
        self.transformer = CogVideoXTransformer3DControlnetXs.from_config(os.path.join(model_path, "controlnetxs_transformer"))
        self.main_transformer_config = main_transformer_config

        inner_dim_for_transformer = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        inner_dim_for_main_transformer = self.main_transformer_config.num_attention_heads * self.main_transformer_config.attention_head_dim

        self.up_down_layer_start_idx = self.transformer.config.up_down_layer_start_idx
        self.up_down_layer_end_idx = self.transformer.config.up_down_layer_end_idx

        self.down_projectors = nn.ModuleList([
            ZeroLayerNormDownProjector(
                dim_in=inner_dim_for_main_transformer,
                dim_out=inner_dim_for_transformer,
                time_embed_dim=self.transformer.config.time_embed_dim,
                dropout=self.transformer.config.dropout,
                activation_fn=self.transformer.config.activation_fn,
                norm_elementwise_affine=self.transformer.config.norm_elementwise_affine,
                norm_eps=self.transformer.config.norm_eps,
            ) for _ in range(self.transformer.config.num_layers)
        ])
        self.up_projectors = nn.ModuleList([
            ZeroLayerNormUpProjector(
                dim_in=inner_dim_for_transformer,
                dim_out=inner_dim_for_main_transformer,
                time_embed_dim=self.transformer.config.time_embed_dim,
                dropout=self.transformer.config.dropout,
                activation_fn=self.transformer.config.activation_fn,
                norm_elementwise_affine=self.transformer.config.norm_elementwise_affine,
                norm_eps=self.transformer.config.norm_eps,
            ) for _ in range(self.transformer.config.num_layers)
        ])

        self.emb_projector = nn.Linear(self.main_transformer_config.time_embed_dim, self.transformer.config.time_embed_dim)

        self.gradient_checkpointing = False

        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for down_projector in self.down_projectors:
            nn.init.constant_(down_projector.scale_shift_linear.weight, 0)
            nn.init.constant_(down_projector.scale_shift_linear.bias, 0)
            nn.init.constant_(down_projector.gate_linear.weight, 0)
            nn.init.constant_(down_projector.gate_linear.bias, 0)

        for up_projector in self.up_projectors:
            nn.init.constant_(up_projector.scale_shift_linear.weight, 0)
            nn.init.constant_(up_projector.scale_shift_linear.bias, 0)
            nn.init.constant_(up_projector.gate_linear.weight, 0)
            nn.init.constant_(up_projector.gate_linear.bias, 0)

        for block in self.transformer.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm2.linear.weight, 0)
            nn.init.constant_(block.norm2.linear.bias, 0)

        nn.init.constant_(self.transformer.camera_condition_gft_beta_embedding.linear_2.weight, 0)
        nn.init.constant_(self.transformer.camera_condition_gft_beta_embedding.linear_2.bias, 0)

    def set_main_transformer(self, main_transformer):
        self.main_transformer = main_transformer

    def forward(
            self,
            hidden_states: torch.Tensor,  # B, F//4, C, H//8, W//8
            encoder_hidden_states: torch.Tensor,
            timestep: Union[int, float, torch.LongTensor],
            timestep_cond: Optional[torch.Tensor] = None,
            ofs: Optional[Union[int, float, torch.LongTensor]] = None,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            plucker_embedding=None,  # B, C, F, H, W
            image_rotary_emb_for_controlnetxs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
            main_transformer=None,
            camera_condition_gft_beta: Optional[Union[int, float, torch.LongTensor]] = None,  # Guidance-Free Training
            camera_condition_dropout=0.0
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(main_transformer, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = main_transformer.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = main_transformer.time_embedding(t_emb, timestep_cond)

        if main_transformer.ofs_embedding is not None:
            ofs_emb = main_transformer.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = main_transformer.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # Patch embed
        emb_for_controlnetxs = self.emb_projector(emb)

        if camera_condition_gft_beta is None:
            camera_condition_gft_beta = torch.ones(timesteps.shape[0]).to(device=hidden_states.device)
        camera_condition_gft_beta_emb = self.transformer.camera_condition_gft_beta_proj(camera_condition_gft_beta)
        camera_condition_gft_beta_emb = camera_condition_gft_beta_emb.to(dtype=hidden_states.dtype)
        camera_condition_gft_beta_emb = self.transformer.camera_condition_gft_beta_embedding(camera_condition_gft_beta_emb)
        emb_for_controlnetxs = emb_for_controlnetxs + camera_condition_gft_beta_emb

        plucker_embedding = plucker_embedding.to(self.vae_encoder.device, dtype=self.vae_encoder.dtype)
        latent_plucker_embedding_dist = self.vae_encoder.encode(plucker_embedding).latent_dist  # B,C=6,F,H,W --> B,128,F//4,H//4,W//4
        latent_plucker_embedding = latent_plucker_embedding_dist.sample()
        patch_size_t = main_transformer.config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent_plucker_embedding.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent_plucker_embedding[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent_plucker_embedding = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent_plucker_embedding], dim=2)
            assert latent_plucker_embedding.shape[2] % patch_size_t == 0
        latent_plucker_embedding = latent_plucker_embedding.permute(0, 2, 1, 3, 4)  # [B, C=128, F//4, H//8, W//8] to [B, F//4, C=128, H//8, W//8]
        num_frames = latent_plucker_embedding.shape[1]
        if camera_condition_dropout > 0.0:
            drop_ids = torch.rand(latent_plucker_embedding.shape[0]).to(latent_plucker_embedding.device) <= camera_condition_dropout
            latent_plucker_embedding = torch.where(
                drop_ids[:, None, None, None, None],
                0.0,
                1.0
            ).to(latent_plucker_embedding.dtype) * latent_plucker_embedding

        hidden_states_for_controlnetxs = self.transformer.patch_embed(latent_plucker_embedding)
        hidden_states_for_controlnetxs = self.transformer.embedding_dropout(hidden_states_for_controlnetxs)

        # 2. Patch embedding
        hidden_states = main_transformer.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = main_transformer.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(main_transformer.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if i >= self.up_down_layer_start_idx and i <= self.up_down_layer_end_idx:
                    hidden_states_for_controlnetxs = self.down_projectors[i](
                        hidden_states, emb_for_controlnetxs, hidden_states_for_controlnetxs
                    )
                    hidden_states_for_controlnetxs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.transformer.transformer_blocks[i]),
                        hidden_states_for_controlnetxs,
                        emb_for_controlnetxs,
                        image_rotary_emb_for_controlnetxs,
                        **ckpt_kwargs,
                    )
                    hidden_states = self.up_projectors[i](
                        hidden_states_for_controlnetxs, emb_for_controlnetxs, hidden_states
                    )
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                if i >= self.up_down_layer_start_idx and i <= self.up_down_layer_end_idx:
                    hidden_states_for_controlnetxs = self.down_projectors[i](
                        hidden_states, emb_for_controlnetxs, hidden_states_for_controlnetxs,
                    )
                    hidden_states_for_controlnetxs = self.transformer.transformer_blocks[i](
                        hidden_states=hidden_states_for_controlnetxs,
                        temb=emb_for_controlnetxs,
                        image_rotary_emb=image_rotary_emb_for_controlnetxs,
                    )
                    hidden_states = self.up_projectors[i](
                        hidden_states_for_controlnetxs, emb_for_controlnetxs, hidden_states
                    )
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not main_transformer.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = main_transformer.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = main_transformer.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = main_transformer.norm_out(hidden_states, temb=emb)
        hidden_states = main_transformer.proj_out(hidden_states)

        # 5. Unpatchify
        p = main_transformer.config.patch_size
        p_t = main_transformer.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(main_transformer, lora_scale)

        if not return_dict:
            return (output,)

        rets = {}
        rets['sample'] = output

        return rets
