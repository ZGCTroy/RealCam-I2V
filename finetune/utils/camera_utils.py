import torch
from einops import rearrange, repeat
from packaging import version as pver
from torch import Tensor


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 3, 3

    def custom_meshgrid(*args):
        # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
        if pver.parse(torch.__version__) < pver.parse('1.10'):
            return torch.meshgrid(*args)
        else:
            return torch.meshgrid(*args, indexing='ij')

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx = K[..., 0, 0].unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1)

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    plucker = rearrange(plucker, "b f h w c -> b c f h w")  # [B, 6, F, H, W]
    return plucker


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def get_relative_pose(RT_4x4: Tensor, cond_frame_index: Tensor, mode='left'):
    '''
    :param
        RT: (B,F,4,4)
        cond_frame_index: (B,)
    :return:
        relative_RT_4x4: (B,F,4,4)
    '''
    b, t, _, _ = RT_4x4.shape  # b,t,4,4
    first_frame_RT = RT_4x4[torch.arange(b, device=RT_4x4.device), cond_frame_index, ...].unsqueeze(1)  # (B, 1, 4, 4)

    if mode == 'left':
        relative_RT_4x4 = first_frame_RT.inverse() @ RT_4x4
    elif mode == 'right':
        relative_RT_4x4 = RT_4x4 @ first_frame_RT.inverse()

    return relative_RT_4x4


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def get_camera_condition(H, W, camera_intrinsics, camera_extrinsics, mode, cond_frame_index=0, align_factor=1.0):
    '''
    :param camera_intrinsics: (B, F, 3, 3)
    :param camera_extrinsics: (B, F, 4, 4)
    :param cond_frame_index:  (B,)
    :param trace_scale_factor: (B,)
    :return: plucker_embedding: (B, 6, F, H, W)
    '''
    B, F = camera_extrinsics.shape[:2]
    camera_intrinsics_3x3 = camera_intrinsics.float()  # B, F, 3, 3
    if mode == "c2w":
        c2w_RT_4x4 = camera_extrinsics.float()  # B, F, 4, 4
    elif mode =="w2c":
        c2w_RT_4x4 = camera_extrinsics.float().inverse()  # B, F, 4, 4
    else:
        raise ValueError(f"Unknown mode {mode}")
    B, F, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

    relative_c2w_RT_4x4 = get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left')  # B,F,4,4
    relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * align_factor

    plucker_embedding = ray_condition(camera_intrinsics_3x3, relative_c2w_RT_4x4, H, W, device, flip_flag=None)  # B, 6, F, H, W

    return plucker_embedding, relative_c2w_RT_4x4 # B 6 F H W
