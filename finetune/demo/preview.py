import sys
import os
import numpy as np
import open3d as o3d
import torch
from mmengine import Config
from pyvirtualdisplay import Display
from tqdm import tqdm

sys.path.append("Metric3D")


def display_wrapper(func):
    def inner(*args, **kwargs):
        with Display(visible=False, size=(1920, 1080)):
            return func(*args, **kwargs)

    return inner


def relative_pose(rt: np.ndarray, mode: str, ref_index: int = 0) -> np.ndarray:
    if mode == "left":
        rt = np.linalg.inv(rt[ref_index]) @ rt
    elif mode == "right":
        rt = rt @ np.linalg.inv(rt[ref_index])
    return rt


def project_point_cloud(
    frame: np.ndarray,
    depth: np.ndarray,
    intrinsics: list[float],
    remove_outliers: bool = True,
    voxel_size: float = None,
) -> o3d.geometry.PointCloud:
    from mono.utils.unproj_pcd import reconstruct_pcd

    points = reconstruct_pcd(depth, *intrinsics).reshape(-1, 3)
    colors = frame.reshape(-1, 3) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.double))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.double))
    if remove_outliers:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=3.0)
        pcd = pcd.select_by_index(ind)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=0.5)

    return pcd


def create_camera_frustum(
    frame: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    c2w: np.ndarray,
    frustum_scale: float = 0.5,
):
    W, H = intrinsic.width, intrinsic.height
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    z = frustum_scale
    x = (W - cx) * z / fx
    y = (H - cy) * z / fy

    points = [[0, 0, 0], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0.8, 0.2, 0.2])
    line_set.transform(c2w)

    vertices = [points[i] for i in [1, 2, 3, 4]]
    triangles = [[0, 1, 2], [0, 2, 3]]
    img_plane = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    img_plane.triangle_uvs = o3d.utility.Vector2dVector(
        np.array([[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]])
    )
    img_plane.transform(c2w)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.albedo_img = o3d.geometry.Image(frame)

    return line_set, img_plane, material


class Previewer:
    def __init__(self, model_path: str = "pretrained/metric_depth_vit_large_800k.pth"):
        self.model_path = model_path
        self.depth_predictor = None

    def init_depth_predictor(self):
        from mono.model.monodepth_model import get_configured_monodepth_model
        from mono.utils.running import load_ckpt

        self.config = Config.fromfile(
            "Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py"
        )
        model = get_configured_monodepth_model(self.config)
        model = torch.nn.DataParallel(model).cuda().eval().requires_grad_(False)
        model, _, _, _ = load_ckpt(self.model_path, model, strict_match=False)
        self.depth_predictor = model

    def estimate_depths(
        self, frames: np.ndarray, intrinsics: list[float]
    ) -> np.ndarray:
        """
        :param frames: `np.ndarray` of shape (B, H, W, C) and range (0, 255)
        :param intrinsics: list of [fx, fy, cx, cy]
        :return depths: `np.ndarray` of shape (B, H, W) and range (0, 300)
        """

        from mono.utils.do_test import transform_test_data_scalecano

        if self.depth_predictor is None:
            self.init_depth_predictor()

        B, H, W, C = frames.shape
        rgb_inputs, pads = [], []
        for frame in frames:
            rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(
                frame, intrinsics, self.config.data_basic
            )
            rgb_inputs.append(rgb_input)
            pads.append(pad)

        with torch.inference_mode(), torch.autocast("cuda"):  # b c h w
            depths, _, _ = self.depth_predictor.module.inference(
                {"input": torch.stack(rgb_inputs).cuda(), "pad_info": pads}
            )

        _, _, h, w = depths.shape
        depths = depths[..., pad[0] : h - pad[1], pad[2] : w - pad[3]]
        depths = depths * self.config.data_basic.depth_range[-1] / label_scale_factor
        depths = torch.nn.functional.interpolate(depths, (H, W), mode="bilinear")

        return depths.clamp(0, 300).squeeze(1).cpu().numpy()

    @display_wrapper
    def render_previews(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        intrinsics: list[float],
        w2cs: np.ndarray,
    ):
        """
        :param frame: `np.ndarray` of shape (H, W, C) and range (0, 255)
        :param depth: `np.ndarray` of shape (H, W) and range (0, 300)
        :param intrinsics: list of [fx, fy, cx, cy]
        :param w2cs: `np.ndarray` of shape (4, 4)
        :return: previews: `np.ndarray of shape (B, H, W, C) and range (0, 255)`
        """

        H, W, _ = frame.shape
        K = o3d.camera.PinholeCameraIntrinsic(W, H, *intrinsics)
        pcd = project_point_cloud(frame, depth, intrinsics)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2

        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))
        renderer.scene.view.set_post_processing(False)
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("point cloud", pcd, mat)

        previews = []
        for w2c in tqdm(relative_pose(w2cs, mode="left")):
            renderer.setup_camera(K, w2c)
            previews.append(renderer.render_to_image())

        return np.stack(previews)

    @display_wrapper
    def render_4d_scene(
        self,
        frames: np.ndarray,
        depths: np.ndarray,
        intrinsics: list[float],
        w2cs: np.ndarray,
    ):
        """
        :param frames: `np.ndarray` of shape (B, H, W, C) and range (0, 255)
        :param depths: `np.ndarray` of shape (B, H, W) and range (0, 300)
        :param intrinsics: list of [fx, fy, cx, cy]
        :param w2cs: `np.ndarray` of shape (4, 4)
        :return: renderings: `np.ndarray of shape (B, H, W, C) and range (0, 255)`
        """

        F, H, W, _ = frames.shape
        K = o3d.camera.PinholeCameraIntrinsic(W, H, *intrinsics)

        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))
        renderer.scene.view.set_post_processing(False)

        c2w_0 = np.linalg.inv(w2cs[0])
        eye_pos_world = (c2w_0 @ np.array([0.3, -0.5, -0.5, 1]))[:3]
        center_pos_world = (c2w_0 @ np.array([0, 0, 2, 1]))[:3]
        up_vector_world = np.array([0, -1, 0])
        renderer.scene.camera.look_at(center_pos_world, eye_pos_world, up_vector_world)

        point_material = o3d.visualization.rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = 2

        line_material = o3d.visualization.rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 3

        renderings = []
        for frame, depth, w2c in tqdm(zip(frames, depths, w2cs), total=F):
            c2w = np.linalg.inv(w2c)
            pcd = project_point_cloud(frame, depth, intrinsics)
            pcd.transform(c2w)

            wire_frame, frustum, frustum_material = create_camera_frustum(frame, K, c2w)

            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("point cloud", pcd, point_material)
            renderer.scene.add_geometry("wire frame", wire_frame, line_material)
            renderer.scene.add_geometry("frustum", frustum, frustum_material)

            renderings.append(renderer.render_to_image())

        return np.stack(renderings)

if __name__ == "__main__":
    with Display(visible=False, size=(512, 320)):
        o3d.visualization.rendering.OffscreenRenderer(512, 320)
