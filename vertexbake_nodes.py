import json
import math
import os
import sys
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, grey_dilation
from .utils import _run_blender_script, get_blender_clean_mesh_func_script

try:
    import nvdiffrast.torch as dr
except Exception:  # pragma: no cover - optional dependency
    dr = None

def create_view_matrix(position, target, up):
    f = target - position
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    
    view_matrix = np.identity(4)
    view_matrix[0, :3] = s
    view_matrix[1, :3] = u
    view_matrix[2, :3] = -f
    t = np.array([np.dot(-s, position), np.dot(-u, position), np.dot(f, position)])
    view_matrix[:3, 3] = t
    return view_matrix

def create_orthographic_projection(height, aspect, near=-1000.0, far=1000.0):
    top =  0.5 * height
    bottom = -top
    right  = 0.5 * height * aspect
    left   = -right

    M = np.zeros((4, 4), dtype=float)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (far - near)
    M[3, 3] = 1.0

    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(far + near) / (far - near)
    return M

def create_perspective_projection(fovy_rad, aspect, near=0.1, far=1000.0):
    f = 1.0 / np.tan(fovy_rad / 2.0)
    M = np.zeros((4, 4), dtype=float)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[3, 2] = -1.0
    M[2, 3] = (2 * far * near) / (near - far)
    return M

def get_camera_position(center, distance, azimuth_deg, elevation_deg):
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    x = center[0] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = center[1] + distance * np.sin(elevation_rad)
    z = center[2] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    
    return np.array([x, y, z])

DEFAULT_CAMERA_DISTANCE = 1.45
DEFAULT_ORTHO_SCALE = 1.2

def resolve_camera_setup(num_views, camera_config=None):
    """
    Resolve a camera setup that matches the multiview image count.
    When no camera_config is supplied we provide sensible defaults so the node works out-of-the-box.
    """
    if num_views <= 0:
        raise ValueError("At least one multiview image is required.")

    if camera_config:
        camera_azims = camera_config.get("selected_camera_azims")
        camera_elevs = camera_config.get("selected_camera_elevs")
        if camera_azims is None or camera_elevs is None:
            raise ValueError("camera_config must include 'selected_camera_azims' and 'selected_camera_elevs'.")
        if len(camera_azims) != num_views or len(camera_elevs) != num_views:
            raise ValueError(
                f"Camera configuration expects {len(camera_azims)}/{len(camera_elevs)} views but "
                f"{num_views} images were provided."
            )
        cam_distance = camera_config.get("camera_distance", 1.45)
        ortho_scale_mult = camera_config.get("ortho_scale", 1.2)
        return camera_azims, camera_elevs, cam_distance, ortho_scale_mult

    default_layouts = {
        4: (
            [0.0, 90.0, 180.0, 270.0],
            [0.0, 0.0, 0.0, 0.0],
        ),
        6: (
            [0.0, 90.0, 180.0, 270.0, 0.0, 180.0],
            [10.0, -10.0, 10.0, -10.0, 90.0, -90.0],
        ),
    }

    if num_views in default_layouts:
        camera_azims, camera_elevs = default_layouts[num_views]
    else:
        step = 360.0 / num_views
        camera_azims = [step * i for i in range(num_views)]
        camera_elevs = [0.0] * num_views

    return camera_azims, camera_elevs, DEFAULT_CAMERA_DISTANCE, DEFAULT_ORTHO_SCALE

def _infer_multiview_count(multiview_images):
    if isinstance(multiview_images, torch.Tensor):
        return int(multiview_images.shape[0])
    try:
        return len(multiview_images)
    except TypeError:
        return 0

def _calculate_seqtex_camera_distance(mesh, camera_lens, camera_sensor_width):
    extents = np.array(getattr(mesh.bounding_box, "extents", [1.0, 1.0, 1.0]), dtype=np.float64)
    if extents.size != 3:
        extents = np.ones(3, dtype=np.float64)

    max_extent = float(np.max(extents))
    if max_extent <= 1e-6:
        normalized_diag = 2.0
    else:
        normalized_diag = (2.0 / max_extent) * float(np.linalg.norm(extents))

    return float((camera_lens / camera_sensor_width) * normalized_diag)

def _compute_seqtex_camera_parameters(
    mesh,
    num_views,
    camera_elevation,
    camera_lens,
    camera_sensor_width,
    azimuth_offset=0.0,
    include_poles=False,
):
    if num_views <= 0:
        raise ValueError("SeqTex camera override requires at least one multiview image.")

    distance = _calculate_seqtex_camera_distance(mesh, camera_lens, camera_sensor_width)
    use_poles = include_poles and num_views >= 6

    ring_views = num_views - 2 if use_poles else num_views
    ring_views = max(ring_views, 0)

    if ring_views > 0:
        base_azims = np.linspace(0.0, 360.0, ring_views, endpoint=False) - 90.0
        ring_azims = (base_azims + azimuth_offset).tolist()
        ring_elevs = [float(camera_elevation)] * ring_views
    else:
        ring_azims = []
        ring_elevs = []

    azims = list(ring_azims)
    elevs = list(ring_elevs)

    if use_poles:
        azims.append(float(azimuth_offset))
        elevs.append(90.0)
        azims.append(float(azimuth_offset))
        elevs.append(-90.0)
    fov_rad = 2.0 * np.arctan(camera_sensor_width / (2.0 * camera_lens))
    fov_deg = float(np.degrees(fov_rad))

    camera_config = {
        "selected_camera_azims": azims,
        "selected_camera_elevs": elevs,
        "camera_distance": float(distance),
        "ortho_scale": 1.2,
    }
    return camera_config, fov_deg

SEQ_TEX_VIEW_PRESETS = ["2", "4", "6", "12"]

def _seqtex_view_preset_angles(view_preset, default_elevation, azimuth_offset=0.0):
    preset = str(view_preset)
    base_elev = float(default_elevation)
    offset = float(azimuth_offset)

    def apply_offset(values):
        return [float(v + offset) for v in values]

    if preset == "2":
        azims = apply_offset([-90.0, 90.0])
        elevs = [base_elev, base_elev]
    elif preset == "4":
        azims = apply_offset([-90.0, 0.0, 90.0, 180.0])
        elevs = [base_elev] * 4
    elif preset == "6":
        azims = apply_offset([-90.0, 0.0, 90.0, 180.0, 0.0, 0.0])
        elevs = [base_elev, base_elev, base_elev, base_elev, 90.0, -90.0]
    elif preset == "12":
        first_six = [
            (-90.0, base_elev),
            (0.0, base_elev),
            (90.0, base_elev),
            (180.0, base_elev),
            (0.0, 90.0),
            (0.0, -90.0),
        ]
        legacy_views = [
            (-90.0, base_elev),
            (-45.0, base_elev),
            (0.0, base_elev),
            (45.0, base_elev),
            (90.0, base_elev),
            (135.0, base_elev),
            (180.0, base_elev),
            (-135.0, base_elev),
            (-90.0, 45.0),
            (90.0, 45.0),
            (-90.0, -45.0),
            (90.0, -45.0),
        ]
        seen = {(round(a, 4), round(e, 4)) for a, e in first_six}
        leftovers = []
        for az, el in legacy_views:
            key = (round(az, 4), round(el, 4))
            if key in seen:
                continue
            seen.add(key)
            leftovers.append((az, el))
            if len(leftovers) >= 6:
                break
        while len(leftovers) < 6:
            # Fallback to mirrored diagonals if the legacy set was smaller than expected.
            leftovers.append((0.0, 75.0 if len(leftovers) % 2 == 0 else -75.0))
        ordered = first_six + leftovers[:6]
        azims = apply_offset([az for az, _ in ordered])
        elevs = [el for _, el in ordered]
    else:
        raise ValueError(f"Unknown SeqTex view preset '{view_preset}'.")

    return azims, elevs

def _build_seqtex_camera_config(mesh, azims, elevs, camera_lens, camera_sensor_width, ortho_scale=1.2):
    if len(azims) != len(elevs) or len(azims) == 0:
        raise ValueError("SeqTex camera config requires matching, non-empty azimuth and elevation lists.")
    distance = _calculate_seqtex_camera_distance(mesh, camera_lens, camera_sensor_width)
    return {
        "selected_camera_azims": [float(a) for a in azims],
        "selected_camera_elevs": [float(e) for e in elevs],
        "camera_distance": float(distance),
        "ortho_scale": float(ortho_scale),
    }

class SeqTexCam:
    VIEW_PRESETS = SEQ_TEX_VIEW_PRESETS.copy()

    @classmethod
    def INPUT_TYPES(cls):
        tooltip = "2=Front/Back, 4=Front/Side loop, 6=+Top/Bottom, 12=adds diagonal & elevated views"
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "view_preset": (cls.VIEW_PRESETS, {"default": "4", "tooltip": tooltip}),
                "camera_elevation": ("INT", {"default": 0, "min": -90, "max": 90}),
                "camera_lens": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 200.0, "step": 0.1}),
                "camera_sensor_width": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "azimuth_offset": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("HY3DCAMERA", "INT")
    RETURN_NAMES = ("camera_config", "num_views")
    FUNCTION = "generate"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def generate(
        self,
        mesh,
        view_preset,
        camera_elevation,
        camera_lens,
        camera_sensor_width,
        azimuth_offset,
        ortho_scale,
    ):
        azims, elevs = _seqtex_view_preset_angles(view_preset, camera_elevation, azimuth_offset=azimuth_offset)
        camera_config = _build_seqtex_camera_config(
            mesh,
            azims,
            elevs,
            camera_lens=float(camera_lens),
            camera_sensor_width=float(camera_sensor_width),
            ortho_scale=float(ortho_scale),
        )
        return (camera_config, len(azims))

def _multiview_tensor_to_pil(multiview_images):
    pil_images = []
    if isinstance(multiview_images, torch.Tensor):
        iterator = multiview_images
    else:
        iterator = list(multiview_images)

    for img in iterator:
        if isinstance(img, torch.Tensor):
            img_array = img.detach().cpu().numpy()
        elif isinstance(img, np.ndarray):
            img_array = img
        else:
            raise ValueError("Unsupported multiview image type. Expected torch.Tensor or numpy.ndarray.")

        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        else:
            img_array = img_array.copy()

        pil_images.append(Image.fromarray(img_array))
    return pil_images

def _save_pil_sequence(pil_images, output_dir, prefix, mode):
    if not pil_images:
        return []
    paths = []
    for idx, img in enumerate(pil_images):
        filename = f"{prefix}_{idx:03d}.png"
        path = os.path.join(output_dir, filename)
        converted = img.convert(mode)
        converted.save(path)
        paths.append(path)
    return paths

BLENDER_SEQTEX_PROJECTION_SCRIPT = r"""
import bpy
import sys
import json
import numpy as np
from mathutils import kdtree

PARAMS_PATH = "__COMFY_PARAMS_PATH__"
DEFAULT_CAMERA_DISTANCE = 1.45
DEFAULT_ORTHO_SCALE = 1.2

def log(message):
    print(f"[SeqTexProjection] {message}")

def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)

def create_view_matrix(position, target, up):
    f = target - position
    norm = np.linalg.norm(f)
    if norm < 1e-8:
        return np.identity(4, dtype=np.float64)
    f = f / norm
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-8:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        s = s / s_norm
    u = np.cross(s, f)
    view = np.identity(4, dtype=np.float64)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(s, position)
    view[1, 3] = -np.dot(u, position)
    view[2, 3] = np.dot(f, position)
    return view

def create_orthographic_projection(height, aspect, near=-1000.0, far=1000.0):
    top = 0.5 * height
    bottom = -top
    right = 0.5 * height * aspect
    left = -right
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 2.0 / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[2, 2] = -2.0 / (far - near)
    proj[3, 3] = 1.0
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    return proj

def create_perspective_projection(fovy_rad, aspect, near=0.1, far=1000.0):
    f = 1.0 / np.tan(max(fovy_rad, 1e-6) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / max(aspect, 1e-6)
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj

def get_camera_position(center, distance, azimuth_deg, elevation_deg):
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    x = center[0] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = center[1] + distance * np.sin(elevation_rad)
    z = center[2] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    return np.array([x, y, z], dtype=np.float64)

def load_image_rgba(path):
    img = bpy.data.images.load(path)
    try:
        width, height = img.size
        channels = img.channels
        pixels = np.array(img.pixels[:], dtype=np.float32)
        pixels = pixels.reshape(height, width, channels)
        if channels < 4:
            rgba = np.ones((height, width, 4), dtype=np.float32)
            rgba[:, :, :channels] = pixels[:, :, :channels]
        else:
            rgba = pixels[:, :, :4]
        rgba = np.flip(rgba, axis=0)
        rgba = np.clip(rgba, 0.0, 1.0) * 255.0
        return rgba.astype(np.uint8)
    finally:
        bpy.data.images.remove(img)

def load_mask(path):
    data = load_image_rgba(path)
    gray = data[:, :, 0].astype(np.float32) / 255.0
    return gray

def build_camera_set(image_count, params):
    config = params.get("camera_config") or {}
    azims = list(config.get("selected_camera_azims") or [])
    elevs = list(config.get("selected_camera_elevs") or [])
    if azims and len(azims) != len(elevs):
        raise ValueError("Camera configuration requires matching azimuth/elevation lengths.")
    if not azims:
        step = 360.0 / max(image_count, 1)
        azims = [step * i for i in range(image_count)]
        elevs = [0.0] * image_count
    if len(azims) < image_count:
        raise ValueError("Camera configuration provides fewer views than images.")
    if len(azims) > image_count:
        azims = azims[:image_count]
        elevs = elevs[:image_count]
    distance = float(config.get("camera_distance", DEFAULT_CAMERA_DISTANCE))
    ortho_scale = float(config.get("ortho_scale", DEFAULT_ORTHO_SCALE))
    return [
        {
            "azim": float(az),
            "elev": float(el),
            "distance": distance,
            "ortho_scale_mult": ortho_scale,
        }
        for az, el in zip(azims, elevs)
    ]

def build_projection_context(vertices, centroid, extents, images, masks, params, camera_set):
    height, width = images[0].shape[0], images[0].shape[1]
    scene_scale = float(np.linalg.norm(extents)) if np.any(extents) else 1.0
    occlusion_epsilon = max(scene_scale * 1e-4, 1e-6)
    cam_target = centroid
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    aspect_ratio = width / max(height, 1)
    projection_mode = params["projection_mode"].lower()
    verts_h = np.hstack((vertices, np.ones((len(vertices), 1), dtype=np.float64)))
    cos_threshold = np.cos(np.deg2rad(params["angle_cutoff"]))
    cameras = []
    default_distance = camera_set[0].get("distance", DEFAULT_CAMERA_DISTANCE)
    default_ortho = camera_set[0].get("ortho_scale_mult", DEFAULT_ORTHO_SCALE)

    for idx, view in enumerate(camera_set):
        img_array = images[idx]
        mask_array = masks[idx] if masks else None
        azim = view.get("azim", 0.0)
        elev = view.get("elev", 0.0)
        cam_distance = view.get("distance", default_distance)
        ortho_scale_override = view.get("ortho_scale_mult", default_ortho)

        cam_pos = get_camera_position(centroid, cam_distance, azim, elev)
        view_mat = create_view_matrix(cam_pos, cam_target, cam_up)

        if projection_mode == "orthographic":
            width_override = params["orthographic_width"]
            height_override = params["orthographic_height"]
            if width_override > 0 and height_override > 0:
                aspect = width_override / max(height_override, 1e-6)
                proj_mat = create_orthographic_projection(height_override, aspect)
            else:
                max_extent = float(np.max(extents)) if np.any(extents) else 1.0
                ortho_height = max_extent * ortho_scale_override
                proj_mat = create_orthographic_projection(ortho_height, aspect_ratio)
        else:
            persp_width = params["perspective_width"]
            persp_height = params["perspective_height"]
            if persp_width > 0 and persp_height > 0:
                fov_y_rad = 2.0 * np.arctan((persp_height / 2.0) / max(cam_distance, 1e-6))
                proj_mat = create_perspective_projection(fov_y_rad, persp_width / max(persp_height, 1e-6))
            else:
                fov_y_rad = np.radians(params["perspective_fov"])
                proj_mat = create_perspective_projection(fov_y_rad, aspect_ratio)

        pvm_matrix = proj_mat @ view_mat
        depth_buffer = None
        if params["use_depth_occlusion"]:
            clip = verts_h @ pvm_matrix.T
            w_coords = clip[:, 3]
            w_coords[np.abs(w_coords) < 1e-6] = 1e-6
            ndc = clip[:, :3] / w_coords[:, None]
            mask = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
            if np.any(mask):
                frustum_idx = np.where(mask)[0]
                px = (ndc[frustum_idx, 0] * 0.5 + 0.5) * width
                py = (1.0 - (ndc[frustum_idx, 1] * 0.5 + 0.5)) * height
                px_int = np.clip(px, 0, width - 1).astype(np.int32, copy=False)
                py_int = np.clip(py, 0, height - 1).astype(np.int32, copy=False)
                view_vectors = vertices[frustum_idx] - cam_pos
                view_distances = np.linalg.norm(view_vectors, axis=1).astype(np.float32)
                depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
                np.minimum.at(depth_buffer, (py_int, px_int), view_distances)

        cameras.append(
            {
                "pvm_matrix": pvm_matrix,
                "cam_pos": cam_pos,
                "image": img_array,
                "mask": mask_array,
                "depth_buffer": depth_buffer,
            }
        )

    return {
        "cameras": cameras,
        "image_width": width,
        "image_height": height,
        "blend_sharpness": params["blend_sharpness"],
        "cos_threshold": cos_threshold,
        "mask_min_value": params["mask_min_value"],
        "use_depth_occlusion": params["use_depth_occlusion"],
        "occlusion_epsilon": occlusion_epsilon,
    }

def project_colors_for_points(points, normals, context):
    count = len(points)
    if count == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0, 1), dtype=np.float64)
    verts_h = np.hstack((points, np.ones((count, 1), dtype=np.float64)))
    colors = np.zeros((count, 4), dtype=np.float64)
    weights = np.zeros((count, 1), dtype=np.float64)
    h = context["image_height"]
    w = context["image_width"]
    cos_thresh = context["cos_threshold"]
    blend_power = context["blend_sharpness"]
    mask_min = context["mask_min_value"]
    occlusion_epsilon = context["occlusion_epsilon"]
    use_depth = context["use_depth_occlusion"]

    for cam in context["cameras"]:
        clip = verts_h @ cam["pvm_matrix"].T
        w_coords = clip[:, 3]
        w_coords[np.abs(w_coords) < 1e-6] = 1e-6
        ndc = clip[:, :3] / w_coords[:, None]
        mask = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        px = (ndc[idx, 0] * 0.5 + 0.5) * w
        py = (1.0 - (ndc[idx, 1] * 0.5 + 0.5)) * h
        px_int = np.clip(px.astype(np.int32), 0, w - 1)
        py_int = np.clip(py.astype(np.int32), 0, h - 1)
        view_vectors = points[idx] - cam["cam_pos"]
        view_distances = np.linalg.norm(view_vectors, axis=1)
        view_distances[view_distances == 0] = 1e-6
        view_dirs = view_vectors / view_distances[:, None]
        dot = np.sum(normals[idx] * view_dirs, axis=1)
        front = dot < 0
        if not np.any(front):
            continue
        idx = idx[front]
        px_int = px_int[front]
        py_int = py_int[front]
        view_distances = view_distances[front]
        cos_values = -dot[front]
        keep = cos_values >= cos_thresh
        if not np.any(keep):
            continue
        idx = idx[keep]
        px_int = px_int[keep]
        py_int = py_int[keep]
        view_distances = view_distances[keep]
        cos_values = cos_values[keep]
        dyn_weights = np.power(cos_values, blend_power)

        if use_depth and cam["depth_buffer"] is not None:
            depth_limits = cam["depth_buffer"][py_int, px_int] + occlusion_epsilon
            depth_mask = view_distances <= depth_limits
            if not np.any(depth_mask):
                continue
            idx = idx[depth_mask]
            px_int = px_int[depth_mask]
            py_int = py_int[depth_mask]
            dyn_weights = dyn_weights[depth_mask]

        if idx.size == 0:
            continue

        if cam["mask"] is not None:
            mask_weights = cam["mask"][py_int, px_int]
            mask_keep = mask_weights > mask_min
            if not np.any(mask_keep):
                continue
            idx = idx[mask_keep]
            px_int = px_int[mask_keep]
            py_int = py_int[mask_keep]
            dyn_weights = dyn_weights[mask_keep] * mask_weights[mask_keep]

        if idx.size == 0:
            continue

        sampled = cam["image"][py_int, px_int].astype(np.float64)
        weight_values = dyn_weights[:, None]
        colors[idx] += sampled * weight_values
        weights[idx] += weight_values

    return colors, weights

def fill_unpainted_colors(vertices, colors, valid_mask):
    missing_mask = ~valid_mask
    if not np.any(missing_mask):
        return colors
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return colors
    kd = kdtree.KDTree(valid_indices.size)
    for i, vid in enumerate(valid_indices):
        kd.insert(vertices[vid], int(vid))
    kd.balance()
    missing_indices = np.where(missing_mask)[0]
    for vid in missing_indices:
        _, nearest, _ = kd.find(vertices[vid])
        colors[vid] = colors[nearest]
    return colors

def ensure_vertex_color_attribute(obj, name="SeqTexVertexColor"):
    mesh = obj.data
    color_attr = mesh.color_attributes.get(name)
    if color_attr and (color_attr.domain != 'POINT' or color_attr.data_type != 'BYTE_COLOR'):
        mesh.color_attributes.remove(color_attr)
        color_attr = None
    if color_attr is None:
        color_attr = mesh.color_attributes.new(name=name, type='BYTE_COLOR', domain='POINT')
    return color_attr

def assign_colors(obj, colors):
    attr = ensure_vertex_color_attribute(obj)
    data = attr.data
    if len(data) != len(colors):
        raise ValueError("Vertex color attribute size mismatch.")
    colors = colors.astype(np.float32) / 255.0
    for idx, value in enumerate(colors):
        data[idx].color = value

def main():
    params = load_params()
    log("Starting Blender projection.")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.import_scene.gltf(filepath=params["mesh_in"])
    bpy.context.view_layer.update()
    obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mesh = obj.data
    mesh.calc_normals()
    vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
    normals = np.array([v.normal[:] for v in mesh.vertices], dtype=np.float64)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    centroid = (bbox_min + bbox_max) / 2.0
    extents = bbox_max - bbox_min
    images = [load_image_rgba(path) for path in params["image_paths"]]
    masks = None
    mask_paths = params.get("mask_paths") or []
    if mask_paths:
        if len(mask_paths) != len(images):
            raise ValueError("Mask count must match image count.")
        masks = [load_mask(path) for path in mask_paths]
    camera_set = build_camera_set(len(images), params)
    proj_context = build_projection_context(
        vertices,
        centroid,
        extents,
        images,
        masks,
        params,
        camera_set,
    )
    colors_acc, weight_acc = project_colors_for_points(vertices, normals, proj_context)
    valid_mask = weight_acc[:, 0] > 0
    safe_weights = weight_acc.copy()
    safe_weights[~valid_mask, 0] = 1.0
    final_colors = np.clip(colors_acc / safe_weights, 0, 255).astype(np.uint8)
    final_colors[valid_mask, 3] = 255
    if params.get("fill_unpainted", True) and np.any(~valid_mask):
        final_colors = fill_unpainted_colors(vertices, final_colors, valid_mask)
    else:
        final_colors[~valid_mask, 3] = 255
    assign_colors(obj, final_colors)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=params["mesh_out"], export_format='GLB', use_selection=True)
    log(f"Projection finished. Saved mesh to {params['mesh_out']}")
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"[SeqTexProjection] Blender script failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
"""

def _build_seqtex_projection_script(params_path):
    normalized = params_path.replace("\\", "\\\\")
    return BLENDER_SEQTEX_PROJECTION_SCRIPT.replace("__COMFY_PARAMS_PATH__", normalized)

def _project_multiview_vertex_colors_with_blender(
    mesh,
    multiview_images,
    projection_mode,
    blend_sharpness,
    angle_cutoff,
    perspective_fov,
    orthographic_width,
    orthographic_height,
    perspective_width,
    perspective_height,
    camera_config,
    multiview_masks=None,
    use_depth_occlusion=True,
    mask_min_value=1e-3,
    fill_unpainted=True,
):
    if camera_config is None:
        raise ValueError("Blender-based SeqTex projection requires a camera_config.")

    images_pil = _multiview_tensor_to_pil(multiview_images)
    if not images_pil:
        raise ValueError("No multiview images provided for Blender projection.")

    masks_pil = None
    if multiview_masks is not None:
        masks_pil = _multiview_tensor_to_pil(multiview_masks)
        if masks_pil:
            masks_pil = masks_pil[: len(images_pil)]

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_in = os.path.join(temp_dir, "seqtex_input.glb")
        mesh_out = os.path.join(temp_dir, "seqtex_output.glb")
        mesh.export(mesh_in)

        image_paths = _save_pil_sequence(images_pil, temp_dir, "seqtex_view", "RGBA")
        mask_paths = _save_pil_sequence(masks_pil, temp_dir, "seqtex_mask", "L") if masks_pil else []

        params = {
            "mesh_in": mesh_in,
            "mesh_out": mesh_out,
            "image_paths": image_paths,
            "mask_paths": mask_paths,
            "projection_mode": projection_mode.lower(),
            "blend_sharpness": float(blend_sharpness),
            "angle_cutoff": float(angle_cutoff),
            "perspective_fov": float(perspective_fov),
            "orthographic_width": float(orthographic_width),
            "orthographic_height": float(orthographic_height),
            "perspective_width": float(perspective_width),
            "perspective_height": float(perspective_height),
            "use_depth_occlusion": bool(use_depth_occlusion),
            "mask_min_value": float(mask_min_value),
            "fill_unpainted": bool(fill_unpainted),
            "camera_config": {
                "selected_camera_azims": [float(v) for v in camera_config.get("selected_camera_azims", [])],
                "selected_camera_elevs": [float(v) for v in camera_config.get("selected_camera_elevs", [])],
                "camera_distance": float(camera_config.get("camera_distance", DEFAULT_CAMERA_DISTANCE)),
                "ortho_scale": float(camera_config.get("ortho_scale", DEFAULT_ORTHO_SCALE)),
            },
        }

        params_path = os.path.join(temp_dir, "seqtex_params.json")
        with open(params_path, "w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2)

        script_path = os.path.join(temp_dir, "seqtex_project.py")
        script_content = _build_seqtex_projection_script(params_path)
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(script_content)

        _run_blender_script(script_path)

        if not os.path.exists(mesh_out):
            raise RuntimeError("Blender SeqTex projection failed to produce an output mesh.")

        return trimesh.load(mesh_out, force="mesh")

CUSTOM_CAMERA_LAYOUTS = {
    "2": [
        {"azim": 0.0, "elev": 0.0, "category": "front_back"},
        {"azim": 180.0, "elev": 0.0, "category": "front_back"},
    ],
    "4": [
        {"azim": 0.0, "elev": 0.0, "category": "front_back"},
        {"azim": 90.0, "elev": 0.0, "category": "side"},
        {"azim": 180.0, "elev": 0.0, "category": "front_back"},
        {"azim": 270.0, "elev": 0.0, "category": "side"},
    ],
    "6": [
        {"azim": 0.0, "elev": 0.0, "category": "front_back"},
        {"azim": 90.0, "elev": 0.0, "category": "side"},
        {"azim": 180.0, "elev": 0.0, "category": "front_back"},
        {"azim": 270.0, "elev": 0.0, "category": "side"},
        {"azim": 0.0, "elev": 90.0, "category": "top_bottom"},
        {"azim": 0.0, "elev": -90.0, "category": "top_bottom"},
    ],
    "12": [
        {"azim": 270.0, "elev": 0.0, "category": "side"},
        {"azim": 0.0, "elev": 0.0, "category": "front_back"},
        {"azim": 90.0, "elev": 0.0, "category": "side"},
        {"azim": 180.0, "elev": 0.0, "category": "front_back"},
        {"azim": 0.0, "elev": 90.0, "category": "top_bottom"},
        {"azim": 0.0, "elev": -90.0, "category": "top_bottom"},
        {"azim": 315.0, "elev": 0.0, "category": "diagonal"},
        {"azim": 45.0, "elev": 0.0, "category": "diagonal"},
        {"azim": 135.0, "elev": 0.0, "category": "diagonal"},
        {"azim": 225.0, "elev": 0.0, "category": "diagonal"},
        {"azim": 270.0, "elev": 45.0, "category": "top_bottom"},
        {"azim": 90.0, "elev": 45.0, "category": "top_bottom"},
    ],
}

def _build_custom_camera_overrides(
    layout_name,
    available_image_count,
    distance_scale,
    front_back_scale,
    side_scale,
    diagonal_scale,
    top_bottom_scale,
    base_distance=DEFAULT_CAMERA_DISTANCE,
    base_ortho_scale=DEFAULT_ORTHO_SCALE,
):
    layout = CUSTOM_CAMERA_LAYOUTS.get(layout_name)
    if layout is None:
        raise ValueError(f"Unsupported camera layout '{layout_name}'.")
    layout_count = len(layout)
    if available_image_count < layout_count:
        raise ValueError(
            f"Camera layout '{layout_name}' requires at least {layout_count} images but {available_image_count} were provided."
        )
    scale_map = {
        "front_back": front_back_scale,
        "side": side_scale,
        "diagonal": diagonal_scale,
        "top_bottom": top_bottom_scale,
    }
    distance = base_distance * distance_scale
    overrides = []
    for view in layout:
        category = view["category"]
        overrides.append(
            {
                "azim": view["azim"],
                "elev": view["elev"],
                "distance": distance,
                "ortho_scale_mult": base_ortho_scale * scale_map.get(category, 1.0),
            }
        )
    return overrides

def _sample_vertices_for_scaling(mesh, max_samples=8192, seed=42):
    vertices = mesh.vertices
    if len(vertices) <= max_samples:
        return vertices
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vertices), size=max_samples, replace=False)
    return vertices[indices]

def _evaluate_mask_leakage(verts_homogeneous, pvm_matrix, mask_array, mask_threshold):
    clip_coords = verts_homogeneous @ pvm_matrix.T
    w_coords = clip_coords[:, 3]
    w_coords[np.abs(w_coords) < 1e-6] = 1e-6
    ndc = clip_coords[:, :3] / w_coords[:, None]

    valid = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
    h, w = mask_array.shape

    px = (ndc[:, 0] * 0.5 + 0.5) * w
    py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * h

    leaks = np.ones(len(verts_homogeneous), dtype=bool)
    if np.any(valid):
        valid_indices = np.where(valid)[0]
        px_valid = np.clip(px[valid_indices].astype(np.int32), 0, w - 1)
        py_valid = np.clip(py[valid_indices].astype(np.int32), 0, h - 1)
        mask_vals = mask_array[py_valid, px_valid]
        leaks[valid_indices] = mask_vals <= mask_threshold

    leak_ratio = leaks.mean() if leaks.size > 0 else 0.0
    return float(leak_ratio)

def _search_scale_for_view(
    evaluate_fn,
    initial_scale,
    leak_tolerance,
    growth_factor=1.25,
    shrink_factor=0.9,
    max_growth_steps=8,
    max_shrink_steps=8,
    binary_steps=10,
):
    scale = max(initial_scale, 1e-4)
    leak = evaluate_fn(scale)

    safe_scale = None
    unsafe_scale = None

    if leak > leak_tolerance:
        unsafe_scale = scale
        current_scale = scale
        for _ in range(max_growth_steps):
            current_scale *= growth_factor
            if evaluate_fn(current_scale) <= leak_tolerance:
                safe_scale = current_scale
                break
        if safe_scale is None:
            return current_scale
    else:
        safe_scale = scale
        current_scale = scale
        for _ in range(max_shrink_steps):
            test_scale = max(current_scale * shrink_factor, 1e-4)
            if evaluate_fn(test_scale) <= leak_tolerance:
                safe_scale = test_scale
                current_scale = test_scale
            else:
                unsafe_scale = test_scale
                break
        if unsafe_scale is None:
            return safe_scale

    if unsafe_scale is None:
        unsafe_scale = safe_scale / growth_factor

    for _ in range(binary_steps):
        mid_scale = (safe_scale + unsafe_scale) * 0.5
        if evaluate_fn(mid_scale) <= leak_tolerance:
            safe_scale = mid_scale
        else:
            unsafe_scale = mid_scale
    return safe_scale

def _auto_scale_camera_overrides(
    mesh,
    camera_overrides,
    mask_arrays,
    leak_tolerance=0.005,
    mask_threshold=0.05,
    max_samples=8192,
):
    if not camera_overrides or not mask_arrays:
        return camera_overrides
    if len(camera_overrides) != len(mask_arrays):
        raise ValueError("Mask count must match camera overrides for auto scaling.")

    bbox = mesh.bounding_box
    centroid = bbox.centroid
    bbox_extents = bbox.extents
    max_extent = float(np.max(bbox_extents)) if np.any(bbox_extents) else 1.0
    sampled_vertices = _sample_vertices_for_scaling(mesh, max_samples=max_samples)
    if len(sampled_vertices) == 0:
        return camera_overrides
    verts_h = np.hstack((sampled_vertices, np.ones((len(sampled_vertices), 1))))

    mask_height, mask_width = mask_arrays[0].shape
    aspect_ratio = mask_width / mask_height
    cam_up = np.array([0, 1, 0])

    for i, override in enumerate(camera_overrides):
        mask = mask_arrays[i]
        if mask.shape[0] != mask_height or mask.shape[1] != mask_width:
            raise ValueError("All multiview masks must share the same resolution for auto scaling.")

        cam_pos = get_camera_position(centroid, override["distance"], override["azim"], override["elev"])
        view_mat = create_view_matrix(cam_pos, centroid, cam_up)

        def eval_scale(scale_mult):
            ortho_height = max_extent * max(scale_mult, 1e-4)
            proj_mat = create_orthographic_projection(ortho_height, aspect_ratio)
            pvm = proj_mat @ view_mat
            return _evaluate_mask_leakage(verts_h, pvm, mask, mask_threshold)

        override["ortho_scale_mult"] = float(
            _search_scale_for_view(
                eval_scale,
                override.get("ortho_scale_mult", DEFAULT_ORTHO_SCALE),
                leak_tolerance,
            )
        )

    return camera_overrides

def _trim_multiview_data(data, target_count, descriptor):
    if data is None or target_count is None or target_count <= 0:
        return data

    length = _infer_multiview_count(data)
    if length == 0:
        return data
    if length < target_count:
        raise ValueError(f"{descriptor} requires at least {target_count} entries but received {length}.")
    if length == target_count:
        return data

    if isinstance(data, torch.Tensor):
        return data[:target_count]
    if isinstance(data, np.ndarray):
        return data[:target_count]
    if isinstance(data, list):
        return data[:target_count]
    if isinstance(data, tuple):
        return tuple(data[:target_count])
    return data

def _trim_multiview_inputs(multiview_images, multiview_masks, target_count, descriptor):
    trimmed_images = _trim_multiview_data(multiview_images, target_count, descriptor)
    trimmed_masks = _trim_multiview_data(multiview_masks, target_count, f"{descriptor} masks")
    return trimmed_images, trimmed_masks

def _build_projection_context(
    mesh,
    multiview_images,
    projection_mode,
    blend_sharpness,
    angle_cutoff,
    perspective_fov,
    orthographic_width,
    orthographic_height,
    perspective_width,
    perspective_height,
    camera_config=None,
    camera_overrides=None,
    multiview_masks=None,
    use_depth_occlusion=True,
    mask_min_value=1e-3,
):
    images_pil = _multiview_tensor_to_pil(multiview_images)
    if not images_pil:
        raise ValueError("No images provided for projection.")

    mask_arrays = None
    if multiview_masks is not None:
        mask_pil = _multiview_tensor_to_pil(multiview_masks)
        if len(mask_pil) != len(images_pil):
            raise ValueError("Number of multiview masks must match multiview images.")
        mask_arrays = [np.array(m.convert('L'), dtype=np.float32) / 255.0 for m in mask_pil]

    if camera_overrides is not None:
        required_views = len(camera_overrides)
        if len(images_pil) < required_views:
            raise ValueError(
                f"Camera layout expects at least {required_views} images but only {len(images_pil)} were provided."
            )
        if len(images_pil) > required_views:
            images_pil = images_pil[:required_views]
            if mask_arrays is not None:
                mask_arrays = mask_arrays[:required_views]
        camera_set = camera_overrides
    else:
        camera_azims, camera_elevs, cam_distance, ortho_scale_mult = resolve_camera_setup(len(images_pil), camera_config)
        camera_set = [
            {
                "azim": az,
                "elev": el,
                "distance": cam_distance,
                "ortho_scale_mult": ortho_scale_mult,
            }
            for az, el in zip(camera_azims, camera_elevs)
        ]

    width, height = images_pil[0].size
    w, h = width, height
    for img in images_pil:
        if img.size != images_pil[0].size:
            raise ValueError("All multiview images must share the same resolution.")
    if mask_arrays is not None:
        for mask in mask_arrays:
            if mask.shape[0] != h or mask.shape[1] != w:
                raise ValueError("Each multiview mask must match its image resolution.")

    bbox = mesh.bounding_box
    centroid = bbox.centroid
    bbox_extents = bbox.extents
    scene_scale = float(np.linalg.norm(bbox_extents)) if np.any(bbox_extents) else 1.0
    occlusion_epsilon = max(scene_scale * 1e-4, 1e-6)
    cam_target = centroid
    cam_up = np.array([0, 1, 0])
    aspect_ratio = w / h
    projection_mode = projection_mode.lower()

    vertices = mesh.vertices
    verts_homogeneous = None
    if len(vertices) > 0:
        verts_homogeneous = np.hstack((vertices, np.ones((len(vertices), 1))))

    cameras = []
    cos_threshold = np.cos(np.deg2rad(angle_cutoff))

    default_distance = camera_set[0].get("distance", DEFAULT_CAMERA_DISTANCE)
    default_ortho_scale = camera_set[0].get("ortho_scale_mult", DEFAULT_ORTHO_SCALE)

    for i, view in enumerate(camera_set):
        img_array = np.array(images_pil[i].convert('RGBA'))
        current_mask = mask_arrays[i] if mask_arrays is not None else None

        azim = view.get("azim", 0.0)
        elev = view.get("elev", 0.0)
        cam_distance = view.get("distance", default_distance)
        ortho_scale_override = view.get("ortho_scale_mult", default_ortho_scale)

        cam_pos = get_camera_position(centroid, cam_distance, azim, elev)
        view_mat = create_view_matrix(cam_pos, cam_target, cam_up)

        if projection_mode == 'orthographic':
            if orthographic_width > 0 and orthographic_height > 0:
                aspect = orthographic_width / max(orthographic_height, 1e-6)
                proj_mat = create_orthographic_projection(orthographic_height, aspect)
            else:
                max_extent = float(np.max(bbox_extents)) if np.any(bbox_extents) else 1.0
                ortho_height = max_extent * ortho_scale_override
                proj_mat = create_orthographic_projection(ortho_height, aspect_ratio)
        else:
            if perspective_width > 0 and perspective_height > 0:
                fov_y_rad = 2 * np.arctan((perspective_height / 2) / max(cam_distance, 1e-6))
                proj_mat = create_perspective_projection(fov_y_rad, perspective_width / max(perspective_height, 1e-6))
            else:
                fov_y_rad = np.radians(perspective_fov)
                proj_mat = create_perspective_projection(fov_y_rad, aspect_ratio)

        pvm_matrix = proj_mat @ view_mat
        depth_buffer = None

        if use_depth_occlusion and verts_homogeneous is not None:
            clip_coords = verts_homogeneous @ pvm_matrix.T
            w_coords = clip_coords[:, 3]
            w_coords[np.abs(w_coords) < 1e-6] = 1e-6
            ndc = clip_coords[:, :3] / w_coords[:, None]
            frustum_mask = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
            if np.any(frustum_mask):
                frustum_indices = np.where(frustum_mask)[0]
                px = (ndc[frustum_indices, 0] * 0.5 + 0.5) * w
                py = (1 - (ndc[frustum_indices, 1] * 0.5 + 0.5)) * h
                frustum_px = np.clip(px, 0, w - 1).astype(np.int32, copy=False)
                frustum_py = np.clip(py, 0, h - 1).astype(np.int32, copy=False)
                view_vectors = vertices[frustum_indices] - cam_pos
                view_distances = np.linalg.norm(view_vectors, axis=1)
                depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
                np.minimum.at(depth_buffer, (frustum_py, frustum_px), view_distances.astype(np.float32))

        cameras.append(
            {
                "pvm_matrix": pvm_matrix,
                "cam_pos": cam_pos,
                "image": img_array,
                "mask": current_mask,
                "depth_buffer": depth_buffer,
            }
        )

    return {
        "cameras": cameras,
        "image_width": w,
        "image_height": h,
        "blend_sharpness": blend_sharpness,
        "cos_threshold": cos_threshold,
        "mask_min_value": mask_min_value,
        "use_depth_occlusion": use_depth_occlusion,
        "occlusion_epsilon": occlusion_epsilon,
    }

def _project_colors_for_points(points, normals, projection_context):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    num_points = len(points)
    if num_points == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0, 1), dtype=np.float64)

    ones = np.ones((num_points, 1), dtype=np.float64)
    verts_homogeneous = np.hstack((points, ones))

    colors_accumulator = np.zeros((num_points, 4), dtype=np.float64)
    weight_accumulator = np.zeros((num_points, 1), dtype=np.float64)

    h = projection_context["image_height"]
    w = projection_context["image_width"]
    cos_threshold = projection_context["cos_threshold"]
    blend_sharpness = projection_context["blend_sharpness"]
    mask_min_value = projection_context["mask_min_value"]
    occlusion_epsilon = projection_context["occlusion_epsilon"]
    use_depth = projection_context["use_depth_occlusion"]

    for cam in projection_context["cameras"]:
        clip_coords = verts_homogeneous @ cam["pvm_matrix"].T
        w_coords = clip_coords[:, 3]
        w_coords[np.abs(w_coords) < 1e-6] = 1e-6
        ndc = clip_coords[:, :3] / w_coords[:, None]

        frustum_mask = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
        if not np.any(frustum_mask):
            continue

        idx = np.where(frustum_mask)[0]
        px = (ndc[idx, 0] * 0.5 + 0.5) * w
        py = (1 - (ndc[idx, 1] * 0.5 + 0.5)) * h
        px_int = np.clip(px.astype(np.int32), 0, w - 1)
        py_int = np.clip(py.astype(np.int32), 0, h - 1)

        view_vectors = points[idx] - cam["cam_pos"]
        view_distances = np.linalg.norm(view_vectors, axis=1)
        view_distances[view_distances == 0] = 1e-6
        view_directions = view_vectors / view_distances[:, None]
        dot_products = np.sum(normals[idx] * view_directions, axis=1)
        front_mask = dot_products < 0
        if not np.any(front_mask):
            continue

        idx = idx[front_mask]
        px_int = px_int[front_mask]
        py_int = py_int[front_mask]
        view_distances = view_distances[front_mask]
        cos_values = -dot_products[front_mask]
        cos_keep = cos_values >= cos_threshold
        if not np.any(cos_keep):
            continue

        idx = idx[cos_keep]
        px_int = px_int[cos_keep]
        py_int = py_int[cos_keep]
        view_distances = view_distances[cos_keep]
        cos_values = cos_values[cos_keep]

        dynamic_weights = np.power(cos_values, blend_sharpness)

        if use_depth and cam["depth_buffer"] is not None:
            depth_limits = cam["depth_buffer"][py_int, px_int] + occlusion_epsilon
            depth_mask = view_distances <= depth_limits
            if not np.any(depth_mask):
                continue
            idx = idx[depth_mask]
            px_int = px_int[depth_mask]
            py_int = py_int[depth_mask]
            dynamic_weights = dynamic_weights[depth_mask]

        if idx.size == 0:
            continue

        mask_array = cam["mask"]
        if mask_array is not None:
            mask_weights = mask_array[py_int, px_int]
            mask_keep = mask_weights > mask_min_value
            if not np.any(mask_keep):
                continue
            idx = idx[mask_keep]
            px_int = px_int[mask_keep]
            py_int = py_int[mask_keep]
            dynamic_weights = dynamic_weights[mask_keep] * mask_weights[mask_keep]

        if idx.size == 0:
            continue

        sampled_colors = cam["image"][py_int, px_int].astype(np.float64)
        weights = dynamic_weights[:, None]
        colors_accumulator[idx] += sampled_colors * weights
        weight_accumulator[idx] += weights

    return colors_accumulator, weight_accumulator

def _rasterize_face_pixels(uv_triangle, tex_width, tex_height):
    uv_triangle = np.asarray(uv_triangle, dtype=np.float64)
    if uv_triangle.shape != (3, 2):
        uv_triangle = uv_triangle.reshape(3, 2)

    if not np.isfinite(uv_triangle).all():
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    px = uv_triangle[:, 0] * (tex_width - 1)
    py = (1.0 - uv_triangle[:, 1]) * (tex_height - 1)

    min_x = int(max(np.floor(np.min(px)), 0))
    max_x = int(min(np.ceil(np.max(px)), tex_width - 1))
    min_y = int(max(np.floor(np.min(py)), 0))
    max_y = int(min(np.ceil(np.max(py)), tex_height - 1))

    if min_x > max_x or min_y > max_y:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts_x = grid_x + 0.5
    pts_y = grid_y + 0.5

    v0 = np.array([px[1] - px[0], py[1] - py[0]])
    v1 = np.array([px[2] - px[0], py[2] - py[0]])
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if np.abs(denom) < 1e-8:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    v2x = pts_x - px[0]
    v2y = pts_y - py[0]
    b1 = (v2x * v1[1] - v1[0] * v2y) / denom
    b2 = (v0[0] * v2y - v2x * v0[1]) / denom
    b0 = 1.0 - b1 - b2

    eps = 1e-4
    mask = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)
    if not np.any(mask):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    px_indices = grid_x[mask].astype(np.int32, copy=False)
    py_indices = grid_y[mask].astype(np.int32, copy=False)
    bary_coords = np.stack((b0, b1, b2), axis=-1)[mask]
    return px_indices, py_indices, bary_coords.astype(np.float64, copy=False)

def _project_multiview_texture_to_uv(mesh, projection_context, texture_resolution, margin):
    tex_size = int(texture_resolution)
    if tex_size <= 0:
        raise ValueError("texture_resolution must be greater than zero.")

    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
        raise ValueError("Mesh has no UV map. Please use the BlenderUnwrap node first.")

    tex_accumulator = np.zeros((tex_size, tex_size, 4), dtype=np.float64)
    weight_buffer = np.zeros((tex_size, tex_size, 1), dtype=np.float64)
    coverage_mask = np.zeros((tex_size, tex_size), dtype=bool)

    uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
    faces = mesh.faces
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals

    for face in faces:
        uv_triangle = uvs[face]
        px_indices, py_indices, bary_coords = _rasterize_face_pixels(uv_triangle, tex_size, tex_size)
        if px_indices.size == 0:
            continue

        coverage_mask[py_indices, px_indices] = True
        verts = vertices[face]
        norms = vertex_normals[face]

        points = bary_coords @ verts
        normals = bary_coords @ norms
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normal_lengths[normal_lengths == 0] = 1.0
        normals /= normal_lengths

        colors_acc, weights = _project_colors_for_points(points, normals, projection_context)
        valid = weights[:, 0] > 0
        if not np.any(valid):
            continue

        px_valid = px_indices[valid]
        py_valid = py_indices[valid]
        weight_values = weights[valid, 0][:, None]
        tex_accumulator[py_valid, px_valid] += colors_acc[valid] * weight_values
        weight_buffer[py_valid, px_valid] += weight_values

    if not np.any(coverage_mask):
        return np.zeros((tex_size, tex_size, 4), dtype=np.uint8)

    valid_color_mask = weight_buffer[:, :, 0] > 0
    texture = np.zeros_like(tex_accumulator)
    texture[valid_color_mask] = tex_accumulator[valid_color_mask] / weight_buffer[valid_color_mask]
    texture = np.clip(texture, 0.0, 255.0)

    target_mask = coverage_mask.copy()
    if margin > 0:
        bleed = min(int(margin), tex_size - 1)
        if bleed > 0:
            size = min(bleed * 2 + 1, tex_size)
            footprint = np.ones((size, size), dtype=bool)
            target_mask = grey_dilation(target_mask.astype(np.uint8), footprint=footprint) > 0

    needs_fill = target_mask & ~valid_color_mask
    if np.any(valid_color_mask) and np.any(needs_fill):
        _, indices = distance_transform_edt(~valid_color_mask, return_indices=True)
        texture[needs_fill] = texture[indices[0][needs_fill], indices[1][needs_fill]]

    alpha_channel = np.zeros((tex_size, tex_size), dtype=np.uint8)
    alpha_channel[target_mask] = 255
    texture[:, :, 3] = alpha_channel

    return texture.astype(np.uint8)

def _apply_texture_to_mesh_vertices(mesh, texture_image):
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
        return

    tex_height, tex_width = texture_image.shape[0], texture_image.shape[1]
    if tex_height == 0 or tex_width == 0:
        return

    uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
    px = np.clip(np.rint(uvs[:, 0] * (tex_width - 1)).astype(np.int32), 0, tex_width - 1)
    py = np.clip(np.rint((1.0 - uvs[:, 1]) * (tex_height - 1)).astype(np.int32), 0, tex_height - 1)
    vertex_colors = texture_image[py, px]

    if hasattr(mesh.visual, 'vertex_attributes'):
        mesh.visual.vertex_attributes['color'] = vertex_colors.copy()
    elif hasattr(mesh.visual, 'vertex_colors'):
        mesh.visual.vertex_colors = vertex_colors.copy()
    else:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors.copy())

def _load_seqtex_tensor(path, description):
    if not path or not str(path).strip():
        raise ValueError(f"{description} path is empty.")
    resolved = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{description} file not found: {resolved}")
    tensor = torch.load(resolved, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{description} file does not contain a torch.Tensor.")
    if tensor.ndim < 2:
        raise ValueError(f"{description} tensor must be at least 2D.")
    return tensor

def _resolve_seqtex_view_indices(preset, available_count):
    preset = str(preset).strip()
    if preset not in SEQ_TEX_VIEW_PRESETS:
        raise ValueError(f"Unsupported SeqTex view preset '{preset}'.")
    if preset == "2":
        if available_count >= 3:
            return [0, 2]
        if available_count >= 2:
            return [0, 1]
        raise ValueError("SeqTex preset '2' requires at least 2 multiview inputs.")
    required = {"4": 4, "6": 6, "12": 12}[preset]
    if available_count < required:
        raise ValueError(f"SeqTex preset '{preset}' requires at least {required} inputs but only {available_count} were provided.")
    return list(range(required))

def _slice_tensor_first_dim(tensor, indices, label):
    if tensor is None:
        return None
    if tensor.shape[0] < max(indices) + 1:
        raise ValueError(f"{label} only has {tensor.shape[0]} entries but indices {indices} were requested.")
    return tensor[indices]

def _finalize_seqtex_texture(accum_np, weight_np, coverage_mask, margin):
    tex_h, tex_w, _ = accum_np.shape
    valid_mask = weight_np[:, :, 0] > 0
    texture_rgb = np.zeros_like(accum_np)
    if np.any(valid_mask):
        texture_rgb[valid_mask] = accum_np[valid_mask] / weight_np[valid_mask]
    target_mask = coverage_mask.copy()
    if margin > 0:
        bleed = min(int(margin), max(tex_h, tex_w) - 1)
        if bleed > 0:
            size = min(bleed * 2 + 1, max(tex_h, tex_w))
            footprint = np.ones((size, size), dtype=bool)
            target_mask = grey_dilation(target_mask.astype(np.uint8), footprint=footprint) > 0
    needs_fill = target_mask & ~valid_mask
    if np.any(valid_mask) and np.any(needs_fill):
        _, indices = distance_transform_edt(~valid_mask, return_indices=True)
        texture_rgb[needs_fill] = texture_rgb[indices[0][needs_fill], indices[1][needs_fill]]
    texture_rgba = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
    texture_rgba[:, :, :3] = np.clip(texture_rgb * 255.0, 0, 255).astype(np.uint8)
    alpha = np.zeros((tex_h, tex_w), dtype=np.uint8)
    alpha[target_mask] = 255
    texture_rgba[:, :, 3] = alpha
    return texture_rgba

def _convert_vertices_to_seqtex_space(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    seqtex_vertices = vertices.copy()
    seqtex_vertices[:, 1] = -vertices[:, 2]
    seqtex_vertices[:, 2] = vertices[:, 1]
    seqtex_normals = normals.copy()
    seqtex_normals[:, 1] = -normals[:, 2]
    seqtex_normals[:, 2] = normals[:, 1]
    return seqtex_vertices, seqtex_normals

def _project_seqtex_multiview_texture(
    mesh,
    images,
    masks,
    mvp_matrix,
    w2c_matrix,
    texture_resolution,
    margin,
    angle_start,
    angle_end,
    blend_smoothness,
):
    if dr is None:
        raise ImportError("nvdiffrast is required for BakeToModel. Install nvdiffrast to enable this node.")
    if not torch.cuda.is_available():
        raise RuntimeError("BakeToModel requires CUDA but no GPU was detected.")
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
        raise ValueError("Mesh has no UV coordinates; run BlenderUnwrap or provide a UV-mapped mesh.")

    device = torch.device("cuda")
    seqtex_vertices, seqtex_normals = _convert_vertices_to_seqtex_space(mesh)
    verts = torch.as_tensor(seqtex_vertices, dtype=torch.float32, device=device)
    normals = torch.as_tensor(seqtex_normals, dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), dtype=torch.int32, device=device)
    uvs = torch.as_tensor(np.asarray(mesh.visual.uv, dtype=np.float32), dtype=torch.float32, device=device)

    num_views = images.shape[0]
    tex_size = int(texture_resolution)
    if tex_size <= 0:
        raise ValueError("texture_resolution must be greater than zero.")

    images = images.to(device=device, dtype=torch.float32).contiguous()
    img_height = images.shape[1]
    img_width = images.shape[2]
    if masks is not None:
        masks = masks.to(device=device, dtype=torch.float32).contiguous()

    verts_h = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device, dtype=torch.float32)], dim=1)
    verts_h_batch = verts_h.unsqueeze(0).repeat(num_views, 1, 1)
    normals_h = torch.cat([normals, torch.zeros((normals.shape[0], 1), device=device, dtype=torch.float32)], dim=1)
    normals_batch = normals_h.unsqueeze(0).repeat(num_views, 1, 1)
    uvs_batch = uvs.unsqueeze(0).repeat(num_views, 1, 1)

    mvp = mvp_matrix.to(device=device, dtype=torch.float32)
    w2c = w2c_matrix.to(device=device, dtype=torch.float32)
    verts_clip = torch.bmm(verts_h_batch, mvp.transpose(1, 2))
    verts_cam = torch.bmm(verts_h_batch, w2c.transpose(1, 2))[..., :3]
    normals_cam = torch.bmm(normals_batch, w2c.transpose(1, 2))[..., :3]
    normals_cam = F.normalize(normals_cam, dim=2, eps=1e-8)

    combined_attr = torch.cat([uvs_batch, normals_cam, verts_cam], dim=2)
    glctx = dr.RasterizeCudaContext(device=device)
    rast, _ = dr.rasterize(glctx, verts_clip, faces.int(), (img_height, img_width))

    accum = torch.zeros((tex_size * tex_size, 3), device=device, dtype=torch.float32)
    weight_buf = torch.zeros((tex_size * tex_size, 1), device=device, dtype=torch.float32)
    coverage_mask = np.zeros((tex_size, tex_size), dtype=bool)

    cos_full = math.cos(math.radians(angle_start))
    cos_zero = math.cos(math.radians(angle_end))
    cos_range = max(cos_full - cos_zero, 1e-6)
    smooth_strength = max(0.0, min(1.0, float(blend_smoothness)))

    for view_idx in range(num_views):
        img = images[view_idx]
        img = torch.flip(img, dims=[0])
        view_rast = rast[view_idx : view_idx + 1]
        coverage = view_rast[0, ..., 3] > 0
        if not coverage.any():
            continue
        interpolated, _ = dr.interpolate(combined_attr[view_idx : view_idx + 1], view_rast, faces.int())
        interpolated = interpolated[0]
        uv_map = interpolated[..., :2]
        normal_map = F.normalize(interpolated[..., 2:5], dim=-1, eps=1e-8)
        pos_map = interpolated[..., 5:8]
        view_dir = F.normalize(-pos_map, dim=-1, eps=1e-8)
        cos_theta = torch.clamp((normal_map * view_dir).sum(dim=-1), min=0.0)
        angle_weight = torch.clamp((cos_theta - cos_zero) / cos_range, min=0.0, max=1.0)
        if smooth_strength > 1e-6:
            smooth = angle_weight * angle_weight * (3.0 - 2.0 * angle_weight)
            angle_weight = angle_weight + (smooth - angle_weight) * smooth_strength

        mask_flat = coverage.view(-1)
        if not mask_flat.any():
            continue
        uv_flat = uv_map.view(-1, 2)[mask_flat]
        color_flat = img.view(-1, 3)[mask_flat]
        weight_flat = angle_weight.view(-1)[mask_flat]

        if masks is not None:
            mask_view = masks[view_idx]
            if mask_view.dim() == 2:
                mask_vals = mask_view
            else:
                mask_vals = mask_view[..., 0]
            mask_vals = torch.clamp(mask_vals, 0.0, 1.0)
            weight_flat = weight_flat * mask_vals.view(-1)[mask_flat]

        valid = weight_flat > 1e-6
        if not valid.any():
            continue
        uv_flat = uv_flat[valid]
        color_flat = color_flat[valid]
        weight_flat = weight_flat[valid]

        tex_u = torch.clamp((uv_flat[:, 0] * (tex_size - 1)).long(), 0, tex_size - 1)
        tex_v = torch.clamp(((1.0 - uv_flat[:, 1]) * (tex_size - 1)).long(), 0, tex_size - 1)
        tex_idx = tex_v * tex_size + tex_u

        accum.index_add_(0, tex_idx, color_flat * weight_flat[:, None])
        weight_buf.index_add_(0, tex_idx, weight_flat[:, None])

        tex_u_cpu = tex_u.detach().cpu().numpy()
        tex_v_cpu = tex_v.detach().cpu().numpy()
        coverage_mask[tex_v_cpu, tex_u_cpu] = True

    if not coverage_mask.any():
        raise RuntimeError("No texels were touched during projection; verify the camera metadata matches the provided mesh.")

    accum_np = accum.view(tex_size, tex_size, 3).detach().cpu().numpy()
    weight_np = weight_buf.view(tex_size, tex_size, 1).detach().cpu().numpy()
    return _finalize_seqtex_texture(accum_np, weight_np, coverage_mask, margin)

def _ensure_vertex_color_channel(mesh):
    visual = getattr(mesh, 'visual', None)
    if visual is None:
        return False

    vertex_colors = getattr(visual, 'vertex_colors', None)
    if vertex_colors is not None and len(vertex_colors) > 0:
        return True

    vertex_attributes = getattr(visual, 'vertex_attributes', None)
    if vertex_attributes and 'color' in vertex_attributes:
        colors = np.asarray(vertex_attributes['color'])
        if colors.size == 0:
            return False
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        if colors.ndim == 2 and colors.shape[1] == 3:
            alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
            colors = np.concatenate([colors, alpha], axis=1)
        if hasattr(visual, 'vertex_colors'):
            visual.vertex_colors = colors
        else:
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        return True

    return False

def project_multiview_vertex_colors(
    source_mesh,
    multiview_images,
    projection_mode,
    blend_sharpness,
    angle_cutoff,
    perspective_fov,
    orthographic_width,
    orthographic_height,
    perspective_width,
    perspective_height,
    camera_config=None,
    fill_unpainted=True,
    multiview_masks=None,
    use_depth_occlusion=True,
    mask_min_value=1e-3,
):
    mesh = source_mesh.copy()

    projection_context = _build_projection_context(
        mesh,
        multiview_images,
        projection_mode,
        blend_sharpness,
        angle_cutoff,
        perspective_fov,
        orthographic_width,
        orthographic_height,
        perspective_width,
        perspective_height,
        camera_config=camera_config,
        multiview_masks=multiview_masks,
        use_depth_occlusion=use_depth_occlusion,
        mask_min_value=mask_min_value,
    )

    vertex_colors_accumulator, weight_accumulator = _project_colors_for_points(
        mesh.vertices, mesh.vertex_normals, projection_context
    )

    valid_color_mask = weight_accumulator[:, 0] > 0
    missing_mask = ~valid_color_mask

    safe_weights = weight_accumulator.copy()
    safe_weights[missing_mask, 0] = 1.0

    final_colors = (vertex_colors_accumulator / safe_weights)
    final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)

    final_colors[valid_color_mask, 3] = 255

    if fill_unpainted and np.any(missing_mask):
        colored_indices = np.where(valid_color_mask)[0]
        if colored_indices.size > 0:
            tree = cKDTree(mesh.vertices[colored_indices])
            query_points = mesh.vertices[missing_mask]
            _, nearest_idx = tree.query(query_points, k=1)
            nearest_colors = final_colors[colored_indices[nearest_idx]]
            final_colors[missing_mask] = nearest_colors

    if hasattr(mesh.visual, 'vertex_attributes'):
        mesh.visual.vertex_attributes['color'] = final_colors
    elif hasattr(mesh.visual, 'vertex_colors'):
        mesh.visual.vertex_colors = final_colors
    else:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=final_colors)

    return mesh

class VertexToHighPoly:
    PROJECTION_MODES = ["orthographic", "perspective"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "projection_mode": (cls.PROJECTION_MODES, {"default": "orthographic"}),
                "blend_sharpness": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "angle_cutoff": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 180.0, "step": 0.5}),
                "perspective_fov": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "orthographic_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "orthographic_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "perspective_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "perspective_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "use_depth_occlusion": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "multiview_masks": ("MASK",),
                "seqtex_view_preset": ([""] + SEQ_TEX_VIEW_PRESETS, {"default": ""}),
                "seqtex_rotation_offset": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("high_poly_mesh",)
    FUNCTION = "project"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def project(
        self,
        high_poly_mesh,
        multiview_images,
        projection_mode,
        blend_sharpness,
        angle_cutoff,
        perspective_fov,
        orthographic_width,
        orthographic_height,
        perspective_width,
        perspective_height,
        use_depth_occlusion,
        camera_config=None,
        multiview_masks=None,
        seqtex_view_preset="",
        seqtex_rotation_offset=0.0,
    ):
        multiview_images_to_use = multiview_images
        multiview_masks_to_use = multiview_masks
        effective_camera_config = camera_config

        preset = (seqtex_view_preset or "").strip()
        use_seqtex_blender = bool(preset)
        if preset:
            if preset not in SEQ_TEX_VIEW_PRESETS:
                raise ValueError(f"Unsupported SeqTex view preset '{preset}'.")
            azims, elevs = _seqtex_view_preset_angles(preset, 0.0, azimuth_offset=seqtex_rotation_offset)
            required_views = len(azims)
            multiview_images_to_use, multiview_masks_to_use = _trim_multiview_inputs(
                multiview_images,
                multiview_masks,
                required_views,
                "SeqTex multiview images",
            )
            effective_camera_config = _build_seqtex_camera_config(
                high_poly_mesh,
                azims,
                elevs,
                camera_lens=50.0,
                camera_sensor_width=36.0,
                ortho_scale=1.2,
            )

        mesh_result = None
        if use_seqtex_blender:
            try:
                mesh_result = _project_multiview_vertex_colors_with_blender(
                    high_poly_mesh,
                    multiview_images_to_use,
                    projection_mode,
                    blend_sharpness,
                    angle_cutoff,
                    perspective_fov,
                    orthographic_width,
                    orthographic_height,
                    perspective_width,
                    perspective_height,
                    camera_config=effective_camera_config,
                    multiview_masks=multiview_masks_to_use,
                    use_depth_occlusion=use_depth_occlusion,
                    mask_min_value=1e-3,
                    fill_unpainted=True,
                )
            except Exception as blender_exc:
                print(
                    f"[SeqTexProjection] Blender projection failed ({blender_exc}). Falling back to CPU implementation.",
                    file=sys.stderr,
                )

        if mesh_result is None:
            mesh_result = project_multiview_vertex_colors(
                high_poly_mesh,
                multiview_images_to_use,
                projection_mode,
                blend_sharpness,
                angle_cutoff,
                perspective_fov,
                orthographic_width,
                orthographic_height,
                perspective_width,
                perspective_height,
                camera_config=effective_camera_config,
                fill_unpainted=True,
                multiview_masks=multiview_masks_to_use,
                use_depth_occlusion=use_depth_occlusion,
            )
        return (mesh_result,)


class MultiviewDisplaceMesh:
    PROJECTION_MODES = ["orthographic", "perspective"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "projection_mode": (cls.PROJECTION_MODES, {"default": "orthographic"}),
                "strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "auto_displacement_map": ("BOOLEAN", {"default": True}),
                "auto_displacement_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "blend_sharpness": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "angle_cutoff": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 180.0, "step": 0.5}),
                "perspective_fov": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "orthographic_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "orthographic_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "perspective_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "perspective_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("displaced_mesh",)
    FUNCTION = "displace"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def displace(
        self,
        mesh,
        multiview_images,
        projection_mode,
        strength,
        auto_displacement_map,
        auto_displacement_contrast,
        blend_sharpness,
        angle_cutoff,
        perspective_fov,
        orthographic_width,
        orthographic_height,
        perspective_width,
        perspective_height,
        camera_config=None,
    ):
        mesh = mesh.copy()
        images_pil = [Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)) for img in multiview_images]

        if not images_pil:
            raise ValueError("No images provided for projection.")

        camera_azims, camera_elevs, cam_distance, ortho_scale_mult = resolve_camera_setup(len(images_pil), camera_config)

        h, w, _ = np.array(images_pil[0]).shape
        aspect_ratio = w / h
        centroid = mesh.bounding_box.centroid
        cam_target = centroid
        cam_up = np.array([0, 1, 0])

        displacement_accumulator = np.zeros((len(mesh.vertices), 1), dtype=np.float64)
        weight_accumulator = np.zeros((len(mesh.vertices), 1), dtype=np.float64)

        normals = mesh.vertex_normals.copy()

        luminance_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        processed_images = []

        if auto_displacement_map:
            for img_pil in images_pil:
                gray = np.array(img_pil.convert('L'), dtype=np.float32) / 255.0
                min_val = float(np.min(gray))
                max_val = float(np.max(gray))
                if max_val > min_val:
                    gray = (gray - min_val) / (max_val - min_val)
                else:
                    gray.fill(0.5)
                processed_images.append(gray)
        else:
            processed_images = [np.array(img.convert('RGBA')) for img in images_pil]

        for i, (azim, elev) in enumerate(zip(camera_azims, camera_elevs)):
            img_data = processed_images[i]
            cam_pos = get_camera_position(centroid, cam_distance, azim, elev)
            view_mat = create_view_matrix(cam_pos, cam_target, cam_up)

            if projection_mode == 'orthographic':
                if orthographic_width > 0 and orthographic_height > 0:
                    proj_mat = create_orthographic_projection(orthographic_height, orthographic_width / orthographic_height)
                else:
                    extents = mesh.bounding_box.extents
                    max_extent = float(np.max(extents))
                    ortho_height = max_extent * ortho_scale_mult
                    proj_mat = create_orthographic_projection(ortho_height, aspect_ratio)
            else:
                if perspective_width > 0 and perspective_height > 0:
                    fov_y_rad = 2 * np.arctan((perspective_height / 2) / cam_distance)
                    proj_mat = create_perspective_projection(fov_y_rad, perspective_width / perspective_height)
                else:
                    fov_y_rad = np.radians(perspective_fov)
                    proj_mat = create_perspective_projection(fov_y_rad, aspect_ratio)

            pvm_matrix = proj_mat @ view_mat
            vertices = mesh.vertices
            verts_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
            clip_coords = verts_homogeneous @ pvm_matrix.T

            w_coords = clip_coords[:, 3]
            w_coords[np.abs(w_coords) < 1e-6] = 1e-6
            ndc = clip_coords[:, :3] / w_coords[:, None]

            px = (ndc[:, 0] * 0.5 + 0.5) * w
            py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * h

            frustum_indices = np.where(
                (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
            )[0]

            view_vectors = vertices - cam_pos
            view_vectors /= np.linalg.norm(view_vectors, axis=1)[:, np.newaxis]
            dot_products = np.sum(normals * view_vectors, axis=1)
            front_facing_vertex_indices = np.where(dot_products < 0)[0]

            valid_indices = np.intersect1d(frustum_indices, front_facing_vertex_indices, assume_unique=True)

            cos = -dot_products[valid_indices]
            cos_thres = np.cos(np.deg2rad(angle_cutoff))
            cos[cos < cos_thres] = 0.0
            dynamic_weights = np.power(cos, blend_sharpness)
            keep = dynamic_weights > 0
            valid_indices = valid_indices[keep]
            dynamic_weights = dynamic_weights[keep]

            if valid_indices.size == 0:
                continue

            valid_px = np.clip(px[valid_indices], 0, w - 1).astype(int)
            valid_py = np.clip(py[valid_indices], 0, h - 1).astype(int)

            if auto_displacement_map:
                displacement_values = img_data[valid_py, valid_px].astype(np.float32)
            else:
                sampled_colors = img_data[valid_py, valid_px, :3].astype(np.float32) / 255.0
                displacement_values = sampled_colors @ luminance_weights
            if auto_displacement_map:
                centered_brightness = (displacement_values - 0.5) * auto_displacement_contrast
            else:
                centered_brightness = displacement_values - 0.5

            displacement_accumulator[valid_indices] += centered_brightness[:, np.newaxis] * dynamic_weights[:, np.newaxis]
            weight_accumulator[valid_indices] += dynamic_weights[:, np.newaxis]

        valid_mask = weight_accumulator[:, 0] > 0
        final_displacement = np.zeros((len(mesh.vertices), 1), dtype=np.float64)
        final_displacement[valid_mask, 0] = displacement_accumulator[valid_mask, 0] / weight_accumulator[valid_mask, 0]

        final_displacement *= (strength * 2.0)

        mesh.vertices = mesh.vertices + normals * final_displacement
        return (mesh,)

class MultiviewTextureBake:
    PROJECTION_MODES = ["orthographic", "perspective"]
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "projection_mode": (cls.PROJECTION_MODES, {"default": "orthographic"}),
                "texture_resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 4096}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "use_seqtex_mv": ("BOOLEAN", {"default": False}),
                "use_depth_occlusion": ("BOOLEAN", {"default": True}),
                "blend_sharpness": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "angle_cutoff": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 180.0, "step": 0.5}),
                "perspective_fov": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "orthographic_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "orthographic_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "perspective_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "perspective_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "seqtex_rotation_offset": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "seqtex_view_preset": ([""] + SEQ_TEX_VIEW_PRESETS, {"default": ""}),
                "multiview_masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("mesh_with_vertex_colors", "color_map")
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def bake(
        self,
        mesh,
        multiview_images,
        projection_mode,
        texture_resolution,
        margin,
        use_gpu,
        use_seqtex_mv,
        use_depth_occlusion,
        blend_sharpness,
        angle_cutoff,
        perspective_fov,
        orthographic_width,
        orthographic_height,
        perspective_width,
        perspective_height,
        brightness,
        contrast,
        saturation,
        camera_config=None,
        seqtex_rotation_offset=0.0,
        seqtex_view_preset="",
        multiview_masks=None,
    ):
        has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        if not has_uv:
            raise ValueError("Mesh has no UV map. Please use the BlenderUnwrap node first.")

        num_views = _infer_multiview_count(multiview_images)
        if num_views <= 0:
            raise ValueError("At least one multiview image is required.")

        effective_camera_config = camera_config
        effective_perspective_fov = perspective_fov
        effective_orthographic_width = orthographic_width
        effective_orthographic_height = orthographic_height
        multiview_images_to_use = multiview_images
        multiview_masks_to_use = multiview_masks
        expected_views = None

        if use_seqtex_mv:
            preset = (seqtex_view_preset or "").strip()
            if not preset:
                guess = str(num_views)
                preset = guess if guess in SEQ_TEX_VIEW_PRESETS else SEQ_TEX_VIEW_PRESETS[-1]
            if preset not in SEQ_TEX_VIEW_PRESETS:
                raise ValueError(f"Unsupported SeqTex view preset '{preset}'.")

            azims, elevs = _seqtex_view_preset_angles(preset, 0.0, azimuth_offset=seqtex_rotation_offset)
            required_views = len(azims)
            effective_camera_config = _build_seqtex_camera_config(
                mesh,
                azims,
                elevs,
                camera_lens=50.0,
                camera_sensor_width=36.0,
                ortho_scale=1.2,
            )
            effective_perspective_fov = float(np.degrees(2.0 * np.arctan(36.0 / (2.0 * 50.0))))
            if orthographic_width > 0 and orthographic_height > 0:
                effective_orthographic_width = orthographic_width
                effective_orthographic_height = orthographic_height
            else:
                effective_orthographic_width = 2.1
                effective_orthographic_height = 2.1
            expected_views = len(azims)
        elif camera_config:
            azims = camera_config.get("selected_camera_azims")
            elevs = camera_config.get("selected_camera_elevs")
            if azims and elevs:
                expected_views = min(len(azims), len(elevs))

        if expected_views:
            multiview_images_to_use, multiview_masks_to_use = _trim_multiview_inputs(
                multiview_images_to_use,
                multiview_masks_to_use,
                expected_views,
                "Camera configuration multiview images",
            )

        textured_mesh = None
        if use_seqtex_mv:
            try:
                textured_mesh = _project_multiview_vertex_colors_with_blender(
                    mesh,
                    multiview_images_to_use,
                    projection_mode,
                    blend_sharpness,
                    angle_cutoff,
                    effective_perspective_fov,
                    effective_orthographic_width,
                    effective_orthographic_height,
                    perspective_width,
                    perspective_height,
                    camera_config=effective_camera_config,
                    multiview_masks=multiview_masks_to_use,
                    use_depth_occlusion=use_depth_occlusion,
                    mask_min_value=1e-3,
                    fill_unpainted=True,
                )
            except Exception as blender_exc:
                print(
                    f"[SeqTexProjection] Blender projection failed ({blender_exc}). Falling back to CPU implementation.",
                    file=sys.stderr,
                )

        if textured_mesh is None:
            textured_mesh = project_multiview_vertex_colors(
                mesh,
                multiview_images_to_use,
                projection_mode,
                blend_sharpness,
                angle_cutoff,
                effective_perspective_fov,
                effective_orthographic_width,
                effective_orthographic_height,
                perspective_width,
                perspective_height,
                camera_config=effective_camera_config,
                fill_unpainted=True,
                multiview_masks=multiview_masks_to_use,
                use_depth_occlusion=use_depth_occlusion,
            )

        color_map_tensor = self._bake_vertex_colors_to_texture(
            textured_mesh,
            resolution=int(texture_resolution),
            margin=margin,
            use_gpu=use_gpu,
        )

        color_map_tensor = color_map_tensor * brightness
        color_map_tensor = (color_map_tensor - 0.5) * contrast + 0.5
        if saturation != 1.0:
            luma_weights = color_map_tensor.new_tensor([0.299, 0.587, 0.114])
            luminance = (color_map_tensor * luma_weights).sum(dim=-1, keepdim=True)
            color_map_tensor = luminance + (color_map_tensor - luminance) * saturation
        color_map_tensor = torch.clamp(color_map_tensor, 0.0, 1.0)

        return (textured_mesh, color_map_tensor)

class BakeToModel:
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]
    PRESETS = SEQ_TEX_VIEW_PRESETS.copy()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "seqtex_view_preset": (cls.PRESETS, {"default": "6"}),
                "texture_resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 4096}),
                "mvp_matrix_path": ("STRING", {"default": ""}),
                "w2c_matrix_path": ("STRING", {"default": ""}),
                "blend_angle_start": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 85.0, "step": 0.5}),
                "blend_angle_end": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 120.0, "step": 0.5}),
                "blend_smoothness": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "multiview_masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("mesh_with_vertex_colors", "color_map")
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def bake(
        self,
        mesh,
        multiview_images,
        seqtex_view_preset,
        texture_resolution,
        margin,
        mvp_matrix_path,
        w2c_matrix_path,
        blend_angle_start,
        blend_angle_end,
        blend_smoothness,
        camera_config=None,
        multiview_masks=None,
    ):
        if camera_config is None:
            raise ValueError("BakeToModel requires a camera_config from SeqTexCam to ensure preset alignment.")
        if not isinstance(multiview_images, torch.Tensor) or multiview_images.ndim != 4:
            raise ValueError("multiview_images must be a 4D IMAGE tensor (batch, height, width, channels).")
        if blend_angle_end <= blend_angle_start:
            raise ValueError("blend_angle_end must be greater than blend_angle_start.")

        preset_indices = _resolve_seqtex_view_indices(seqtex_view_preset, multiview_images.shape[0])
        images_selected = _slice_tensor_first_dim(multiview_images, preset_indices, "Multiview images").contiguous()

        masks_selected = None
        if multiview_masks is not None:
            if not isinstance(multiview_masks, torch.Tensor) or multiview_masks.ndim < 3:
                raise ValueError("multiview_masks must be a tensor shaped (N, H, W) or (N, H, W, 1).")
            masks_selected = _slice_tensor_first_dim(multiview_masks, preset_indices, "Multiview masks").contiguous()
            if masks_selected.ndim == 4 and masks_selected.shape[-1] == 1:
                masks_selected = masks_selected[..., 0]

        mvp_tensor = _load_seqtex_tensor(mvp_matrix_path, "MVP matrix")
        w2c_tensor = _load_seqtex_tensor(w2c_matrix_path, "World-to-camera matrix")
        mvp_tensor = _slice_tensor_first_dim(mvp_tensor, preset_indices, "MVP tensor")
        w2c_tensor = _slice_tensor_first_dim(w2c_tensor, preset_indices, "W2C tensor")

        if mvp_tensor.shape[1:] != (4, 4):
            raise ValueError(f"MVP tensor must have shape (N, 4, 4); got {mvp_tensor.shape}.")
        if w2c_tensor.shape[1:] != (4, 4):
            raise ValueError(f"W2C tensor must have shape (N, 4, 4); got {w2c_tensor.shape}.")

        tex_res = int(texture_resolution)
        if tex_res <= 0:
            raise ValueError("texture_resolution must be positive.")

        selected_view_count = len(preset_indices)
        cam_azims = camera_config.get("selected_camera_azims") if isinstance(camera_config, dict) else None
        if cam_azims and len(cam_azims) < selected_view_count:
            raise ValueError(
                f"camera_config only defines {len(cam_azims)} cameras but preset '{seqtex_view_preset}' needs {selected_view_count}."
            )

        images_selected = torch.clamp(images_selected, 0.0, 1.0)
        if masks_selected is not None:
            masks_selected = torch.clamp(masks_selected, 0.0, 1.0)

        texture_rgba = _project_seqtex_multiview_texture(
            mesh,
            images_selected,
            masks_selected,
            mvp_tensor,
            w2c_tensor,
            tex_res,
            int(margin),
            float(blend_angle_start),
            float(blend_angle_end),
            float(blend_smoothness),
        )

        mesh_result = mesh.copy()
        _apply_texture_to_mesh_vertices(mesh_result, texture_rgba)

        color_map = torch.from_numpy(texture_rgba[:, :, :3].astype(np.float32) / 255.0).unsqueeze(0)
        return (mesh_result, color_map)

class AutoBakeTextureFromMV:
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]
    CAMERA_LAYOUTS = ["AUTO", "2", "4", "6", "12"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "multiview_masks": ("MASK",),
                "texture_resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 512}),
                "camera_layout": (cls.CAMERA_LAYOUTS, {"default": "AUTO"}),
                "camera_distance_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "front_back_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.01}),
                "side_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.01}),
                "diagonal_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.01}),
                "top_bottom_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.01}),
                "blend_sharpness": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "angle_cutoff": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 180.0, "step": 0.5}),
                "leak_tolerance": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.25, "step": 0.0005}),
                "mask_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005}),
                "use_depth_occlusion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("mesh_with_texture", "color_map")
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def bake(
        self,
        mesh,
        multiview_images,
        multiview_masks,
        texture_resolution,
        margin,
        camera_layout,
        camera_distance_scale,
        front_back_scale,
        side_scale,
        diagonal_scale,
        top_bottom_scale,
        blend_sharpness,
        angle_cutoff,
        leak_tolerance,
        mask_threshold,
        use_depth_occlusion,
    ):
        images_pil = _multiview_tensor_to_pil(multiview_images)
        mask_pil = _multiview_tensor_to_pil(multiview_masks)

        if not images_pil:
            raise ValueError("No multiview images provided.")
        if len(images_pil) != len(mask_pil):
            raise ValueError("Number of multiview masks must match the multiview images.")

        mask_arrays = [np.array(m.convert('L'), dtype=np.float32) / 255.0 for m in mask_pil]

        layout_key = camera_layout.upper()
        if layout_key not in self.CAMERA_LAYOUTS:
            raise ValueError(f"Unsupported camera layout '{camera_layout}'.")

        if layout_key == "AUTO":
            images_used = images_pil
            masks_used = mask_arrays
        else:
            layout_size = len(CUSTOM_CAMERA_LAYOUTS[layout_key])
            if len(images_pil) < layout_size:
                raise ValueError(
                    f"Camera layout '{layout_key}' requires at least {layout_size} images but {len(images_pil)} were provided."
                )
            images_used = images_pil[:layout_size]
            masks_used = mask_arrays[:layout_size]

        for mask in masks_used:
            if mask.shape != masks_used[0].shape:
                raise ValueError("All masks must share the same resolution for auto-scaling.")

        num_views = len(images_used)
        if num_views == 0:
            raise ValueError("At least one multiview image is required.")

        if layout_key == "AUTO":
            camera_azims, camera_elevs, base_distance, base_ortho_scale = resolve_camera_setup(num_views)
            base_distance *= camera_distance_scale
            camera_overrides = [
                {
                    "azim": az,
                    "elev": el,
                    "distance": base_distance,
                    "ortho_scale_mult": base_ortho_scale,
                }
                for az, el in zip(camera_azims, camera_elevs)
            ]
        else:
            camera_overrides = _build_custom_camera_overrides(
                layout_key,
                num_views,
                camera_distance_scale,
                front_back_scale,
                side_scale,
                diagonal_scale,
                top_bottom_scale,
            )

        camera_overrides = _auto_scale_camera_overrides(
            mesh,
            camera_overrides,
            masks_used,
            leak_tolerance=leak_tolerance,
            mask_threshold=mask_threshold,
        )

        def _images_to_tensor(pil_list):
            tensors = []
            for img in pil_list:
                arr = np.array(img.convert('RGBA')).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr))
            return torch.stack(tensors)

        def _masks_to_tensor(mask_list):
            tensors = []
            for mask in mask_list:
                tensors.append(torch.from_numpy(mask.astype(np.float32)))
            return torch.stack(tensors)

        images_tensor = _images_to_tensor(images_used)
        masks_tensor = _masks_to_tensor(masks_used)

        projection_context = _build_projection_context(
            mesh,
            images_tensor,
            "orthographic",
            blend_sharpness,
            angle_cutoff,
            perspective_fov=0.0,
            orthographic_width=0.0,
            orthographic_height=0.0,
            perspective_width=0.0,
            perspective_height=0.0,
            camera_config=None,
            camera_overrides=camera_overrides,
            multiview_masks=masks_tensor,
            use_depth_occlusion=use_depth_occlusion,
        )

        texture_image = _project_multiview_texture_to_uv(
            mesh,
            projection_context,
            texture_resolution=int(texture_resolution),
            margin=margin,
        )

        color_map_tensor = torch.from_numpy(
            texture_image[:, :, :3].astype(np.float32) / 255.0
        )[None, ...]

        textured_mesh = mesh.copy()
        _apply_texture_to_mesh_vertices(textured_mesh, texture_image)

        return (textured_mesh, color_map_tensor)

class VertexColorBake:
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]
    BAKE_TYPES = ["diffuse"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "low_poly_mesh": ("TRIMESH",),
                "resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "bake_type": (cls.BAKE_TYPES, {"default": "diffuse"}),
                "cage_extrusion": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_ray_distance": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
                "margin": ("INT", {"default": 1024, "min": 0, "max": 2048}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "brightness": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE",)
    RETURN_NAMES = ("low_poly_mesh", "color_map",)
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def bake(self, high_poly_mesh, low_poly_mesh, resolution, bake_type, cage_extrusion, max_ray_distance, margin, use_gpu, brightness, contrast):
        if not _ensure_vertex_color_channel(high_poly_mesh):
            raise ValueError("High-poly mesh does not have vertex colors to bake.")

        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            high_poly_path = os.path.join(temp_dir, "high.glb")
            low_poly_path = os.path.join(temp_dir, "low.glb")
            final_low_poly_path = os.path.join(temp_dir, "final_low.glb")
            script_path = os.path.join(temp_dir, "s.py")

            high_poly_mesh.export(file_obj=high_poly_path)
            low_poly_mesh.export(file_obj=low_poly_path)

            params = {
                'high_poly_path': high_poly_path, 'low_poly_path': low_poly_path,
                'final_low_poly_path': final_low_poly_path, 'temp_dir': temp_dir,
                'bake_type': bake_type.upper(),
                'resolution': int(resolution), 
                'cage_extrusion': cage_extrusion, 'max_ray_distance': max_ray_distance, 'margin': margin,
                'use_gpu': use_gpu
            }

            clean_mesh_func_script = """
def clean_mesh(obj, merge_distance):
    import bpy
    from bpy.types import Context
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    
    override = bpy.context.copy()
    override['object'] = obj
    override['active_object'] = obj
    override['selected_objects'] = [obj]
    override['mode'] = 'EDIT_MESH'
    override['edit_object'] = obj
    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='SELECT')
        if merge_distance > 0.0:
            bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except:
            pass
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.shade_smooth()
"""

            script = f'''
{clean_mesh_func_script}
import bpy, sys, os, traceback
p = {{ {", ".join(f'"{k}": r"{v}"' if isinstance(v, str) else f'"{k}": {v}' for k, v in params.items())} }}

def setup_gpu():
    if not p['use_gpu']:
        print("GPU baking disabled by user.")
        bpy.context.scene.cycles.device = 'CPU'
        return

    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'NONE'

        device_types = ['OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI']
        for device_type in device_types:
            if hasattr(prefs, f'get_devices_for_type'):
                devices = prefs.get_devices_for_type(device_type)
                if devices:
                    prefs.compute_device_type = device_type
                    print(f"Found and set compute device type to: {{device_type}}")
                    break
        
        prefs.get_devices()
        for device in prefs.devices:
            if device.type != 'CPU':
                device.use = True
                print(f"Enabled GPU device: {{device.name}}")
        
        bpy.context.scene.cycles.device = 'GPU'
        print("Successfully set scene device to GPU.")
    except Exception as e:
        print(f"Could not setup GPU, falling back to CPU. Error: {{e}}", file=sys.stderr)
        bpy.context.scene.cycles.device = 'CPU'

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = 'CYCLES'
    setup_gpu()
    bpy.context.scene.cycles.samples = 1
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.cycles.use_direct_light = False
    bpy.context.scene.cycles.use_indirect_light = False

def align_meshes(high_obj, low_obj):
    import bpy
    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True)
    bpy.context.view_layer.objects.active = high_obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    low_obj.location = high_obj.location
    low_obj.scale = high_obj.scale

def import_meshes():
    bpy.ops.import_scene.gltf(filepath=p['high_poly_path'])
    bpy.context.view_layer.update()
    high_obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    high_obj.name = "HighPoly"
    
    if not high_obj.data.color_attributes:
        raise Exception("High-poly mesh has no vertex colors.")
    
    bpy.ops.import_scene.gltf(filepath=p['low_poly_path'])
    bpy.context.view_layer.update()
    low_obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name != "HighPoly")
    low_obj.name = "LowPoly"
    
    align_meshes(high_obj, low_obj)
    clean_mesh(high_obj, 0.0001)
    clean_mesh(low_obj, 0.0001)
    
    if not low_obj.data.uv_layers:
        raise Exception("Low-poly mesh has no UV map. Please use the BlenderUnwrap node first.")
    
    return high_obj, low_obj

def setup_lowpoly_material(low_obj):
    mat = bpy.data.materials.new(name="BakeMaterial")
    mat.use_nodes = True
    if low_obj.data.materials: low_obj.data.materials[0] = mat
    else: low_obj.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    tex_node = nodes.new('ShaderNodeTexImage')
    nodes.active = tex_node
    return mat, tex_node

def execute_bake(bake_type, tex_node):
    res = p['resolution']
    bake_image = bpy.data.images.new(name=f"Color_BakeImage", width=res, height=res, alpha=False)
    tex_node.image = bake_image
    
    bake_kwargs = {{
        'type': bake_type, 'use_selected_to_active': True,
        'margin': p['margin'], 'cage_extrusion': p['cage_extrusion'], 'use_clear': True,
    }}
    if p['max_ray_distance'] > 0.0:
        bake_kwargs['max_ray_distance'] = p['max_ray_distance']
    if bake_type == 'DIFFUSE':
        bake_kwargs['pass_filter'] = {{'COLOR'}}

    bpy.ops.object.select_all(action='DESELECT')
    low_obj = [o for o in bpy.context.scene.objects if o.name == "LowPoly"][0]
    high_obj = [o for o in bpy.context.scene.objects if o.name == "HighPoly"][0]
    high_obj.select_set(True)
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    bpy.context.view_layer.update()
    
    bpy.ops.object.bake(**bake_kwargs)
    return bake_image

try:
    setup_scene()
    high_obj, low_obj = import_meshes()
    low_poly_mat, tex_node = setup_lowpoly_material(low_obj)

    color_image = execute_bake(p['bake_type'], tex_node)
    output_path = os.path.join(p['temp_dir'], "color_map.png")
    color_image.filepath_raw = output_path
    color_image.file_format = 'PNG'; color_image.save()
    bpy.data.images.remove(color_image)

    clean_mesh(low_obj, 0.0)
    bpy.ops.object.select_all(action='DESELECT')
    low_obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=p['final_low_poly_path'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            def load_image_as_tensor(path):
                if not os.path.exists(path): return None
                img = Image.open(path).convert('RGB')
                return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

            color_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "color_map.png"))

            if color_map_tensor is None:
                print(f"Warning: Failed to load color map at {os.path.join(temp_dir, 'color_map.png')}")
                final_mesh = trimesh.load(final_low_poly_path, force="mesh")
                return (final_mesh, dummy_image)

            color_map_tensor = color_map_tensor * brightness
            color_map_tensor = (color_map_tensor - 0.5) * contrast + 0.5
            color_map_tensor = torch.clamp(color_map_tensor, 0.0, 1.0)

            final_mesh = trimesh.load(final_low_poly_path, force="mesh")

            return (final_mesh, color_map_tensor)
