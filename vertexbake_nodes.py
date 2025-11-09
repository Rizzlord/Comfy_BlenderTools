import os
import sys
import tempfile
import torch
import numpy as np
import folder_paths
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, grey_dilation
from .utils import _run_blender_script, get_blender_clean_mesh_func_script

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
        8: (
            [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            [15.0, 15.0, 0.0, 0.0, 0.0, 0.0, -15.0, -15.0],
        ),
    }

    if num_views in default_layouts:
        camera_azims, camera_elevs = default_layouts[num_views]
    else:
        step = 360.0 / num_views
        camera_azims = [step * i for i in range(num_views)]
        camera_elevs = [0.0] * num_views

    return camera_azims, camera_elevs, 1.45, 1.2

def _infer_multiview_count(multiview_images):
    if isinstance(multiview_images, torch.Tensor):
        return int(multiview_images.shape[0])
    try:
        return len(multiview_images)
    except TypeError:
        return 0

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

    extents = np.array(getattr(mesh.bounding_box, "extents", [1.0, 1.0, 1.0]), dtype=np.float64)
    if extents.size != 3:
        extents = np.ones(3, dtype=np.float64)

    max_extent = float(np.max(extents))
    if max_extent <= 1e-6:
        normalized_diag = 2.0
    else:
        normalized_diag = (2.0 / max_extent) * float(np.linalg.norm(extents))

    distance = (camera_lens / camera_sensor_width) * normalized_diag
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
    mask_min_value=1e-3,
):
    mesh = source_mesh.copy()
    images_pil = _multiview_tensor_to_pil(multiview_images)

    if not images_pil:
        raise ValueError("No images provided for projection.")

    mask_arrays = None
    if multiview_masks is not None:
        mask_pil = _multiview_tensor_to_pil(multiview_masks)
        if len(mask_pil) != len(images_pil):
            raise ValueError("Number of multiview masks must match multiview images.")
        mask_arrays = [np.array(m.convert('L'), dtype=np.float32) / 255.0 for m in mask_pil]

    camera_azims, camera_elevs, cam_distance, ortho_scale_mult = resolve_camera_setup(len(images_pil), camera_config)

    h, w, _ = np.array(images_pil[0]).shape
    aspect_ratio = w / h
    centroid = mesh.bounding_box.centroid
    cam_target = centroid
    cam_up = np.array([0, 1, 0])

    vertex_colors_accumulator = np.zeros((len(mesh.vertices), 4), dtype=np.float64)
    weight_accumulator = np.zeros((len(mesh.vertices), 1), dtype=np.float64)

    normals = mesh.vertex_normals

    for i, (azim, elev) in enumerate(zip(camera_azims, camera_elevs)):
        img_pil = images_pil[i]
        img = np.array(img_pil.convert('RGBA'))
        current_mask = None
        if mask_arrays is not None:
            current_mask = mask_arrays[i]
            if current_mask.shape[0] != h or current_mask.shape[1] != w:
                raise ValueError("Each multiview mask must match its image resolution.")
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
        norms = np.linalg.norm(view_vectors, axis=1)
        norms[norms == 0] = 1e-6
        view_vectors /= norms[:, np.newaxis]
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

        if current_mask is not None:
            mask_weights = current_mask[valid_py, valid_px]
            mask_keep = mask_weights > mask_min_value
            if not np.any(mask_keep):
                continue
            valid_indices = valid_indices[mask_keep]
            valid_px = valid_px[mask_keep]
            valid_py = valid_py[mask_keep]
            dynamic_weights = dynamic_weights[mask_keep]
            mask_weights = mask_weights[mask_keep]
            dynamic_weights *= mask_weights

        sampled_colors = img[valid_py, valid_px]

        vertex_colors_accumulator[valid_indices] += sampled_colors * dynamic_weights[:, np.newaxis]
        weight_accumulator[valid_indices] += dynamic_weights[:, np.newaxis]

    valid_color_mask = weight_accumulator[:, 0] > 0
    missing_mask = ~valid_color_mask

    safe_weights = weight_accumulator.copy()
    safe_weights[missing_mask, 0] = 1.0

    final_colors = (vertex_colors_accumulator / safe_weights)
    final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)

    final_colors[valid_color_mask, 3] = 255

    if fill_unpainted:
        if np.any(missing_mask):
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
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "multiview_masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("high_poly_mesh",)
    FUNCTION = "project"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def project(self, high_poly_mesh, multiview_images, projection_mode, blend_sharpness, angle_cutoff, perspective_fov, 
                orthographic_width, orthographic_height, perspective_width, perspective_height, camera_config=None,
                multiview_masks=None):
        mesh = project_multiview_vertex_colors(
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
            camera_config=camera_config,
            fill_unpainted=True,
            multiview_masks=multiview_masks,
        )
        return (mesh,)


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
                "seqtex_include_poles": ("BOOLEAN", {"default": False}),
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
        seqtex_include_poles=False,
        multiview_masks=None,
    ):
        has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        if not has_uv:
            raise ValueError("Mesh has no UV map. Please use the BlenderUnwrap node first.")

        effective_camera_config = camera_config
        effective_perspective_fov = perspective_fov
        effective_orthographic_width = orthographic_width
        effective_orthographic_height = orthographic_height

        if use_seqtex_mv:
            num_views = _infer_multiview_count(multiview_images)
            seqtex_camera_config, seqtex_fov = _compute_seqtex_camera_parameters(
                mesh,
                num_views,
                camera_elevation=0.0,
                camera_lens=50.0,
                camera_sensor_width=36.0,
                azimuth_offset=seqtex_rotation_offset,
                include_poles=seqtex_include_poles,
            )
            effective_camera_config = seqtex_camera_config
            effective_perspective_fov = seqtex_fov
            effective_orthographic_width = 2.1
            effective_orthographic_height = 2.1

        textured_mesh = project_multiview_vertex_colors(
            mesh,
            multiview_images,
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
            multiview_masks=multiview_masks,
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

    def _bake_vertex_colors_to_texture(self, mesh, resolution, margin, use_gpu):
        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = os.path.join(temp_dir, "mesh.glb")
            texture_path = os.path.join(temp_dir, "color_map.png")
            script_path = os.path.join(temp_dir, "s.py")

            mesh.export(file_obj=mesh_path)

            params = {
                "mesh_path": mesh_path,
                "texture_path": texture_path,
                "resolution": int(resolution),
                "margin": int(margin),
                "use_gpu": use_gpu,
            }

            script = f'''
import bpy, sys, os, traceback
p = {{ {", ".join(f'"{k}": r"{v}"' if isinstance(v, str) else f'"{k}": {v}' for k, v in params.items())} }}

def setup_gpu():
    if not p['use_gpu']:
        bpy.context.scene.cycles.device = 'CPU'
        return
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'NONE'
        device_types = ['OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI']
        for device_type in device_types:
            if hasattr(prefs, 'get_devices_for_type'):
                devices = prefs.get_devices_for_type(device_type)
            else:
                devices = []
            if devices:
                prefs.compute_device_type = device_type
                print(f"Found and set compute device type to: {{device_type}}")
                break
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type != 'CPU'
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

def import_mesh():
    bpy.ops.import_scene.gltf(filepath=p['mesh_path'])
    bpy.context.view_layer.update()
    obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    obj.name = "BakeMesh"
    if not obj.data.color_attributes:
        raise Exception("Mesh has no vertex colors to bake.")
    if not obj.data.uv_layers:
        raise Exception("Mesh has no UV map.")
    return obj

def setup_material(obj):
    mat = bpy.data.materials.new(name="MultiviewBakeMaterial")
    mat.use_nodes = True
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    vc_layer = obj.data.color_attributes[0]
    color_output = None
    try:
        vc_node = nodes.new('ShaderNodeVertexColor')
        vc_node.layer_name = vc_layer.name
        color_output = vc_node.outputs['Color']
    except Exception:
        attr_node = nodes.new('ShaderNodeAttribute')
        attr_node.attribute_name = vc_layer.name
        attr_node.attribute_type = 'GEOMETRY'
        color_output = attr_node.outputs['Color']

    emission = nodes.new('ShaderNodeEmission')
    material_output = nodes.new('ShaderNodeOutputMaterial')
    tex_image = nodes.new('ShaderNodeTexImage')
    nodes.active = tex_image

    links.new(color_output, emission.inputs['Color'])
    links.new(emission.outputs['Emission'], material_output.inputs['Surface'])

    return tex_image

def bake_texture(obj, tex_node):
    image = bpy.data.images.new(name="MultiviewColorMap", width=p['resolution'], height=p['resolution'], alpha=False)
    tex_node.image = image

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='EMIT', margin=p['margin'], use_clear=True)
    return image

try:
    setup_scene()
    obj = import_mesh()
    tex_node = setup_material(obj)
    baked_image = bake_texture(obj, tex_node)
    baked_image.filepath_raw = p['texture_path']
    baked_image.file_format = 'PNG'
    baked_image.save()
    bpy.data.images.remove(baked_image)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''

            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            if not os.path.exists(texture_path):
                return dummy_image

            img = Image.open(texture_path).convert('RGB')
            tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
            return tensor

        return dummy_image

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
        if not hasattr(high_poly_mesh.visual, 'vertex_colors') or high_poly_mesh.visual.vertex_colors.shape[0] == 0:
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
