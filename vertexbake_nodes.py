import torch
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, grey_dilation

# Helper functions from the original script
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

def create_perspective_projection(fov_y, aspect, near=0.01, far=100.0):
    projection_matrix = np.zeros((4, 4))
    f = 1.0 / np.tan(fov_y / 2.0)

    projection_matrix[0, 0] = f / aspect
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = (far + near) / (near - far)
    projection_matrix[2, 3] = (2.0 * far * near) / (near - far)
    projection_matrix[3, 2] = -1.0

    return projection_matrix

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

def get_camera_position(center, distance, azimuth_deg, elevation_deg):
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    x = center[0] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = center[1] + distance * np.sin(elevation_rad)
    z = center[2] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    
    return np.array([x, y, z])

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
    RETURN_NAMES = ("high_poly_mesh",)
    FUNCTION = "project"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def project(self, high_poly_mesh, multiview_images, projection_mode, blend_sharpness, perspective_fov, 
                orthographic_width, orthographic_height, perspective_width, perspective_height, camera_config=None):
        mesh = high_poly_mesh.copy()
        images_pil = [Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)) for img in multiview_images]

        if not images_pil:
            raise ValueError("No images provided for projection.")

        if camera_config:
            camera_azims = camera_config.get("selected_camera_azims", [0, 90, 180, 270, 0, 180])
            camera_elevs = camera_config.get("selected_camera_elevs", [10, -10, 10, -10, 90, -90])
            cam_distance = camera_config.get("camera_distance", 1.45)
            ortho_scale_mult = camera_config.get("ortho_scale", 1.2)
        else:
            camera_azims = [0, 90, 180, 270, 0, 180]
            camera_elevs = [10, -10, 10, -10, 90, -90]
            cam_distance = 1.45
            ortho_scale_mult = 1.2

        if len(images_pil) != len(camera_azims):
            raise ValueError(f"Number of images ({len(images_pil)}) does not match number of camera views ({len(camera_azims)}).")

        h, w, _ = np.array(images_pil[0]).shape
        aspect_ratio = w / h
        centroid = mesh.bounding_box.centroid
        cam_target = centroid
        cam_up = np.array([0, 1, 0]) # Common up vector

        vertex_colors_accumulator = np.zeros((len(mesh.vertices), 4))
        weight_accumulator = np.zeros((len(mesh.vertices), 1))
        
        mesh.vertex_normals

        for i, (azim, elev) in enumerate(zip(camera_azims, camera_elevs)):
            img_pil = images_pil[i]
            img = np.array(img_pil.convert('RGBA'))
            cam_pos = get_camera_position(centroid, cam_distance, azim, elev)
            view_mat = create_view_matrix(cam_pos, cam_target, cam_up)

            if projection_mode == 'orthographic':
                # Use custom width/height if provided, otherwise use bounding box calculation
                if orthographic_width > 0 and orthographic_height > 0:
                    proj_mat = create_orthographic_projection(orthographic_height, orthographic_width / orthographic_height)
                else:
                    extents = mesh.bounding_box.extents
                    max_extent = float(np.max(extents))
                    ortho_height = max_extent * ortho_scale_mult
                    proj_mat = create_orthographic_projection(ortho_height, aspect_ratio)
            else:  # perspective
                # Use custom width/height if provided, otherwise use FOV
                if perspective_width > 0 and perspective_height > 0:
                    # Calculate FOV based on width/height
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
            dot_products = np.sum(mesh.vertex_normals * view_vectors, axis=1)
            front_facing_vertex_indices = np.where(dot_products < 0)[0]
            
            valid_indices = np.intersect1d(frustum_indices, front_facing_vertex_indices, assume_unique=True)
            
            dynamic_weights = (-dot_products[valid_indices])**blend_sharpness
            
            valid_px = np.clip(px[valid_indices], 0, w - 1).astype(int)
            valid_py = np.clip(py[valid_indices], 0, h - 1).astype(int)
            
            sampled_colors = img[valid_py, valid_px]
            
            vertex_colors_accumulator[valid_indices] += sampled_colors * dynamic_weights[:, np.newaxis]
            weight_accumulator[valid_indices] += dynamic_weights[:, np.newaxis]

        valid_color_indices = np.where(weight_accumulator[:, 0] > 0)[0]
        weight_accumulator[weight_accumulator == 0] = 1.0
        
        final_colors = (vertex_colors_accumulator / weight_accumulator)
        final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
        
        final_colors[valid_color_indices, 3] = 255
        
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=final_colors)
        return (mesh,)

class VertexToLowPoly:
    BLEED_METHODS = ["distance", "dilate"]
    SUPERSAMPLE_OPTIONS = ["1x", "2x", "4x"]
    BAKING_MODES = ["Raycast (High Quality)", "Vertex Color Interpolation (Fast)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "low_poly_mesh": ("TRIMESH",),
                "texture_resolution": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "supersampling": (cls.SUPERSAMPLE_OPTIONS, {"default": "1x"}),
                "baking_mode": (cls.BAKING_MODES, {"default": "Raycast (High Quality)"}),
                "seam_bleed": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "bleed_method": (cls.BLEED_METHODS, {"default": "distance"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE",)
    RETURN_NAMES = ("low_poly_mesh", "baked_texture",)
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def bake(self, high_poly_mesh, low_poly_mesh, texture_resolution, supersampling, baking_mode, seam_bleed, bleed_method):
        if not hasattr(high_poly_mesh.visual, 'vertex_colors') or high_poly_mesh.visual.vertex_colors.shape[0] == 0:
            raise ValueError("High-poly mesh does not have vertex colors to bake.")
        
        if not hasattr(low_poly_mesh.visual, 'uv') or low_poly_mesh.visual.uv is None:
                raise ValueError("Low-poly mesh must have UV coordinates. Use an unwrap node first.")

        low_poly = low_poly_mesh.copy()
        
        ss_factor = int(supersampling.replace('x', ''))
        render_resolution = texture_resolution * ss_factor
        texture_map = np.zeros((render_resolution, render_resolution, 4), dtype=np.uint8)

        kdtree = cKDTree(high_poly_mesh.vertices)
        _, nearest_indices = kdtree.query(low_poly.vertices)
        low_poly_vertex_colors = high_poly_mesh.visual.vertex_colors[nearest_indices]

        if baking_mode == "Raycast (High Quality)":
            intersector = high_poly_mesh.ray
            low_poly.face_normals
            low_poly.vertex_normals
        
        uvs = low_poly.visual.uv
        faces = low_poly.faces
        vertices = low_poly.vertices
        vertex_normals = low_poly.vertex_normals

        for face_idx, face in enumerate(faces):
            v0_idx, v1_idx, v2_idx = face
            
            uv0, uv1, uv2 = uvs[[v0_idx, v1_idx, v2_idx]] * (render_resolution - 1)
            v0, v1, v2 = vertices[[v0_idx, v1_idx, v2_idx]]
            
            min_x = int(max(0, min(uv0[0], uv1[0], uv2[0])))
            max_x = int(min(render_resolution - 1, max(uv0[0], uv1[0], uv2[0])))
            min_y = int(max(0, min(uv0[1], uv1[1], uv2[1])))
            max_y = int(min(render_resolution - 1, max(uv0[1], uv1[1], uv2[1])))
            
            if min_x >= max_x or min_y >= max_y:
                continue
            
            uv_v0, uv_v1 = uv1 - uv0, uv2 - uv0
            den = uv_v0[0] * uv_v1[1] - uv_v1[0] * uv_v0[1]
            if abs(den) < 1e-6: continue

            x_range = np.arange(min_x, max_x + 1)
            y_range = np.arange(min_y, max_y + 1)
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            p = np.stack((grid_x, grid_y), axis=-1)

            uv_p = p - uv0
            w1 = (uv_p[..., 0] * uv_v1[1] - uv_v1[0] * uv_p[..., 1]) / den
            w2 = (uv_v0[0] * uv_p[..., 1] - uv_p[..., 0] * uv_v0[1]) / den
            w0 = 1.0 - w1 - w2

            mask = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not mask.any(): continue

            w0_masked, w1_masked, w2_masked = w0[mask], w1[mask], w2[mask]

            if baking_mode == "Raycast (High Quality)":
                n0, n1, n2 = vertex_normals[[v0_idx, v1_idx, v2_idx]]
                origins = w0_masked[:, None] * v0 + w1_masked[:, None] * v1 + w2_masked[:, None] * v2
                normals = w0_masked[:, None] * n0 + w1_masked[:, None] * n1 + w2_masked[:, None] * n2
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                
                origins_offset = origins + normals * 1e-4
                
                hit_locs, index_ray, hit_faces = intersector.intersects_location(origins_offset, normals)
                
                if len(hit_locs) == 0: continue

                bary = trimesh.triangles.points_to_barycentric(
                    triangles=high_poly_mesh.triangles[hit_faces],
                    points=hit_locs
                )
                hp_face_verts = high_poly_mesh.faces[hit_faces]
                hp_colors = high_poly_mesh.visual.vertex_colors[hp_face_verts]
                
                final_colors = np.einsum('ij,ijk->ik', bary, hp_colors)
                pixel_y = render_resolution - 1 - grid_y[mask]
                pixel_x = grid_x[mask]
                
                texture_map[pixel_y[index_ray], pixel_x[index_ray]] = np.clip(final_colors, 0, 255).astype(np.uint8)

            else:
                c0, c1, c2 = low_poly_vertex_colors[[v0_idx, v1_idx, v2_idx]]
                final_colors = w0_masked[:, None] * c0 + w1_masked[:, None] * c1 + w2_masked[:, None] * c2
                pixel_y = render_resolution - 1 - grid_y[mask]
                pixel_x = grid_x[mask]
                texture_map[pixel_y, pixel_x] = np.clip(final_colors, 0, 255).astype(np.uint8)


        if seam_bleed > 0:
            bleed_pixels = seam_bleed * ss_factor
            alpha_channel = texture_map[:, :, 3]
            filled_mask = alpha_channel > 0

            if bleed_method == "distance":
                mask_to_fill = alpha_channel == 0
                distances, indices = distance_transform_edt(mask_to_fill, return_indices=True)
                bleed_mask = (distances > 0) & (distances <= bleed_pixels)
                fill_coords_y, fill_coords_x = np.where(bleed_mask)
                src_coords_y = indices[0, fill_coords_y, fill_coords_x]
                src_coords_x = indices[1, fill_coords_y, fill_coords_x]
                texture_map[fill_coords_y, fill_coords_x] = texture_map[src_coords_y, src_coords_x]
            else:
                dilation_footprint = np.ones((3, 3), dtype=bool)
                for _ in range(bleed_pixels):
                    for i in range(3):
                        channel = texture_map[:, :, i]
                        dilated_channel = grey_dilation(channel, footprint=dilation_footprint)
                        channel[~filled_mask] = dilated_channel[~filled_mask]
                    filled_mask = grey_dilation(filled_mask, footprint=dilation_footprint)
                texture_map[:, :, 3][filled_mask] = 255

        baked_texture_pil = Image.fromarray(texture_map, 'RGBA')

        if ss_factor > 1:
            baked_texture_pil = baked_texture_pil.resize(
                (texture_resolution, texture_resolution), 
                Image.Resampling.LANCZOS
            )
        
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=baked_texture_pil)
        visuals = trimesh.visual.texture.TextureVisuals(uv=low_poly.visual.uv, material=material)
        visuals.vertex_colors = low_poly_vertex_colors
        low_poly.visual = visuals
        
        baked_texture_tensor = torch.from_numpy(np.array(baked_texture_pil).astype(np.float32) / 255.0)[None,]

        return (low_poly, baked_texture_tensor,)