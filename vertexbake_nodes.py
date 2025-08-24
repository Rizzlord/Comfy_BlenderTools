import torch
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

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
                "perspective_fov": ("FLOAT", {"default": 49.13, "min": 1.0, "max": 120.0, "step": 0.1}),
                "orthographic_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "orthographic_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "perspective_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "perspective_height": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "low_poly_mesh": ("TRIMESH",),
                "texture_resolution": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE",)
    RETURN_NAMES = ("low_poly_mesh", "baked_texture",)
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def bake(self, high_poly_mesh, low_poly_mesh, texture_resolution):
        if not hasattr(high_poly_mesh.visual, 'vertex_colors'):
            raise ValueError("High-poly mesh does not have vertex colors to bake.")
        
        if not hasattr(low_poly_mesh.visual, 'uv') or low_poly_mesh.visual.uv is None:
             raise ValueError("Low-poly mesh must have UV coordinates. Use an unwrap node first.")

        low_poly = low_poly_mesh.copy()
        
        # Create a KD-Tree for efficient nearest neighbor search on high-poly vertices
        kdtree = cKDTree(high_poly_mesh.vertices)
        
        # Get the barycentric coordinates of the low-poly UVs
        uvs = low_poly.visual.uv
        
        # Create an empty texture map
        texture_map = np.zeros((texture_resolution, texture_resolution, 4), dtype=np.uint8)
        
        # Iterate through each pixel of the texture map
        for y in range(texture_resolution):
            for x in range(texture_resolution):
                uv_point = np.array([x / (texture_resolution - 1), 1.0 - y / (texture_resolution - 1)])
                
                # Find the closest point on the low-poly mesh surface for this UV coordinate
                # This is a simplification; a more robust method would use raycasting from the UV map
                # For now, we find the closest UV vertex and use its corresponding 3D vertex
                distances = np.linalg.norm(uvs - uv_point, axis=1)
                closest_uv_idx = np.argmin(distances)
                
                # Get the 3D position of the closest vertex on the low-poly mesh
                low_poly_vertex = low_poly.vertices[closest_uv_idx]
                
                # Find the nearest vertex on the high-poly mesh
                _, nearest_high_poly_idx = kdtree.query(low_poly_vertex)
                
                # Get the color from that high-poly vertex
                color = high_poly_mesh.visual.vertex_colors[nearest_high_poly_idx]
                texture_map[y, x] = color

        # Convert the numpy array to a PIL Image and apply it as a texture
        baked_texture_pil = Image.fromarray(texture_map, 'RGBA')
        
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=baked_texture_pil)
        low_poly.visual = trimesh.visual.texture.TextureVisuals(uv=low_poly.visual.uv, material=material)
        
        # Convert PIL image to tensor for output
        baked_texture_tensor = torch.from_numpy(np.array(baked_texture_pil).astype(np.float32) / 255.0)[None,]

        return (low_poly, baked_texture_tensor,)
