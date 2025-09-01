import os
import sys
import tempfile
import traceback
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image
from .utils import _run_blender_script, get_blender_clean_mesh_func_script

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
                "margin": ("INT", {"default": 64, "min": 0, "max": 64}),
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
                final_mesh = trimesh_loader.load(final_low_poly_path, force="mesh")
                return (final_mesh, dummy_image)

            color_map_tensor = color_map_tensor * brightness
            color_map_tensor = (color_map_tensor - 0.5) * contrast + 0.5
            color_map_tensor = torch.clamp(color_map_tensor, 0.0, 1.0)

            final_mesh = trimesh_loader.load(final_low_poly_path, force="mesh")

            return (final_mesh, color_map_tensor)

class DiffuseHighpolyCol:
    PROJECTION_MODES = ["orthographic", "perspective"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "projection_mode": (cls.PROJECTION_MODES, {"default": "orthographic"}),
                "blend_sharpness": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "angle_cutoff": ("FLOAT", {"default": 75.0, "min": 0.0, "max": 90.0, "step": 0.5}),
                "perspective_fov": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "orthographic_width": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "orthographic_height": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "perspective_width": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "perspective_height": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("high_poly_mesh",)
    FUNCTION = "project"
    CATEGORY = "Comfy_BlenderTools/VertexBake"

    def project(self, high_poly_mesh, multiview_images, projection_mode, blend_sharpness, angle_cutoff, perspective_fov,
                orthographic_width, orthographic_height, perspective_width, perspective_height, camera_config=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            high_poly_path = os.path.join(temp_dir, "high.glb")
            final_high_poly_path = os.path.join(temp_dir, "final_high.glb")
            script_path = os.path.join(temp_dir, "s.py")

            image_paths = []
            for i, img_tensor in enumerate(multiview_images):
                img_np = np.clip(img_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                if img_np.shape[0] == 1:
                    img_np = img_np.squeeze(0)
                img = Image.fromarray(img_np)
                path = os.path.join(temp_dir, f"view_{i}.png")
                img.save(path)
                image_paths.append(path)

            params = {
                'high_poly_path': high_poly_path,
                'final_high_poly_path': final_high_poly_path,
                'image_paths': image_paths,
                'projection_mode': projection_mode,
                'blend_sharpness': blend_sharpness,
                'angle_cutoff': angle_cutoff,
                'perspective_fov': perspective_fov,
                'orthographic_width': orthographic_width,
                'orthographic_height': orthographic_height,
                'perspective_width': perspective_width,
                'perspective_height': perspective_height,
            }

            if camera_config:
                params['camera_azims'] = camera_config.get("selected_camera_azims", [0, 90, 180, 270, 0, 180])
                params['camera_elevs'] = camera_config.get("selected_camera_elevs", [10, -10, 10, -10, 90, -90])
                params['cam_distance'] = camera_config.get("camera_distance", 1.45)
                params['ortho_scale_mult'] = camera_config.get("ortho_scale", 1.2)
            else:
                params['camera_azims'] = [0, 90, 180, 270, 0, 180]
                params['camera_elevs'] = [10, -10, 10, -10, 90, -90]
                params['cam_distance'] = 1.45
                params['ortho_scale_mult'] = 1.2

            high_poly_mesh.export(file_obj=high_poly_path)

            clean_mesh_func_script = get_blender_clean_mesh_func_script()

            script = f'''
{clean_mesh_func_script}
import bpy
import numpy as np
import sys, os, traceback

p = {{ {", ".join(f'"{k}": {v!r}' for k, v in params.items())} }}

def get_camera_position(center, distance, azimuth_deg, elevation_deg):
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    x = center[0] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = center[1] + distance * np.sin(elevation_rad)
    z = center[2] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    return np.array([x, y, z])

def create_view_matrix(position, target, up):
    f = target - position
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    view_matrix = np.eye(4)
    view_matrix[0, :3] = s
    view_matrix[1, :3] = u
    view_matrix[2, :3] = -f
    view_matrix[0, 3] = -np.dot(s, position)
    view_matrix[1, 3] = -np.dot(u, position)
    view_matrix[2, 3] = -np.dot(-f, position)
    return view_matrix

def create_orthographic_projection(height, aspect, near=-1000.0, far=1000.0):
    top = height / 2.0
    bottom = -top
    right = (height * aspect) / 2.0
    left = -right
    proj = np.zeros((4, 4))
    proj[0, 0] = 2.0 / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[2, 2] = -2.0 / (far - near)
    proj[3, 3] = 1.0
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    return proj

def create_perspective_projection(fovy, aspect, near=0.01, far=1000.0):
    f = 1.0 / np.tan(fovy / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2.0 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p['high_poly_path'])
    bpy.context.view_layer.update()
    high_obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    high_obj.name = "HighPoly"
    clean_mesh(high_obj, 0.0001)
    
    bpy.context.view_layer.objects.active = high_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    mesh_data = high_obj.data
    vertices = np.array([list(v.co) for v in mesh_data.vertices])
    normals = np.array([list(v.normal) for v in mesh_data.vertices])
    
    print(f"Normals shape: {{normals.shape}}")
    print(f"Normals min/max: {{np.min(normals)}}, {{np.max(normals)}}")
    
    centroid = np.mean(vertices, axis=0)
    radius = np.max(np.linalg.norm(vertices - centroid, axis=1)) + 1e-6
    if p['projection_mode'] == 'orthographic':
        p['cam_distance'] = radius * 100
    else:
        if not (p['perspective_width'] > 0 and p['perspective_height'] > 0):
            fovy_rad = np.radians(p['perspective_fov']) / 2
            p['cam_distance'] = radius / np.tan(fovy_rad) + radius * 0.1
    print(f"Computed cam_distance: {{p['cam_distance']}}")

    vertex_colors_accumulator = np.zeros((vertices.shape[0], 4))
    weight_accumulator = np.zeros((vertices.shape[0], ))
    camera_azims = p['camera_azims']
    camera_elevs = p['camera_elevs']
    cam_distance = p['cam_distance']
    ortho_scale_mult = p['ortho_scale_mult']
    image_paths = p['image_paths']

    for i in range(len(image_paths)):
        azim = camera_azims[i]
        elev = camera_elevs[i]
        image_path = image_paths[i]
        image = bpy.data.images.load(image_path)
        w, h = image.size
        aspect_ratio = w / h
        img_array = np.array(image.pixels).reshape((h, w, 4)) * 255.0
        print(f"Image {{i}} loaded, shape: {{img_array.shape}}, min/max: {{img_array.min()}}, {{img_array.max()}}")
        
        cam_pos = get_camera_position(centroid, cam_distance, azim, elev)
        view_mat = create_view_matrix(cam_pos, centroid, np.array([0.0, 1.0, 0.0]))
        
        if p['projection_mode'] == 'orthographic':
            if p['orthographic_width'] > 0 and p['orthographic_height'] > 0:
                height = p['orthographic_height']
                aspect = p['orthographic_width'] / height
            else:
                extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
                max_extent = float(np.max(extents))
                height = max_extent * ortho_scale_mult
                aspect = aspect_ratio
            proj_mat = create_orthographic_projection(height, aspect)
        else:
            if p['perspective_width'] > 0 and p['perspective_height'] > 0:
                fovy = 2 * np.arctan(p['perspective_height'] / 2 / cam_distance)
                aspect = p['perspective_width'] / p['perspective_height']
            else:
                fovy = np.radians(p['perspective_fov'])
                aspect = aspect_ratio
            proj_mat = create_perspective_projection(fovy, aspect)
        
        pvm_matrix = proj_mat @ view_mat
        verts_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        clip_coords = verts_homogeneous @ pvm_matrix.T
        
        w_coords = clip_coords[:, 3].copy()
        positive_w = w_coords > 1e-6
        w_coords[~positive_w] = 1e-6
        ndc = clip_coords[:, :3] / w_coords[:, None]
        
        px = (ndc[:, 0] * 0.5 + 0.5) * (w - 1)
        py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)
        
        frustum_indices = np.where(positive_w & (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1))[0]
        
        view_vectors = vertices - cam_pos
        view_vectors /= np.linalg.norm(view_vectors, axis=1)[:, None]
        dot_products = np.sum(normals * view_vectors, axis=1)
        print(f"View {{i}}: Dot products min/max: {{np.min(dot_products)}}, {{np.max(dot_products)}}")
        
        front_facing_vertex_indices = np.where(dot_products < 0)[0]
        print(f"View {{i}}: Front-facing vertices: {{len(front_facing_vertex_indices)}}")
        
        valid_indices = np.intersect1d(frustum_indices, front_facing_vertex_indices)
        print(f"View {{i}}: {{len(valid_indices)}} valid vertices")
        if len(valid_indices) == 0:
            continue
        
        cos = -dot_products[valid_indices]
        cos_thres = np.cos(np.radians(p['angle_cutoff']))
        cos[cos < cos_thres] = 0.0
        dynamic_weights = np.power(cos, p['blend_sharpness'])
        
        keep = dynamic_weights > 0
        valid_indices = valid_indices[keep]
        dynamic_weights = dynamic_weights[keep]
        
        if len(valid_indices) == 0:
            print(f"No vertices after weight filtering for view {{i}}")
            continue
        
        print(f"View {{i}} weights range: {{dynamic_weights.min()}} to {{dynamic_weights.max()}}")
        valid_px = np.clip(px[valid_indices], 0, w - 1).astype(int)
        valid_py = np.clip(py[valid_indices], 0, h - 1).astype(int)
        sampled_colors = img_array[valid_py, valid_py]
        
        vertex_colors_accumulator[valid_indices] += sampled_colors * dynamic_weights[:, None]
        weight_accumulator[valid_indices] += dynamic_weights

    valid_color_indices = np.where(weight_accumulator > 0)[0]
    weight_accumulator[weight_accumulator == 0] = 1.0
    final_colors = vertex_colors_accumulator / weight_accumulator[:, None]
    final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
    final_colors[valid_color_indices, 3] = 255
    
    print(f"Final colors shape: {{final_colors.shape}}, non-zero colors: {{np.sum(final_colors[:, :3] > 0)}}")
    
    color_attr = mesh_data.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
    colors_flat = (final_colors.astype(np.float32) / 255.0).ravel()
    color_attr.data.foreach_set("color", colors_flat)
    
    for i, color in enumerate(color_attr.data):
        if i < 5:
            print(f"Vertex {{i}} color: {{list(color.color)}}")
    
    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=p['final_high_poly_path'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            if not os.path.exists(final_high_poly_path):
                raise ValueError("Failed to generate projected high-poly mesh in Blender.")

            final_mesh = trimesh_loader.load(final_high_poly_path, force="mesh")
            if hasattr(final_mesh.visual, 'vertex_colors') and final_mesh.visual.vertex_colors.shape[0] > 0:
                print(f"Loaded mesh with {{final_mesh.visual.vertex_colors.shape[0]}} vertex colors")
            else:
                print("Warning: No vertex colors found in loaded mesh")
            return (final_mesh,)