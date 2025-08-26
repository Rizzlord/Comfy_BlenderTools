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
    BAKE_TYPES = ["diffuse"]  # Removed "emit"

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
    
    # Switch to edit mode with proper context
    override = bpy.context.copy()
    override['object'] = obj
    override['active_object'] = obj
    override['selected_objects'] = [obj]
    override['mode'] = 'EDIT_MESH'
    override['edit_object'] = obj
    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode='EDIT')
        # Ensure mesh context for selection
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='SELECT')
        if merge_distance > 0.0:
            bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except:
            pass  # Skip if no custom split normals
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.shade_smooth()
"""

            script = f"""
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
    # No light added, direct and indirect light turned off
    bpy.context.scene.cycles.use_direct_light = False
    bpy.context.scene.cycles.use_indirect_light = False

def align_meshes(high_obj, low_obj):
    import bpy
    # Set origins to geometry center
    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True)
    bpy.context.view_layer.objects.active = high_obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    # Apply transformations
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Ensure same scale and position
    low_obj.location = high_obj.location
    low_obj.scale = high_obj.scale

def import_meshes():
    bpy.ops.import_scene.gltf(filepath=p['high_poly_path'])
    bpy.context.view_layer.update()
    high_obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    high_obj.name = "HighPoly"
    
    # Verify vertex colors exist
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

    # Ensure proper context for baking
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
"""
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
                return (final_mesh, dummy_image)

            # Apply brightness and contrast adjustments
            color_map_tensor = color_map_tensor * brightness
            color_map_tensor = (color_map_tensor - 0.5) * contrast + 0.5
            color_map_tensor = torch.clamp(color_map_tensor, 0.0, 1.0)

            final_mesh = trimesh_loader.load(final_low_poly_path, force="mesh")

            return (final_mesh, color_map_tensor)