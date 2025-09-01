import os
import subprocess
import sys
import tempfile
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

def get_blender_path():
    blender_path = os.environ.get("BLENDER_EXE")
    
    if blender_path and os.path.isfile(blender_path):
        print(f"INFO: Found Blender executable via BLENDER_EXE environment variable: {blender_path}")
        return blender_path
    
    if blender_path:
        print(f"WARNING: BLENDER_EXE environment variable was found, but the path '{blender_path}' is not a valid file.")
    else:
        print(f"INFO: BLENDER_EXE environment variable not set. Falling back to default path.")

    fallback_path = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
    
    if not os.path.isfile(fallback_path):
        raise FileNotFoundError(
            f"Blender executable not found at the default path: {fallback_path}. "
            "Please set the BLENDER_EXE environment variable to the correct path of your blender.exe."
        )
    
    print(f"INFO: Using fallback Blender executable path: {fallback_path}")
    return fallback_path

def get_blender_clean_mesh_func_script():
    return """
def clean_mesh(obj, merge_distance):
    import bpy
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if merge_distance > 0.0:
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)

    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.shade_smooth()
"""

def get_mof_path():
    """
    Finds the path to the Ministry of Flat executable.
    Searches for the MOF_EXE environment variable.
    """
    mof_path = os.environ.get("MOF_EXE")
    if mof_path and os.path.isfile(mof_path):
        print(f"INFO: Found Ministry of Flat executable via MOF_EXE: {mof_path}")
        return mof_path
    
    return None

def _run_mof_command(command):
    """
    Executes a command-line process for Ministry of Flat.
    """
    try:
        result = subprocess.run(
            command,
            check=True, capture_output=True, text=True
        )
        if result.stdout:
            print(f"Ministry of Flat stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Ministry of Flat execution failed with return code {e.returncode}.\n"
            f"--- Stderr ---\n{e.stderr}\n"
            f"--- Stdout ---\n{e.stdout}"
        )
        raise RuntimeError(error_message)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ministry of Flat executable not found. Please set the MOF_EXE environment variable.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while running Ministry of Flat: {e}")
        
def _run_blender_script(script_path):
    blender_exe = get_blender_path()
    try:
        result = subprocess.run(
            [blender_exe, '--factory-startup', '--background', '--python', script_path],
            check=True, capture_output=True, text=True
        )
        if result.stdout:
            print(f"Blender script stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Blender execution failed with return code {e.returncode}.\n"
            f"--- Stderr ---\n{e.stderr}\n"
            f"--- Stdout ---\n{e.stdout}"
        )
        raise RuntimeError(error_message)
    except FileNotFoundError:
        raise FileNotFoundError(f"Blender executable not found at '{blender_exe}'. Please set the BLENDER_EXE environment variable.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while running Blender: {e}")

class Voxelize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "mode": (["Voxel", "Blocks", "Smooth", "Sharp"], {"default": "Voxel"}),
                "smooth_shading": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Voxel_Settings": ("GROUP",),
                "Other_Modes_Settings": ("GROUP",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "remesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def remesh(self, trimesh, mode, smooth_shading, Voxel_Settings=None, Other_Modes_Settings=None):
        defaults = {
            'voxel_size': 0.0125, 'adaptivity': 0.0, 'octree_depth': 4,
            'scale': 0.9, 'remove_disconnected': True, 
            'disconnected_threshold': 1.0, 'sharpness': 1.0
        }

        params = defaults.copy()
        if Voxel_Settings:
            params.update(Voxel_Settings)
        if Other_Modes_Settings:
            params.update(Other_Modes_Settings)
        
        params.update({
            'mode': mode,
            'smooth_shading': smooth_shading
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            script_params = {k: repr(v) for k, v in params.items()}
            script_params['i'] = repr(input_mesh_path)
            script_params['o'] = repr(output_mesh_path)

            script = f"""
import bpy, sys, traceback
p = {{ {', '.join(f'\"{k}\": {v}' for k, v in script_params.items())} }}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p['i'])
    mesh_obj = next((obj for obj in bpy.context.scene.objects if obj.type == 'MESH'), None)
    if mesh_obj is None: raise Exception("No mesh found in imported file.")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    mod = mesh_obj.modifiers.new(name="Remesh", type='REMESH')
    mod.mode = p['mode'].upper()
    
    if mod.mode == 'VOXEL':
        mod.voxel_size = p['voxel_size']
        mod.adaptivity = p['adaptivity']
    else:
        mod.octree_depth = p['octree_depth']
        mod.scale = p['scale']
        mod.use_remove_disconnected = p['remove_disconnected']
        if p['remove_disconnected']:
            mod.threshold = p['disconnected_threshold']
        if mod.mode == 'SHARP':
            mod.sharpness = p['sharpness']

    bpy.ops.object.modifier_apply(modifier=mod.name)
    
    if p['smooth_shading']:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()

    bpy.ops.export_scene.gltf(filepath=p['o'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e: 
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)
            
            _run_blender_script(script_path)
            
            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh")
            return (processed_mesh,)

class VoxelSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "voxel_size": ("FLOAT", {"default": 0.0100, "min": 0.0010, "max": 1.0, "step": 0.0010, "display": "number"}),
            "adaptivity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
        }}
    RETURN_TYPES = ("GROUP",)
    FUNCTION = "get_settings"
    CATEGORY = "Comfy_BlenderTools/Utils/Settings"
    def get_settings(self, **kwargs): return (kwargs,)

class OtherModesSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "octree_depth": ("INT", {"default": 6, "min": 2, "max": 10, "step": 2}),
            "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01, "display": "number"}),
            "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
            "remove_disconnected": ("BOOLEAN", {"default": True}),
            "disconnected_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
        }}
    RETURN_TYPES = ("GROUP",)
    FUNCTION = "get_settings"
    CATEGORY = "Comfy_BlenderTools/Utils/Settings"
    def get_settings(self, **kwargs): return (kwargs,)

class TextureToHeight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "median_filter": ("BOOLEAN", {"default": False}),
                "normalize": ("BOOLEAN", {"default": True}),
                "generate_normal_map": ("BOOLEAN", {"default": False}),
                "normal_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("height_map", "normal_map")
    FUNCTION = "convert"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def convert(self, image, scale, contrast, brightness, blur_radius, median_filter, normalize, generate_normal_map, normal_strength):
        if image.dim() == 4 and image.shape[0] == 1:
            img_tensor = image[0]
        else:
            img_tensor = image

        i = 255. * img_tensor.cpu().numpy()
        pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        original_width, original_height = pil_img.size
        
        if scale != 1.0:
            tile_width = int(original_width / scale)
            tile_height = int(original_height / scale)

            if tile_width > 0 and tile_height > 0:
                tile = pil_img.resize((tile_width, tile_height), Image.LANCZOS)
                tiled_image = Image.new(pil_img.mode, (original_width, original_height))
                
                for y in range(0, original_height, tile_height):
                    for x in range(0, original_width, tile_width):
                        tiled_image.paste(tile, (x, y))
                
                pil_img = tiled_image

        processed_img = pil_img.convert('L')

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(processed_img)
            processed_img = enhancer.enhance(contrast)
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(processed_img)
            processed_img = enhancer.enhance(brightness)
            
        if blur_radius > 0.0:
            processed_img = processed_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        if median_filter:
            processed_img = processed_img.filter(ImageFilter.MedianFilter(size=3))

        height_map_np = np.array(processed_img).astype(np.float32)

        if normalize:
            min_val, max_val = np.min(height_map_np), np.max(height_map_np)
            if max_val > min_val:
                height_map_np = (height_map_np - min_val) / (max_val - min_val)
            else:
                height_map_np = np.zeros_like(height_map_np)
        else:
            height_map_np /= 255.0

        height_map_tensor = torch.from_numpy(height_map_np).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        normal_map_tensor = torch.zeros_like(height_map_tensor)

        if generate_normal_map:
            with tempfile.TemporaryDirectory() as temp_dir:
                source_image_path = os.path.join(temp_dir, "source.png")
                output_normal_path = os.path.join(temp_dir, "normal.png")
                script_path = os.path.join(temp_dir, "s.py")

                pil_img.save(source_image_path)

                params = {
                    'strength': normal_strength,
                }
                script_params = {k: repr(v) for k, v in params.items()}
                script_params['image_path'] = repr(source_image_path)
                script_params['output_path'] = repr(output_normal_path)

                script = f"""
import bpy, sys, os, traceback
p = {{ {', '.join(f'\"{k}\": {v}' for k, v in script_params.items())} }}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object

    mat = bpy.data.materials.new(name="NormalBakeMaterial")
    mat.use_nodes = True
    plane.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    if not bsdf:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        links.new(bsdf.outputs['BSDF'], nodes['Material Output'].inputs['Surface'])

    source_tex_node = nodes.new('ShaderNodeTexImage')
    try:
        source_tex_node.image = bpy.data.images.load(p['image_path'])
    except Exception as e:
        raise Exception(f"Failed to load source image: {{e}}")

    bump_node = nodes.new('ShaderNodeBump')
    bump_node.inputs['Strength'].default_value = p['strength']
    links.new(source_tex_node.outputs['Color'], bump_node.inputs['Height'])
    links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])

    res_x, res_y = source_tex_node.image.size
    bake_image = bpy.data.images.new(name="NormalBakeImage", width=res_x, height=res_y, alpha=False)

    bake_tex_node = nodes.new('ShaderNodeTexImage')
    bake_tex_node.image = bake_image
    nodes.active = bake_tex_node

    bpy.context.scene.cycles.samples = 1
    bpy.ops.object.bake(type='NORMAL', use_selected_to_active=False, margin=0)

    bake_image.filepath_raw = p['output_path']
    bake_image.file_format = 'PNG'
    bake_image.save()
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
                with open(script_path, 'w') as f:
                    f.write(script)
                
                _run_blender_script(script_path)

                if os.path.exists(output_normal_path):
                    normal_img_pil = Image.open(output_normal_path).convert('RGB')
                    normal_img_np = np.array(normal_img_pil).astype(np.float32) / 255.0
                    normal_map_tensor = torch.from_numpy(normal_img_np).unsqueeze(0)

        return (height_map_tensor, normal_map_tensor)

class DisplaceMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "strength": ("FLOAT", {"default": 0.005, "min": -10.0, "max": 10.0, "step": 0.001}),
                "midlevel": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
                "uv_space": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "displacement_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "displace"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def displace(self, trimesh, strength, midlevel, merge_distance, uv_space, displacement_map=None):
        if uv_space and (not hasattr(trimesh.visual, 'uv') or len(trimesh.visual.uv) == 0):
            raise Exception("Input mesh must have UV coordinates for displacement. Use BlenderUnwrap first.")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            image_path = os.path.join(temp_dir, "displace.png")
            script_path = os.path.join(temp_dir, "s.py")

            trimesh.export(file_obj=input_mesh_path)

            disp_map_tensor = displacement_map
            if disp_map_tensor is None:
                disp_map_tensor = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

            i = 255. * disp_map_tensor[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(image_path)
            
            params = {
                'strength': strength,
                'midlevel': midlevel,
                'merge_distance': merge_distance,
                'uv_space': uv_space,
            }
            script_params = {k: repr(v) for k, v in params.items()}
            script_params['input_mesh'] = repr(input_mesh_path)
            script_params['output_mesh'] = repr(output_mesh_path)
            script_params['image_path'] = repr(image_path)
            
            clean_mesh_func_script = get_blender_clean_mesh_func_script()

            script = f"""
{clean_mesh_func_script}
import bpy, sys, os, traceback
p = {{ {', '.join(f'\"{k}\": {v}' for k, v in script_params.items())} }}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p['input_mesh'])
    
    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    if not obj:
        raise Exception("No mesh found in the imported GLB file.")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    clean_mesh(obj, p['merge_distance'])

    disp_tex = bpy.data.textures.new('DisplaceTexture', type='IMAGE')
    try:
        disp_img = bpy.data.images.load(p['image_path'])
        disp_tex.image = disp_img
    except Exception as e:
        raise Exception(f"Failed to load displacement image: {{e}}")

    disp_mod = obj.modifiers.new(name="DisplaceMod", type='DISPLACE')
    disp_mod.texture = disp_tex
    
    if p['uv_space']:
        disp_mod.texture_coords = 'UV'
        if not obj.data.uv_layers:
            raise Exception("Mesh does not have a UV map. Cannot use UV coordinates for displacement.")
    else:
        disp_mod.texture_coords = 'LOCAL'

    disp_mod.strength = p['strength']
    disp_mod.mid_level = p['midlevel']
    
    bpy.ops.object.modifier_apply(modifier=disp_mod.name)

    bpy.ops.export_scene.gltf(filepath=p['output_mesh'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh")
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                processed_mesh.visual.material = trimesh.visual.material

            return (processed_mesh,)

class SmoothMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "factor": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 30}),
                "axis_x": ("BOOLEAN", {"default": True}),
                "axis_y": ("BOOLEAN", {"default": True}),
                "axis_z": ("BOOLEAN", {"default": True}),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "smooth"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def smooth(self, trimesh, factor, repeat, axis_x, axis_y, axis_z, merge_distance):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            params = {
                'factor': factor,
                'repeat': repeat,
                'axis_x': axis_x,
                'axis_y': axis_y,
                'axis_z': axis_z,
                'merge_distance': merge_distance,
            }
            script_params = {k: repr(v) for k, v in params.items()}
            script_params['input_mesh'] = repr(input_mesh_path)
            script_params['output_mesh'] = repr(output_mesh_path)
            
            clean_mesh_func_script = get_blender_clean_mesh_func_script()

            script = f"""
{clean_mesh_func_script}
import bpy, sys, os, traceback
p = {{ {', '.join(f'\"{k}\": {v}' for k, v in script_params.items())} }}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p['input_mesh'])
    
    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    if not obj:
        raise Exception("No mesh found in the imported GLB file.")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    clean_mesh(obj, p['merge_distance'])

    smooth_mod = obj.modifiers.new(name="SmoothMod", type='SMOOTH')
    smooth_mod.factor = p['factor']
    smooth_mod.iterations = p['repeat']
    smooth_mod.use_x = p['axis_x']
    smooth_mod.use_y = p['axis_y']
    smooth_mod.use_z = p['axis_z']
    
    bpy.ops.object.modifier_apply(modifier=smooth_mod.name)

    bpy.ops.export_scene.gltf(filepath=p['output_mesh'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh")
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                processed_mesh.visual.material = trimesh.visual.material

            return (processed_mesh,)

class ProcessMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
                "keep_biggest": ("BOOLEAN", {"default": True}),
                "fill_holes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "process"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def process(self, trimesh, merge_distance, keep_biggest, fill_holes):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            params = {
                'merge_distance': merge_distance,
                'keep_biggest': keep_biggest,
                'fill_holes': fill_holes,
            }
            script_params = {k: repr(v) for k, v in params.items()}
            script_params['input_mesh'] = repr(input_mesh_path)
            script_params['output_mesh'] = repr(output_mesh_path)

            script = f"""
import bpy, sys, os, traceback
p = {{ {', '.join(f'\"{k}\": {v}' for k, v in script_params.items())} }}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p['input_mesh'])
    
    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    if not obj:
        raise Exception("No mesh found in the imported GLB file.")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    if p['merge_distance'] > 0.0:
        bpy.ops.mesh.remove_doubles(threshold=p['merge_distance'])

    if p['keep_biggest']:
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        mesh_objects = [o for o in bpy.context.selected_objects if o.type == 'MESH' and o.data]

        if len(mesh_objects) > 1:
            biggest = max(mesh_objects, key=lambda o: len(o.data.vertices))
            
            for o in mesh_objects:
                if o != biggest:
                    bpy.data.objects.remove(o, do_unlink=True)
            
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = biggest
            biggest.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    if p['fill_holes']:
        bpy.ops.mesh.fill_holes(sides=0)

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.export_scene.gltf(filepath=p['output_mesh'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh")
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                processed_mesh.visual.material = trimesh.visual.material

            return (processed_mesh,)
