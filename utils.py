import os
import subprocess
import sys
import tempfile
import shutil
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
import uuid
from PIL import Image, ImageEnhance, ImageFilter
import pymeshlab

def get_blender_path():
    blender_path = os.environ.get("BLENDER_EXE")
    
    if blender_path and os.path.isfile(blender_path):
        print(f"INFO: Found Blender executable via BLENDER_EXE environment variable: {blender_path}")
        return blender_path
    
    if blender_path:
        print(f"WARNING: BLENDER_EXE environment variable was found, but the path '{blender_path}' is not a valid file.")
    else:
        print(f"INFO: BLENDER_EXE environment variable not set. Falling back to other detection methods.")

    # Try to find blender in PATH
    blender_path = shutil.which("blender")
    if blender_path:
        print(f"INFO: Found Blender executable in system PATH: {blender_path}")
        return blender_path
        
    # Check common Linux paths
    if sys.platform.startswith("linux"):
        common_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender"
        ]
        for p in common_paths:
            if os.path.isfile(p):
                print(f"INFO: Found Blender executable in common path: {p}")
                return p

    # Default fallback for Windows (or if nothing else works)
    fallback_path = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
    
    if os.path.isfile(fallback_path):
        print(f"INFO: Using fallback Blender executable path: {fallback_path}")
        return fallback_path
        
    raise FileNotFoundError(
        "Blender executable not found. "
        "Please set the BLENDER_EXE environment variable to the correct path of your blender.exe, "
        "or ensure 'blender' is in your system PATH."
    )

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

    bpy.ops.object.mode_set(mode='OBJECT')
"""

def get_mof_path():
    # Check for MOF_EXE environment variable
    mof_path = os.environ.get("MOF_EXE")
    if mof_path and os.path.isfile(mof_path):
        print(f"INFO: Found Ministry of Flat executable via MOF_EXE environment variable: {mof_path}")
        return mof_path

    # Check for UnwrapConsole3.exe in the custom node folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple common casings (Linux is case-sensitive)
    for filename in ["UnWrapConsole3.exe", "UnwrapConsole3.exe"]:
        local_path = os.path.join(current_dir, filename)
        if os.path.isfile(local_path):
            print(f"INFO: Found Ministry of Flat executable in custom node folder: {local_path}")
            return local_path
    
    return None

def _run_mof_command(command, cwd=None):
    # Handle Wine execution for .exe on Linux
    if sys.platform.startswith("linux") and command[0].lower().endswith(".exe"):
        # Check if wine is available
        if not shutil.which("wine"):
            raise RuntimeError("Wine is required to run Ministry of Flat (UnWrapConsole3.exe) on Linux, but 'wine' was not found in PATH.")
        
        # Prepend wine to the command
        command = ["wine"] + command
        
    try:
        result = subprocess.run(
            command,
            check=True, capture_output=True, text=True, cwd=cwd
        )
        if result.stdout:
            print(f"Ministry of Flat stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        # Ministry of Flat sometimes returns exit code 1 even on success.
        # Check if the output indicates success.
        if e.returncode == 1 and "Saving complete" in e.stdout:
            print(f"Ministry of Flat finished with exit code 1 (treated as success). Stdout: {e.stdout}")
            return

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
            
            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh", process=False)
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
                try:
                    # Try newer PIL version first
                    tile = pil_img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fallback to older PIL version
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
                "texture_coords": (["Local", "Global", "UV"], {"default": "UV"}),
                "direction": (["X", "Y", "Z", "Normal", "Custom Normal", "RGB to XYZ"], {"default": "Normal"}),
                "strength": ("FLOAT", {"default": 0.005, "min": -10.0, "max": 10.0, "step": 0.001}),
                "midlevel": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
                "texture_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "displacement_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "displace"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def displace(self, trimesh, texture_coords, direction, strength, midlevel, merge_distance, texture_scale, displacement_map=None):
        if texture_coords.upper() == 'UV' and (not hasattr(trimesh.visual, 'uv') or len(trimesh.visual.uv) == 0):
            raise Exception("Input mesh must have UV coordinates for displacement with 'UV' texture coordinates. Use BlenderUnwrap first.")

        original_material = None
        original_uv = None
        if hasattr(trimesh, "visual"):
            if hasattr(trimesh.visual, "material") and trimesh.visual.material is not None:
                mat = trimesh.visual.material
                original_material = mat.copy() if hasattr(mat, "copy") else mat
            if hasattr(trimesh.visual, "uv") and trimesh.visual.uv is not None:
                original_uv = np.array(trimesh.visual.uv, copy=True)

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

            if displacement_map is not None and texture_scale != 1.0:
                original_width, original_height = img.size
                tile_width = int(original_width / texture_scale)
                tile_height = int(original_height / texture_scale)

                if tile_width > 0 and tile_height > 0:
                    try:
                        # Try newer PIL version first
                        tile = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # Fallback to older PIL version
                        tile = img.resize((tile_width, tile_height), Image.LANCZOS)
                    tiled_image = Image.new(img.mode, (original_width, original_height))
                    
                    for y in range(0, original_height, tile_height):
                        for x in range(0, original_width, tile_width):
                            tiled_image.paste(tile, (x, y))
                    
                    img = tiled_image

            img.save(image_path)
            
            params = {
                'strength': strength,
                'midlevel': midlevel,
                'merge_distance': merge_distance,
                'texture_coords': texture_coords,
                'direction': direction,
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
    
    disp_mod.texture_coords = p['texture_coords'].upper()
    if disp_mod.texture_coords == 'UV':
        if not obj.data.uv_layers:
            raise Exception("Mesh does not have a UV map. Cannot use UV coordinates for displacement.")
            
    direction_map = {{ "RGB TO XYZ": "RGB_TO_XYZ" }}
    blender_direction = direction_map.get(p['direction'].upper(), p['direction'].upper())
    disp_mod.direction = blender_direction

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

            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh", process=False)
            if original_uv is not None or original_material is not None:
                visuals = getattr(processed_mesh, "visual", None)
                material = original_material if original_material is not None else getattr(visuals, "material", None)
                uv_data = original_uv if original_uv is not None else getattr(visuals, "uv", None)
                image_data = getattr(visuals, "image", None)
                if visuals is None or (uv_data is not None and not hasattr(visuals, "uv")):
                    processed_mesh.visual = trimesh_loader.visual.texture.TextureVisuals(
                        uv=uv_data,
                        image=image_data,
                        material=material
                    )
                else:
                    if uv_data is not None:
                        processed_mesh.visual.uv = uv_data
                    if image_data is not None and hasattr(processed_mesh.visual, "image"):
                        processed_mesh.visual.image = image_data
                    if material is not None:
                        processed_mesh.visual.material = material

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

            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh", process=False)
            visuals = getattr(processed_mesh, "visual", None)
            blender_uv = getattr(visuals, "uv", None) if visuals is not None else None
            blender_material = getattr(visuals, "material", None) if visuals is not None else None

            uv_data = blender_uv
            if uv_data is None and original_uv is not None and original_uv.shape[0] == processed_mesh.vertices.shape[0]:
                uv_data = original_uv

            material = blender_material if blender_material is not None else original_material

            if visuals is None or not hasattr(visuals, "uv"):
                if uv_data is not None or material is not None:
                    processed_mesh.visual = trimesh_loader.visual.texture.TextureVisuals(
                        uv=uv_data,
                        material=material
                    )
            else:
                if uv_data is not None:
                    processed_mesh.visual.uv = uv_data
                if material is not None:
                    processed_mesh.visual.material = material
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
                "recalculate_normals": ("BOOLEAN", {"default": True}),
                "smooth_angle": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 180.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "process"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def process(self, trimesh, merge_distance, keep_biggest, fill_holes, recalculate_normals, smooth_angle):
        original_material = None
        original_uv = None
        if hasattr(trimesh, 'visual'):
            if hasattr(trimesh.visual, 'material') and trimesh.visual.material is not None:
                original_material = trimesh.visual.material.copy() if hasattr(trimesh.visual.material, 'copy') else trimesh.visual.material
            if hasattr(trimesh.visual, 'uv') and trimesh.visual.uv is not None:
                original_uv = np.array(trimesh.visual.uv, copy=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            params = {
                'merge_distance': merge_distance,
                'keep_biggest': keep_biggest,
                'fill_holes': fill_holes,
                'recalculate_normals': recalculate_normals,
                'smooth_angle': smooth_angle,
            }
            params['input_mesh'] = input_mesh_path
            params['output_mesh'] = output_mesh_path
            script = """
import bpy
import bmesh
import sys
import traceback
from math import radians

p = """ + str(params) + """

def merge_preserve_uv(mesh_obj, distance):
    if distance <= 0.0:
        return
    bm = bmesh.from_edit_mesh(mesh_obj.data)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer:
        verts_to_merge = []
        for v in bm.verts:
            if not v.link_loops:
                verts_to_merge.append(v)
                continue
            uv_coords = {
                (round(loop[uv_layer].uv.x, 6), round(loop[uv_layer].uv.y, 6))
                for loop in v.link_loops
            }
            if len(uv_coords) <= 1:
                verts_to_merge.append(v)
    else:
        verts_to_merge = list(bm.verts)
    if verts_to_merge:
        bmesh.ops.remove_doubles(bm, verts=verts_to_merge, dist=distance)
        bmesh.update_edit_mesh(mesh_obj.data)
    bm.free()

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

    merge_preserve_uv(obj, float(p['merge_distance']))
    bpy.ops.mesh.select_all(action='SELECT')

    if p['recalculate_normals']:
        bpy.ops.mesh.normals_make_consistent(inside=False)

    if p['keep_biggest']:
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')

        mesh_objects = [o for o in bpy.context.selected_objects if o.type == 'MESH' and o.data]

        if len(mesh_objects) > 1:
            biggest = max(mesh_objects, key=lambda o: len(o.data.vertices))

            for other in mesh_objects:
                if other != biggest:
                    bpy.data.objects.remove(other, do_unlink=True)

            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = biggest
            biggest.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    if p['fill_holes']:
        bpy.ops.mesh.fill_holes(sides=0)

    merge_preserve_uv(bpy.context.view_layer.objects.active, float(p['merge_distance']))

    if p['recalculate_normals']:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        bpy.ops.mesh.mark_sharp(clear=True)
        bpy.ops.mesh.normals_make_consistent(inside=False)

    bpy.ops.object.mode_set(mode='OBJECT')

    angle_value = radians(float(p['smooth_angle']))
    active_obj = bpy.context.view_layer.objects.active
    if active_obj and active_obj.type == 'MESH':
        data = active_obj.data
        if hasattr(data, "use_auto_smooth"):
            data.use_auto_smooth = True
        if hasattr(data, "auto_smooth_angle"):
            data.auto_smooth_angle = angle_value
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=angle_value, keep_sharp_edges=True)
    except TypeError:
        bpy.ops.object.shade_smooth_by_angle(angle=float(p['smooth_angle']), keep_sharp_edges=True)

    bpy.ops.export_scene.gltf(filepath=p['output_mesh'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)

            _run_blender_script(script_path)

            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh", process=False)

            visuals = getattr(processed_mesh, "visual", None)
            blender_uv = getattr(visuals, "uv", None) if visuals is not None else None
            blender_material = getattr(visuals, "material", None) if visuals is not None else None

            uv_data = blender_uv
            if uv_data is None and original_uv is not None and original_uv.shape[0] == processed_mesh.vertices.shape[0]:
                uv_data = original_uv

            material = blender_material if blender_material is not None else original_material

            if visuals is None or not hasattr(visuals, "uv"):
                if uv_data is not None or material is not None:
                    processed_mesh.visual = trimesh_loader.visual.texture.TextureVisuals(
                        uv=uv_data,
                        material=material
                    )
            else:
                if uv_data is not None:
                    processed_mesh.visual.uv = uv_data
                if material is not None:
                    processed_mesh.visual.material = material

            try:
                processed_normals = np.array(processed_mesh.vertex_normals, copy=True)
                processed_mesh.vertex_normals = processed_normals
            except Exception:
                pass

            return (processed_mesh,)

class MirrorMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "axis_x": ("BOOLEAN", {"default": True}),
                "axis_y": ("BOOLEAN", {"default": False}),
                "axis_z": ("BOOLEAN", {"default": False}),
                "use_clip": ("BOOLEAN", {"default": True}),
                "use_merge": ("BOOLEAN", {"default": True}),
                "merge_threshold": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "mirror"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def mirror(self, trimesh, axis_x, axis_y, axis_z, use_clip, use_merge, merge_threshold):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            params = {
                'axis_x': axis_x, 'axis_y': axis_y, 'axis_z': axis_z,
                'use_clip': use_clip, 'use_merge': use_merge,
                'merge_threshold': merge_threshold,
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
    if not obj: raise Exception("No mesh found in the imported GLB file.")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    if p['axis_x']:
        bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(1, 0, 0), clear_outer=True)
    if p['axis_y']:
        bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 1, 0), clear_outer=True)
    if p['axis_z']:
        bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_outer=True)

    bpy.ops.object.mode_set(mode='OBJECT')

    mod = obj.modifiers.new(name="MirrorMod", type='MIRROR')
    mod.use_axis[0] = p['axis_x']
    mod.use_axis[1] = p['axis_y']
    mod.use_axis[2] = p['axis_z']
    mod.use_clip = p['use_clip']
    mod.use_mirror_merge = p['use_merge']
    mod.merge_threshold = p['merge_threshold']

    bpy.ops.object.modifier_apply(modifier=mod.name)

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

class SubdivisionMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
                "levels": ("INT", {"default": 2, "min": 0, "max": 6}),
                "quality": ("INT", {"default": 3, "min": 1, "max": 6}),
                "uv_smooth": (['PRESERVE_BOUNDARIES', 'ALL', 'PRESERVE_CORNERS', 'NONE'], {"default": 'PRESERVE_BOUNDARIES'}),
                "boundary_smooth": (['ALL', 'PRESERVE_CORNERS'], {"default": 'ALL'}),
                "use_limit_surface": ("BOOLEAN", {"default": True}),
                "use_custom_normals": ("BOOLEAN", {"default": False}),
                "smooth_shading": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "subdivide"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def subdivide(self, trimesh, merge_distance, levels, quality, uv_smooth, boundary_smooth, use_limit_surface, use_custom_normals, smooth_shading):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            params = {
                'merge_distance': merge_distance,
                'levels': levels,
                'quality': quality,
                'uv_smooth': uv_smooth,
                'boundary_smooth': boundary_smooth,
                'use_limit_surface': use_limit_surface,
                'use_custom_normals': use_custom_normals,
                'smooth_shading': smooth_shading,
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
    if not obj: raise Exception("No mesh found in the imported GLB file.")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if p['merge_distance'] > 0.0:
        bpy.ops.mesh.remove_doubles(threshold=p['merge_distance'])
    bpy.ops.object.mode_set(mode='OBJECT')

    mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    mod.subdivision_type = 'CATMULL_CLARK'
    mod.levels = p['levels']
    mod.render_levels = p['levels']
    mod.quality = p['quality']
    mod.uv_smooth = p['uv_smooth']
    mod.boundary_smooth = p['boundary_smooth']
    mod.use_limit_surface = p['use_limit_surface']
    mod.use_custom_normals = p['use_custom_normals']

    bpy.ops.object.modifier_apply(modifier=mod.name)
    
    if p['smooth_shading']:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()

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

class QuadriflowRemesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "smooth_shading": ("BOOLEAN", {"default": True}),
                "auto_scale_fix": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Quadriflow_Settings": ("GROUP",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "remesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def remesh(self, trimesh, smooth_shading, auto_scale_fix, Quadriflow_Settings=None):
        defaults = {
            'target_faces': 62000,
            'mode': 'FACES',
            'target_ratio': 1.0,
            'target_edge_length': 0.1,
            'use_mesh_symmetry': False,
            'use_preserve_sharp': False,
            'use_preserve_boundary': True,
            'preserve_attributes': False,
            'seed': 0,
        }

        params = defaults.copy()
        if Quadriflow_Settings:
            params.update(Quadriflow_Settings)
        
        params.update({
            'smooth_shading': smooth_shading,
            'auto_scale_fix': auto_scale_fix
            })

        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            quadriflow_props = {k: v for k, v in params.items() if k not in ['smooth_shading', 'auto_scale_fix']}

            script_params = {
                'input_mesh': repr(input_mesh_path),
                'output_mesh': repr(output_mesh_path),
                'quadriflow_props': repr(quadriflow_props),
                'smooth_shading': repr(params['smooth_shading']),
                'auto_scale_fix': repr(params['auto_scale_fix'])
            }

            script = f"""
import bpy, sys, traceback
p = {{ {', '.join(f'\"{k}\": {v}' for k, v in script_params.items())} }}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p['input_mesh'])
    
    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    if not obj:
        raise Exception("No mesh found in the imported GLB file.")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    if p['auto_scale_fix']:
        scale_factor = 10.0
        bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
        bpy.ops.object.transform_apply(scale=True, location=False, rotation=False)
        
        bpy.ops.object.quadriflow_remesh(**p['quadriflow_props'])
        
        bpy.ops.transform.resize(value=(1.0/scale_factor, 1.0/scale_factor, 1.0/scale_factor))
        bpy.ops.object.transform_apply(scale=True, location=False, rotation=False)
    else:
        bpy.ops.object.quadriflow_remesh(**p['quadriflow_props'])

    if p['smooth_shading']:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()

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
            return (processed_mesh,)

class QuadriflowSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "target_faces": ("INT", {"default": 62000, "min": 100, "max": 1000000, "step": 100}),
            "mode": (['FACES', 'RATIO', 'EDGE'], {"default": 'FACES'}),
            "target_ratio": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number"}),
            "target_edge_length": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001, "display": "number"}),
            "use_mesh_symmetry": ("BOOLEAN", {"default": False}),
            "use_preserve_sharp": ("BOOLEAN", {"default": False}),
            "use_preserve_boundary": ("BOOLEAN", {"default": True}),
            "preserve_attributes": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 10000}),
        }}
    RETURN_TYPES = ("GROUP",)
    FUNCTION = "get_settings"
    CATEGORY = "Comfy_BlenderTools/Utils/Settings"
    def get_settings(self, **kwargs): return (kwargs,)
    
class Pyremesh:
    """
    A ComfyUI node for adaptive mesh simplification and remeshing using PyMeshLab.
    It creates a high-quality, uniform mesh that adapts to the curvature of the input,
    placing more polygons in areas of high detail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_faces": ("INT", {
                    "doc_string": "Approximate number of faces desired after remeshing.",
                    "default": 20000, "min": 100, "max": 500000, "step": 100
                }),
                "adaptive_sizing": ("BOOLEAN", {
                    "doc_string": "Enable to adapt triangle size to local curvature. More triangles will be used in detailed areas.",
                    "default": True
                }),
                "adaptive_sensitivity": ("FLOAT", {
                    "doc_string": "Controls how aggressively the mesh adapts to curvature. Higher values mean more refinement in detailed areas.",
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05
                }),
                "use_curvature_weighted": ("BOOLEAN", {
                    "doc_string": "Concentrate triangles in high-curvature regions using curvature-weighted decimation.",
                    "default": True
                }),
                "preserve_boundaries": ("BOOLEAN", {
                    "doc_string": "If enabled, tries to preserve the original mesh boundaries (open edges).",
                    "default": True
                }),
                "preserve_features": ("BOOLEAN", {
                    "doc_string": "If enabled, tries to preserve sharp features based on the feature angle.",
                    "default": True
                }),
                "feature_angle_deg": ("FLOAT", {
                    "doc_string": "The angle (in degrees) used to detect sharp features when 'preserve_features' is enabled.",
                    "default": 30.0, "min": 0.0, "max": 180.0, "step": 1.0
                }),
                "make_watertight": ("BOOLEAN", {
                    "doc_string": "After remeshing, run a post-processing step to fill holes and make the mesh watertight.",
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "remesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def remesh(self, trimesh, target_faces, adaptive_sizing, adaptive_sensitivity, use_curvature_weighted, preserve_boundaries, preserve_features, feature_angle_deg, make_watertight):
        import numpy as np
        ms = pymeshlab.MeshSet()
        pymeshlab_mesh = pymeshlab.Mesh(vertex_matrix=trimesh.vertices, face_matrix=trimesh.faces)
        ms.add_mesh(pymeshlab_mesh, "input_mesh")

        v = trimesh.vertices.astype(np.float64)
        f = trimesh.faces.astype(np.int64)
        tri = v[f]
        areas = 0.5 * np.linalg.norm(np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0]), axis=1)
        total_area = float(np.sum(areas))
        bb_min = v.min(axis=0)
        bb_max = v.max(axis=0)
        bbox_diag = float(np.linalg.norm(bb_max - bb_min))
        sens_bias = max(0.2, min(1.8, 1.0 + (adaptive_sensitivity - 0.5) * 0.3))
        eff_target = max(50, int(target_faces * sens_bias))
        if total_area <= 0 or bbox_diag <= 0:
            targetlen_percent = 1.0
        else:
            edge_len = np.sqrt((4.0 * total_area) / (np.sqrt(3.0) * eff_target))
            targetlen_percent = max(0.01, min(20.0, (edge_len / bbox_diag) * 100.0))

        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PercentageValue(targetlen_percent),
            adaptive=adaptive_sizing,
            checksurfdist=False,
            featuredeg=feature_angle_deg
        )

        cur_faces = ms.current_mesh().face_matrix().shape[0]
        if cur_faces > target_faces:
            try:
                if use_curvature_weighted:
                    ms.compute_curvature_principal_directions()
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=int(target_faces),
                    qualityweight=bool(use_curvature_weighted)
                )
            except Exception as e:
                print(f"Decimation fallback: {e}")
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=int(target_faces)
                )

        if make_watertight:
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_unreferenced_vertices()

        processed_mesh = ms.current_mesh()
        new_vertices = processed_mesh.vertex_matrix()
        new_faces = processed_mesh.face_matrix()

        if new_vertices.shape[0] == 0 or new_faces.shape[0] == 0:
            print("WARNING: PyMeshLab remeshing resulted in an empty mesh. Returning original mesh.")
            return (trimesh,)

        output_mesh = trimesh_loader.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        output_mesh.remove_unreferenced_vertices()
        output_mesh.remove_degenerate_faces()

        if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
            output_mesh.visual = trimesh_loader.visual.texture.TextureVisuals()
            output_mesh.visual.material = trimesh.visual.material

        return (output_mesh,)


class O3DRemesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_faces": ("INT", {
                    "default": 50000, "min": 1000, "max": 500000, "step": 1000
                }),
                "method": (["poisson_quadric", "adaptive_decimation", "cluster_simplify"], {
                    "default": "poisson_quadric"
                }),
                "poisson_depth": ("INT", {
                    "default": 9, "min": 6, "max": 12, "step": 1
                }),
                "sample_points": ("INT", {
                    "default": 100000, "min": 10000, "max": 1000000, "step": 10000
                }),
                "preserve_boundary": ("BOOLEAN", {"default": True}),
                "smooth_iterations": ("INT", {
                    "default": 5, "min": 0, "max": 20, "step": 1
                }),
                "make_watertight": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "o3d_remesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def o3d_remesh(self, trimesh, target_faces, method, poisson_depth, sample_points, 
                      preserve_boundary, smooth_iterations, make_watertight):
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is required for InstaRemesh. Install with: pip install open3d")
        
        # Convert trimesh to Open3D mesh
        vertices = trimesh.vertices.astype(np.float64)
        faces = trimesh.faces.astype(np.int32)
        
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh.compute_vertex_normals()
        
        if method == "poisson_quadric":
            # Sample points from mesh surface
            if make_watertight:
                # Use Poisson reconstruction for watertight mesh
                pcd = o3d_mesh.sample_points_poisson_disk(sample_points)
                pcd.estimate_normals()
                
                # Poisson surface reconstruction
                mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
                )
                
                # Remove disconnected components
                mesh_poisson.remove_degenerate_triangles()
                mesh_poisson.remove_duplicated_triangles()
                mesh_poisson.remove_duplicated_vertices()
                mesh_poisson.remove_non_manifold_edges()
                
                processed_mesh = mesh_poisson
            else:
                processed_mesh = o3d_mesh
            
            # Quadric decimation to target face count
            if len(processed_mesh.triangles) > target_faces:
                processed_mesh = processed_mesh.simplify_quadric_decimation(target_faces)
                
        elif method == "adaptive_decimation":
            # Optimized adaptive decimation with preprocessing for speed
            current_faces = len(o3d_mesh.triangles)
            print(f"Input mesh: {current_faces} faces, target: {target_faces}")
            
            if current_faces <= target_faces:
                print("Mesh already at or below target face count")
                processed_mesh = o3d_mesh
            else:
                # Multi-stage decimation for better speed and quality
                reduction_ratio = target_faces / current_faces
                
                if reduction_ratio < 0.1 and current_faces > 100000:
                    # Extreme reduction on high-poly mesh: use multi-stage approach
                    print("Using multi-stage decimation for extreme reduction...")
                    
                    # Stage 1: Fast cluster-based pre-reduction to ~10x target
                    intermediate_target = min(target_faces * 10, current_faces // 2)
                    bbox = o3d_mesh.get_axis_aligned_bounding_box()
                    bbox_diag = np.linalg.norm(bbox.get_extent())
                    voxel_size = bbox_diag / np.sqrt(intermediate_target)
                    
                    stage1_mesh = o3d_mesh.simplify_vertex_clustering(
                        voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average
                    )
                    print(f"Stage 1 (clustering): {len(stage1_mesh.triangles)} faces")
                    
                    # Stage 2: Quadric decimation to final target
                    if len(stage1_mesh.triangles) > target_faces:
                        processed_mesh = stage1_mesh.simplify_quadric_decimation(
                            target_number_of_triangles=target_faces
                        )
                        print(f"Stage 2 (quadric): {len(processed_mesh.triangles)} faces")
                    else:
                        processed_mesh = stage1_mesh
                        
                elif reduction_ratio < 0.3 and current_faces > 50000:
                    # Heavy reduction on medium-high poly: cluster first
                    print("Using cluster + quadric decimation for heavy reduction...")
                    
                    # Pre-reduce with clustering to ~3x target
                    intermediate_target = target_faces * 3
                    bbox = o3d_mesh.get_axis_aligned_bounding_box()
                    bbox_diag = np.linalg.norm(bbox.get_extent())
                    voxel_size = bbox_diag / np.sqrt(intermediate_target * 2)
                    
                    cluster_mesh = o3d_mesh.simplify_vertex_clustering(
                        voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average
                    )
                    print(f"Pre-clustering: {len(cluster_mesh.triangles)} faces")
                    
                    # Final quadric decimation
                    processed_mesh = cluster_mesh.simplify_quadric_decimation(
                        target_number_of_triangles=target_faces
                    )
                    print(f"Final quadric: {len(processed_mesh.triangles)} faces")
                    
                else:
                    # Direct quadric decimation for moderate reductions
                    print("Using direct quadric decimation...")
                    processed_mesh = o3d_mesh.simplify_quadric_decimation(
                        target_number_of_triangles=target_faces
                    )
                    print(f"Direct quadric result: {len(processed_mesh.triangles)} faces")
                
        elif method == "cluster_simplify":
            # Vertex clustering for fast simplification with improved robustness
            bbox = o3d_mesh.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            bbox_diag = np.linalg.norm(bbox_size)
            
            # Adaptive voxel size based on mesh size and target faces
            current_faces = len(o3d_mesh.triangles)
            if current_faces > 0:
                # Calculate voxel size to approximately achieve target faces
                reduction_factor = min(0.9, target_faces / current_faces)
                base_voxel_size = bbox_diag / 100  # Base size
                voxel_size = base_voxel_size * np.sqrt(reduction_factor)
                voxel_size = max(voxel_size, bbox_diag / 1000)  # Minimum voxel size
            else:
                voxel_size = bbox_diag / 200
            
            print(f"Using voxel size: {voxel_size:.6f} for bbox diagonal: {bbox_diag:.6f}")
            
            # Apply vertex clustering
            processed_mesh = o3d_mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
            
            # Validate mesh after clustering
            if len(processed_mesh.vertices) == 0 or len(processed_mesh.triangles) == 0:
                print("WARNING: Vertex clustering produced empty mesh. Using original.")
                processed_mesh = o3d_mesh
            
            # Further reduce if still too many faces
            elif len(processed_mesh.triangles) > target_faces:
                try:
                    processed_mesh = processed_mesh.simplify_quadric_decimation(target_faces)
                except Exception as e:
                    print(f"WARNING: Quadric decimation failed: {e}. Using clustered result.")
        
        # Post-processing
        if smooth_iterations > 0:
            processed_mesh = processed_mesh.filter_smooth_laplacian(
                number_of_iterations=smooth_iterations
            )
        
        # Clean up mesh
        processed_mesh.remove_degenerate_triangles()
        processed_mesh.remove_duplicated_triangles()
        processed_mesh.remove_duplicated_vertices()
        processed_mesh.remove_non_manifold_edges()
        
        # Convert back to trimesh
        new_vertices = np.asarray(processed_mesh.vertices)
        new_faces = np.asarray(processed_mesh.triangles)
        
        # Validate and clean NaN/infinite values
        if np.any(np.isnan(new_vertices)) or np.any(np.isinf(new_vertices)):
            print("WARNING: Found NaN or infinite values in vertices. Cleaning mesh...")
            # Remove faces with NaN/infinite vertices
            valid_vertices = ~(np.isnan(new_vertices).any(axis=1) | np.isinf(new_vertices).any(axis=1))
            vertex_map = np.full(len(new_vertices), -1, dtype=np.int32)
            vertex_map[valid_vertices] = np.arange(np.sum(valid_vertices))
            
            new_vertices = new_vertices[valid_vertices]
            
            # Update faces to use new vertex indices
            valid_faces = []
            for face in new_faces:
                if all(vertex_map[v] >= 0 for v in face):
                    valid_faces.append([vertex_map[v] for v in face])
            
            if len(valid_faces) == 0:
                print("WARNING: All faces removed due to NaN/infinite values. Returning original mesh.")
                return (trimesh,)
                
            new_faces = np.array(valid_faces)
        
        if new_vertices.shape[0] == 0 or new_faces.shape[0] == 0:
            print("WARNING: O3DRemesh produced an empty mesh. Returning original mesh.")
            return (trimesh,)
        
        output_mesh = trimesh_loader.Trimesh(
            vertices=new_vertices, 
            faces=new_faces, 
            process=False
        )
        
        # Preserve material if exists
        if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
            output_mesh.visual = trimesh_loader.visual.texture.TextureVisuals()
            output_mesh.visual.material = trimesh.visual.material
        
        return (output_mesh,)

class InstantMeshes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_faces": ("INT", {
                    "default": 5000, "min": 100, "max": 500000, "step": 100
                }),
                "make_watertight": ("BOOLEAN", {"default": True}),
                "simplify_before": ("BOOLEAN", {"default": False}),
                "simplify_ratio": ("FLOAT", {
                    "default": 0.3, "min": 0.1, "max": 0.9, "step": 0.1
                }),
                "adaptive_detail": ("BOOLEAN", {"default": False}),
                "orientation_symmetry": (["2", "4", "6"], {"default": "4"}),
                "position_symmetry": (["4", "6"], {"default": "4"}),
                "crease_angle": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 180.0, "step": 1.0
                }),
                "boundary_alignment": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "run_instant_meshes"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def get_instant_meshes_path(self):
        """Get Instant Meshes executable path from custom node folder"""
        # Get the directory where utils.py is located (custom node folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for Linux/Mac binary first on non-Windows platforms
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            linux_path = os.path.join(current_dir, "Instant_Meshes")
            if os.path.isfile(linux_path):
                print(f"INFO: Found Instant Meshes in custom node folder: {linux_path}")
                return linux_path

        # Check for Windows binary (default for Windows or fallback)
        exe_path = os.path.join(current_dir, "Instant_Meshes.exe")
        if os.path.isfile(exe_path):
            print(f"INFO: Found Instant Meshes in custom node folder: {exe_path}")
            return exe_path
                
        raise FileNotFoundError(
            f"Instant Meshes executable not found in: {current_dir}\n"
            "Please download Instant_Meshes (Linux/Mac) or Instant_Meshes.exe (Windows) from https://github.com/wjakob/instant-meshes \n"
            "and place it in the custom node folder next to utils.py"
        )

    def preprocess_mesh(self, trimesh_mesh, make_watertight, simplify_before, simplify_ratio, adaptive_detail=False):
        """Preprocess mesh using PyMeshLab/Open3D with optional detail adaptation"""
        try:
            import open3d as o3d
            
            # Convert to Open3D
            vertices = trimesh_mesh.vertices.astype(np.float64)
            faces = trimesh_mesh.faces.astype(np.int32)
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_vertex_normals()
            
            # Clean up
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh.remove_duplicated_triangles()
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_non_manifold_edges()
            
            # Detail-adaptive preprocessing for better Instant Meshes results
            if adaptive_detail:
                print("Applying detail-adaptive preprocessing...")
                
                # Compute curvature to identify high-detail areas
                o3d_mesh.compute_triangle_normals()
                o3d_mesh.compute_vertex_normals()
                
                # Apply targeted smoothing to reduce noise while preserving features
                # Light smoothing to reduce noise that could confuse Instant Meshes
                smoothed_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=2)
                
                # Use the smoothed version for better field computation
                o3d_mesh = smoothed_mesh
            
            if make_watertight:
                # Sample points and reconstruct
                sample_count = 50000 if not adaptive_detail else 75000  # More samples for detail preservation
                pcd = o3d_mesh.sample_points_poisson_disk(sample_count)
                pcd.estimate_normals()
                
                # Poisson reconstruction for watertight mesh
                depth = 8 if not adaptive_detail else 9  # Higher depth for detail preservation
                mesh_watertight, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, width=0, scale=1.1, linear_fit=False
                )
                
                mesh_watertight.remove_degenerate_triangles()
                mesh_watertight.remove_duplicated_triangles()
                mesh_watertight.remove_duplicated_vertices()
                
                o3d_mesh = mesh_watertight
            
            if simplify_before:
                # Simplify before Instant Meshes
                current_faces = len(o3d_mesh.triangles)
                target_faces = int(current_faces * simplify_ratio)
                if target_faces < current_faces:
                    if adaptive_detail:
                        # Use more conservative simplification to preserve detail
                        target_faces = int(current_faces * (simplify_ratio + 0.1))
                    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
            
            # Convert back to trimesh
            new_vertices = np.asarray(o3d_mesh.vertices)
            new_faces = np.asarray(o3d_mesh.triangles)
            
            preprocessed_mesh = trimesh_loader.Trimesh(
                vertices=new_vertices, 
                faces=new_faces, 
                process=False
            )
            
            return preprocessed_mesh
            
        except ImportError:
            print("WARNING: Open3D not available. Using PyMeshLab for preprocessing.")
            # Fallback to PyMeshLab
            ms = pymeshlab.MeshSet()
            pymeshlab_mesh = pymeshlab.Mesh(vertex_matrix=trimesh_mesh.vertices, face_matrix=trimesh_mesh.faces)
            ms.add_mesh(pymeshlab_mesh, "input_mesh")
            
            # Clean mesh
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_duplicate_faces()
            
            if make_watertight:
                ms.meshing_close_holes(maxholesize=30)
                
            if simplify_before:
                current_faces = ms.current_mesh().face_number()
                target_faces = int(current_faces * simplify_ratio)
                if target_faces < current_faces:
                    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            
            processed_mesh = ms.current_mesh()
            new_vertices = processed_mesh.vertex_matrix()
            new_faces = processed_mesh.face_matrix()
            
            return trimesh_loader.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    def calculate_instant_meshes_target(self, desired_faces, input_face_count=None):
        """Calculate target faces for Instant Meshes to achieve desired output"""
        # Instant Meshes tends to output more faces than requested
        # Based on empirical testing, apply correction factors
        
        if input_face_count:
            # If we know input face count, use ratio-based approach
            reduction_ratio = desired_faces / input_face_count
            if reduction_ratio > 0.8:
                # Light reduction: Instant Meshes is fairly accurate
                target_faces = int(desired_faces * 0.9)
            elif reduction_ratio > 0.3:
                # Medium reduction: Apply moderate correction
                target_faces = int(desired_faces * 0.7)
            else:
                # Heavy reduction: Apply strong correction
                target_faces = int(desired_faces * 0.5)
        else:
            # Fallback: General correction based on face count ranges
            if desired_faces < 2000:
                target_faces = int(desired_faces * 0.6)
            elif desired_faces < 10000:
                target_faces = int(desired_faces * 0.7)
            elif desired_faces < 50000:
                target_faces = int(desired_faces * 0.8)
            else:
                target_faces = int(desired_faces * 0.85)
        
        # Ensure minimum face count
        target_faces = max(target_faces, 100)
        
        print(f"Desired faces: {desired_faces}, Calculated target for Instant Meshes: {target_faces}")
        return target_faces

    def run_instant_meshes_iterative(self, trimesh, desired_faces, processed_mesh, temp_dir, exe_path, 
                                    adaptive_detail=False, orientation_symmetry="4", position_symmetry="4", 
                                    crease_angle=30.0, boundary_alignment=True, max_iterations=3):
        """Run Instant Meshes with iterative target adjustment and adaptive detail control"""
        input_mesh_path = os.path.join(temp_dir, "input.obj")
        output_mesh_path = os.path.join(temp_dir, "output.obj")
        
        # Export input mesh
        processed_mesh.export(input_mesh_path)
        input_face_count = len(processed_mesh.faces)
        
        best_result = None
        best_difference = float('inf')
        
        for iteration in range(max_iterations):
            if iteration == 0:
                # First attempt with calculated target
                target_faces = self.calculate_instant_meshes_target(desired_faces, input_face_count)
            else:
                # Adjust based on previous result
                if best_result is not None:
                    actual_faces = len(best_result.faces)
                    ratio = actual_faces / target_faces if target_faces > 0 else 1
                    
                    # Adjust target for next iteration
                    if actual_faces > desired_faces:
                        # Got too many faces, reduce target more aggressively
                        adjustment = 0.7 if ratio > 2 else 0.8
                    else:
                        # Got too few faces, increase target
                        adjustment = 1.3 if ratio < 0.5 else 1.2
                    
                    target_faces = int(target_faces * (desired_faces / actual_faces) * adjustment)
                    target_faces = max(target_faces, 100)
                
            print(f"Iteration {iteration + 1}: Target faces = {target_faces}")
            
            # Build command with adaptive detail parameters
            cmd = [
                exe_path,
                "-o", output_mesh_path,
                "-f", str(target_faces),
                "-S", "2"
            ]
            
            # Add adaptive detail parameters
            if adaptive_detail:
                print(f"Using adaptive detail with rosy={orientation_symmetry}, posy={position_symmetry}")
                
                # Start with simpler parameters to avoid extraction failure
                if iteration == 0:
                    # First iteration: Use conservative settings
                    cmd.extend(["-r", "4"])  # Conservative orientation symmetry
                    if boundary_alignment:
                        cmd.append("-b")      # Boundary alignment helps
                else:
                    # Later iterations: Use more advanced settings if needed
                    cmd.extend(["-r", orientation_symmetry])
                    if int(position_symmetry) <= 4:  # Only use posy if reasonable
                        cmd.extend(["-p", position_symmetry])
                    if boundary_alignment:
                        cmd.append("-b")
            else:
                # Non-adaptive mode: Use minimal parameters for reliability
                pass
                
            # Add crease angle only if reasonable and not causing conflicts
            if crease_angle > 0 and crease_angle < 90 and not adaptive_detail:
                cmd.extend(["-c", str(crease_angle)])
                
            cmd.append(input_mesh_path)  # Input file last
            
            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    timeout=300
                )
                
                if result.stdout:
                    print(f"Instant Meshes stdout: {result.stdout}")
                
                # Load and check result
                if os.path.exists(output_mesh_path):
                    current_result = trimesh_loader.load(output_mesh_path, force="mesh")
                    actual_faces = len(current_result.faces)
                    
                    if actual_faces == 0:
                        print(f"Warning: Iteration {iteration + 1} produced 0 faces. Trying fallback...")
                        
                        # Fallback: Try without adaptive parameters
                        fallback_cmd = [
                            exe_path,
                            "-o", output_mesh_path,
                            "-f", str(target_faces),
                            "-S", "1",  # Reduced smoothing
                            input_mesh_path
                        ]
                        
                        try:
                            print("Running fallback command without adaptive parameters...")
                            fallback_result = subprocess.run(
                                fallback_cmd,
                                check=True,
                                capture_output=True,
                                text=True,
                                timeout=300
                            )
                            
                            if os.path.exists(output_mesh_path):
                                current_result = trimesh_loader.load(output_mesh_path, force="mesh")
                                actual_faces = len(current_result.faces)
                                print(f"Fallback result: {actual_faces} faces")
                        except Exception as fallback_error:
                            print(f"Fallback also failed: {fallback_error}")
                            continue
                    
                    if actual_faces > 0:
                        difference = abs(actual_faces - desired_faces)
                        print(f"Result: {actual_faces} faces (difference: {difference})")
                        
                        # Keep best result
                        if difference < best_difference:
                            best_difference = difference
                            best_result = current_result
                        
                        # If we're close enough, stop iterating
                        tolerance = max(desired_faces * 0.15, 1000)  # Increased tolerance for adaptive mode
                        if difference <= tolerance:
                            print(f"Achieved acceptable result within tolerance ({tolerance} faces)")
                            break
                    else:
                        print(f"Iteration {iteration + 1} still produced 0 faces, continuing...")
                        
            except Exception as e:
                print(f"Iteration {iteration + 1} failed: {e}")
                if iteration == max_iterations - 1:  # Last iteration
                    raise
                continue
        
        if best_result is None:
            raise RuntimeError("All Instant Meshes iterations failed")
            
        return best_result

    def run_instant_meshes(self, trimesh, target_faces, make_watertight, 
                          simplify_before, simplify_ratio, adaptive_detail, orientation_symmetry, 
                          position_symmetry, crease_angle, boundary_alignment):
        """Run Instant Meshes remeshing with iterative target adjustment and adaptive detail (always preprocessed)"""
        
        # Get executable path
        exe_path = self.get_instant_meshes_path()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Always preprocess for InstantMeshes - it improves results significantly
            print("Preprocessing mesh for InstantMeshes (always enabled)...")
            processed_mesh = self.preprocess_mesh(
                trimesh, make_watertight, simplify_before, simplify_ratio, adaptive_detail
            )
            
            # Run iterative Instant Meshes with adaptive parameters
            try:
                output_mesh = self.run_instant_meshes_iterative(
                    trimesh, target_faces, processed_mesh, temp_dir, exe_path,
                    adaptive_detail, orientation_symmetry, position_symmetry,
                    crease_angle, boundary_alignment
                )
            except Exception as e:
                raise RuntimeError(f"Instant Meshes processing failed: {e}")
            
            # Preserve material if exists
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
                output_mesh.visual = trimesh_loader.visual.texture.TextureVisuals()
                output_mesh.visual.material = trimesh.visual.material
            
            return (output_mesh,)

class UnwrapColoredMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "colored_mesh": ("TRIMESH",),  # Colored mesh from weight painting workflow
                "uv_margin": ("FLOAT", {
                    "default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001
                }),
                "seam_sensitivity": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "unwrap_colored_mesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def unwrap_colored_mesh(self, colored_mesh, uv_margin, seam_sensitivity):
        """Unwrap colored mesh using color boundaries as seam guides with uvgami approach"""
        
        if not hasattr(colored_mesh, 'vertices') or not hasattr(colored_mesh, 'faces'):
            raise ValueError("Input must be a valid trimesh")
        
        print(f"Unwrapping colored mesh using color boundaries as seam guides...")
        
        try:
            import xatlas
        except ImportError:
            raise ImportError("xatlas is required for unwrapping. Install with: pip install xatlas")
        
        # Get color information from mesh
        colors = None
        if hasattr(colored_mesh, 'visual') and hasattr(colored_mesh.visual, 'vertex_colors'):
            colors = colored_mesh.visual.vertex_colors
        elif hasattr(colored_mesh, 'visual') and hasattr(colored_mesh.visual, 'face_colors'):
            colors = colored_mesh.visual.face_colors
        else:
            print("Warning: No color information found, using standard unwrap")
            return self.standard_unwrap(colored_mesh, uv_margin)
        
        # Convert to numpy arrays
        vertices = np.array(colored_mesh.vertices, dtype=np.float32)
        faces = np.array(colored_mesh.faces, dtype=np.uint32)
        
        # Detect color boundaries for seam restrictions
        seam_weights = self.detect_color_boundaries(vertices, faces, colors, seam_sensitivity)
        
        try:
            # Use xatlas with seam guidance based on colors
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            
            # Generate with color-guided seaming
            atlas.generate(
                pack_options=xatlas.PackOptions(
                    padding=int(uv_margin * 1024),  # Convert margin to texel padding
                    texels_per_unit=1024,
                    resolution=1024
                ),
                chart_options=xatlas.ChartOptions(
                    max_iterations=5,  # Moderate iterations for good quality
                    normal_seam_weight=2.0,  # Reduced since we're using color guidance
                    texture_seam_weight=0.5
                )
            )
            
            # Get result
            vmapping, indices, uvs = atlas[0]
            
            # Create unwrapped mesh
            unwrapped_mesh = trimesh.Trimesh(
                vertices=vertices[vmapping],
                faces=indices.reshape(-1, 3),
                process=False
            )
            
            # Preserve original colors if available
            if colors is not None:
                if len(colors) == len(vertices):  # Vertex colors
                    unwrapped_mesh.visual = trimesh.visual.ColorVisuals(
                        vertex_colors=colors[vmapping]
                    )
                else:  # Face colors
                    unwrapped_mesh.visual = trimesh.visual.ColorVisuals(
                        face_colors=colors
                    )
            
            # Add UV coordinates
            if hasattr(unwrapped_mesh.visual, 'uv'):
                unwrapped_mesh.visual.uv = uvs
            else:
                # Create texture visuals with UVs while preserving colors
                if hasattr(unwrapped_mesh, 'visual') and hasattr(unwrapped_mesh.visual, 'vertex_colors'):
                    unwrapped_mesh.visual = trimesh.visual.TextureVisuals(
                        uv=uvs,
                        material=trimesh.visual.material.SimpleMaterial(
                            diffuse=[255, 255, 255, 255]
                        )
                    )
                else:
                    unwrapped_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            print(f"Colored mesh unwrapped successfully using color-guided seaming")
            return (unwrapped_mesh,)
            
        except Exception as e:
            print(f"Color-guided unwrap failed: {str(e)}, falling back to standard unwrap")
            return self.standard_unwrap(colored_mesh, uv_margin)
    
    def detect_color_boundaries(self, vertices, faces, colors, sensitivity):
        """Detect color boundaries to guide seam placement (uvgami approach)"""
        seam_weights = np.ones(len(vertices), dtype=np.float32)
        
        if colors is None:
            return seam_weights
        
        try:
            # Convert colors to comparable format
            if len(colors.shape) > 2:
                colors = colors.reshape(-1, colors.shape[-1])
            
            # For each vertex, check color similarity with neighbors
            for face in faces:
                for i in range(3):
                    v1_idx = face[i]
                    v2_idx = face[(i + 1) % 3]
                    
                    if v1_idx < len(colors) and v2_idx < len(colors):
                        # Calculate color difference
                        color_diff = np.linalg.norm(colors[v1_idx] - colors[v2_idx])
                        
                        # If colors are significantly different, reduce seam weight
                        # (following uvgami's approach where low weight = avoid seams)
                        if color_diff > sensitivity:
                            seam_weights[v1_idx] *= 0.1  # Strongly avoid seams on color boundaries
                            seam_weights[v2_idx] *= 0.1
            
            print(f"Detected color boundaries with sensitivity {sensitivity}")
            return seam_weights
            
        except Exception as e:
            print(f"Color boundary detection failed: {e}")
            return seam_weights
    
    def standard_unwrap(self, mesh, uv_margin):
        """Fallback standard unwrap without color guidance"""
        try:
            import xatlas
            
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)
            
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            
            atlas.generate(
                pack_options=xatlas.PackOptions(
                    padding=int(uv_margin * 1024)
                )
            )
            
            vmapping, indices, uvs = atlas[0]
            
            unwrapped_mesh = trimesh.Trimesh(
                vertices=vertices[vmapping],
                faces=indices.reshape(-1, 3),
                process=False
            )
            
            unwrapped_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            return (unwrapped_mesh,)
            
        except Exception as e:
            print(f"Standard unwrap failed: {e}")
            return (mesh,)  # Return original mesh as fallback
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "split_meshes": ("LIST_TRIMESH",),  # List of mesh parts from PartField Splitter
                "seam_strategy": (["edge_to_furthest", "boundary_loop", "smart_seam"], {"default": "edge_to_furthest"}),
                "uv_margin": ("FLOAT", {
                    "default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001
                }),
                "angle_limit": ("FLOAT", {
                    "default": 66.0, "min": 30.0, "max": 89.0, "step": 1.0
                }),
                "pack_uv": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "unwrap_split_mesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def unwrap_split_mesh(self, split_meshes, seam_strategy, uv_margin, angle_limit, pack_uv):
        """Unwrap split meshes quickly and combine into one output"""
        
        if not isinstance(split_meshes, list) or len(split_meshes) == 0:
            raise ValueError("Input must be a non-empty list of mesh parts")
        
        print(f"Quick unwrapping {len(split_meshes)} mesh parts...")
        
        # Quick unwrap each part
        unwrapped_parts = []
        for i, mesh_part in enumerate(split_meshes):
            if not hasattr(mesh_part, 'vertices') or not hasattr(mesh_part, 'faces'):
                continue
            if len(mesh_part.vertices) == 0 or len(mesh_part.faces) == 0:
                continue
                
            print(f"Unwrapping part {i+1}/{len(split_meshes)}...")
            unwrapped_part = self.quick_unwrap_part(mesh_part, i)
            
            if unwrapped_part is not None:
                unwrapped_parts.append(unwrapped_part)
        
        if not unwrapped_parts:
            raise RuntimeError("No valid mesh parts could be unwrapped")
        
        # Just combine all parts into one mesh
        print(f"Combining {len(unwrapped_parts)} unwrapped parts...")
        combined_mesh = self.combine_meshes_simple(unwrapped_parts)
        
        # Pack UVs with Blender using specified settings
        print("Packing UVs with Blender (0.005 margin, minimum stretch 10)...")
        packed_mesh = self.pack_uvs_with_blender(combined_mesh)
        
        return (packed_mesh,)
    
    def quick_unwrap_part(self, mesh_part, part_index):
        """Quick unwrap keeping original position, with simple xatlas settings"""
        try:
            import xatlas
        except ImportError:
            print(f"Warning: xatlas not available, returning original part {part_index}")
            return mesh_part  # Return original if no xatlas
        
        # Convert to numpy arrays
        vertices = np.array(mesh_part.vertices, dtype=np.float32)
        faces = np.array(mesh_part.faces, dtype=np.uint32)
        
        try:
            # Simple unwrap with xatlas - no complex options
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            
            # Generate with minimal settings (following Fast Per-Part Unwrapping Strategy)
            atlas.generate()
            
            # Get result
            vmapping, indices, uvs = atlas[0]
            
            # Create unwrapped mesh keeping original vertex positions
            unwrapped_mesh = trimesh_loader.Trimesh(
                vertices=vertices[vmapping],  # Keep original positions
                faces=indices.reshape(-1, 3),
                process=False
            )
            
            # Add UV coordinates
            unwrapped_mesh.visual = trimesh_loader.visual.TextureVisuals(uv=uvs)
            
            print(f"Part {part_index} unwrapped successfully")
            return unwrapped_mesh
            
        except Exception as e:
            print(f"Unwrap failed for part {part_index}: {e}, returning original")
            return mesh_part  # Return original on failure
    
    def pack_uvs_together(self, unwrapped_parts, temp_dir, uv_margin):
        """Pack UV islands from multiple meshes using xatlas"""
        try:
            import xatlas
        except ImportError:
            print("Warning: xatlas not available for UV packing, using simple combination")
            return self.combine_meshes_simple(unwrapped_parts)
        
        print("Packing UVs using xatlas with 10 iterations...")
        
        # Combine all meshes first
        combined_mesh = self.combine_meshes_simple(unwrapped_parts)
        
        # Get vertices and faces
        vertices = np.array(combined_mesh.vertices, dtype=np.float32)
        faces = np.array(combined_mesh.faces, dtype=np.uint32)
        
        try:
            # Create xatlas atlas for packing
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices, faces)
            
            # Generate with packing focus and 10 iterations
            atlas.generate(
                pack_options=xatlas.PackOptions(
                    padding=int(uv_margin * 1024),
                    texels_per_unit=1024,
                    resolution=1024,
                    bruteforce=True,  # Better packing
                    create_image=False,
                    rotate_charts_to_axis=True,
                    rotate_charts=True
                ),
                chart_options=xatlas.ChartOptions(
                    max_iterations=10  # Your requested 10 iterations
                )
            )
            
            # Get packed result
            vmapping, indices, uvs = atlas[0]
            
            # Create final mesh with packed UVs
            packed_vertices = vertices[vmapping]
            packed_mesh = trimesh_loader.Trimesh(
                vertices=packed_vertices,
                faces=indices.reshape(-1, 3),
                process=False
            )
            
            # Add UV coordinates
            if hasattr(packed_mesh, 'visual'):
                if not hasattr(packed_mesh.visual, 'uv'):
                    packed_mesh.visual = trimesh.visual.TextureVisuals()
                packed_mesh.visual.uv = uvs
            else:
                packed_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            print("UV packing completed with xatlas (10 iterations)")
            return packed_mesh
            
        except Exception as e:
            print(f"Error packing UVs with xatlas: {str(e)}")
            # Fallback: return first unwrapped part if packing fails
            if unwrapped_parts:
                print("Fallback: returning first unwrapped part")
                return unwrapped_parts[0]
            return None
    
    def combine_meshes_simple(self, mesh_list):
        """Simple mesh combination without UV processing"""
        if len(mesh_list) == 1:
            return mesh_list[0]
        
        # Combine vertices and faces keeping original positions
        all_vertices = []
        all_faces = []
        all_uvs = []
        vertex_offset = 0
        
        for mesh in mesh_list:
            # Add vertices (keep original positions)
            all_vertices.append(mesh.vertices)
            
            # Add faces with vertex offset
            offset_faces = mesh.faces + vertex_offset
            all_faces.append(offset_faces)
            
            # Add UVs if available
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                all_uvs.append(mesh.visual.uv)
            else:
                # Create dummy UVs if none exist
                dummy_uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
                all_uvs.append(dummy_uvs)
            
            vertex_offset += len(mesh.vertices)
        
        # Combine all arrays
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        combined_uvs = np.vstack(all_uvs)
        
        # Create final mesh
        final_mesh = trimesh_loader.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False  # No processing, no merge by distance
        )
        
        # Add combined UVs
        final_mesh.visual = trimesh_loader.visual.TextureVisuals(uv=combined_uvs)
        
        print(f"Combined {len(mesh_list)} parts into single mesh (no merge by distance)")
        return final_mesh
    
    def pack_uvs_with_blender(self, mesh):
        """Pack UVs using Blender with 0.005 margin and minimum stretch 10"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save mesh as OBJ with UVs
            input_obj = os.path.join(temp_dir, "input_mesh.obj")
            output_obj = os.path.join(temp_dir, "packed_mesh.obj")
            
            try:
                mesh.export(input_obj)
            except Exception as e:
                print(f"Failed to export mesh for UV packing: {e}")
                return mesh
            
            # Create Blender script for UV packing
            script_content = f"""
import bpy
import sys

try:
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import the mesh
    bpy.ops.wm.obj_import(filepath=r'{input_obj}')
    
    # Get the imported object
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Ensure we have a UV map
    if not obj.data.uv_layers:
        obj.data.uv_layers.new()
    
    # Switch to Edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # First do a basic unwrap to get proper UV coordinates
    bpy.ops.uv.unwrap(method='ANGLE_BASED')
    
    # Now pack the UV islands with specified settings
    bpy.ops.uv.pack_islands(
        margin=0.005  # 0.005 margin as requested
    )
    
    # Switch back to Object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Export with UVs
    bpy.ops.wm.obj_export(
        filepath=r'{output_obj}',
        export_selected_objects=True,
        export_uv=True,
        export_materials=False
    )
    
    print("UV packing completed successfully")
    sys.exit(0)
    
except Exception as e:
    print(f"Blender UV packing failed: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            
            # Write script to temporary file
            script_file = os.path.join(temp_dir, "pack_uvs.py")
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            try:
                # Run Blender script
                blender_exe = get_blender_path()
                cmd = [blender_exe, "--background", "--python", script_file]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    timeout=120
                )
                
                if result.stdout:
                    print(f"Blender UV packing stdout: {result.stdout.strip()}")
                if result.stderr:
                    print(f"Blender UV packing stderr: {result.stderr.strip()}")
                
                # Load the packed result
                if os.path.exists(output_obj):
                    packed_mesh = trimesh_loader.load(output_obj)
                    print("UV packing with Blender completed successfully")
                    return packed_mesh
                else:
                    print("Warning: Blender UV packing failed, no output file generated")
                    return mesh
                    
            except Exception as e:
                print(f"Error running Blender UV packing: {str(e)}")
                return mesh

class BlenderPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render_thumbnails"
    CATEGORY = "Comfy_BlenderTools/Preview"

    def _load_image_to_tensor(self, image_path):
        if not os.path.exists(image_path):
            print(f"Warning: Output image not found at {image_path}. Returning black image.")
            return torch.zeros((1, 2048, 2048, 3), dtype=torch.float32)
        
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros((1, 2048, 2048, 3), dtype=torch.float32)

    def render_thumbnails(self, trimesh):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "input_mesh.glb")
            trimesh.export(file_obj=input_mesh_path)
            
            material_output_path = os.path.join(temp_dir, "material.png")
            wireframe_output_path = os.path.join(temp_dir, "wireframe.png")
            normal_output_path = os.path.join(temp_dir, "normal.png")
            script_path = os.path.join(temp_dir, "render_script.py")

            script_params = {
                'input_mesh': repr(input_mesh_path.replace('\\', '/')),
                'material_output': repr(material_output_path.replace('\\', '/')),
                'wireframe_output': repr(wireframe_output_path.replace('\\', '/')),
                'normal_output': repr(normal_output_path.replace('\\', '/')),
            }

            blender_script = f"""
import bpy, sys, os
from math import radians
from mathutils import Vector

p = {{ {', '.join(f'"{k}": {v}' for k, v in script_params.items())} }}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.film_transparent = True
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True

    try:
        cprefs = bpy.context.preferences.addons['cycles'].preferences
        cprefs.get_devices()
        
        for device_type in ('CUDA', 'OPTIX', 'HIP', 'METAL'):
            if any(d.type == device_type for d in cprefs.devices):
                cprefs.compute_device_type = device_type
                break
        
        for device in cprefs.devices:
            if device.type == cprefs.compute_device_type:
                device.use = True

        scene.cycles.device = 'GPU'
        print(f"Successfully configured GPU for rendering with {{cprefs.compute_device_type}}.")
    except Exception as e:
        print(f"GPU rendering setup failed: {{e}}")
        print("Falling back to CPU.")
        scene.cycles.device = 'CPU'

    bpy.ops.import_scene.gltf(filepath=p['input_mesh'])
    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    
    bpy.ops.object.camera_add(location=(0,0,0))
    camera_obj = bpy.context.active_object
    scene.camera = camera_obj
    
    empty_target = bpy.data.objects.new("LookAtTarget", None)
    scene.collection.objects.link(empty_target)
    empty_target.location = (0, 0, 0)
    
    if obj:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)
        
        max_dim = max(obj.dimensions)
        
        camera_obj.location = Vector((0, -max_dim * 2.5, max_dim * 0.5))

        constraint = camera_obj.constraints.new(type='TRACK_TO')
        constraint.target = empty_target
    
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
    key_light = bpy.context.active_object
    key_light.data.energy = 1500
    key_light.data.size = 3
    key_constraint = key_light.constraints.new(type='TRACK_TO')
    key_constraint.target = empty_target
    
    bpy.ops.object.light_add(type='AREA', location=(5, -4, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 700
    fill_light.data.size = 4
    fill_constraint = fill_light.constraints.new(type='TRACK_TO')
    fill_constraint.target = empty_target

    bpy.ops.object.light_add(type='AREA', location=(-2, 5, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 1000
    rim_light.data.size = 3
    rim_constraint = rim_light.constraints.new(type='TRACK_TO')
    rim_constraint.target = empty_target

    scene.view_layers[0].material_override = None
    scene.render.filepath = p['material_output']
    bpy.ops.render.render(write_still=True)
    
    wire_mat = bpy.data.materials.new(name="WireframeMaterial")
    wire_mat.use_nodes = True
    nodes = wire_mat.node_tree.nodes
    nodes.clear()
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    wire_node = nodes.new(type='ShaderNodeWireframe')
    wire_node.inputs['Size'].default_value = 0.005
    transparent_node = nodes.new(type='ShaderNodeBsdfTransparent')
    
    wire_mat.node_tree.links.new(wire_node.outputs['Fac'], mix_shader.inputs['Fac'])
    wire_mat.node_tree.links.new(transparent_node.outputs['BSDF'], mix_shader.inputs[1])
    wire_mat.node_tree.links.new(emission_node.outputs['Emission'], mix_shader.inputs[2])
    wire_mat.node_tree.links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])
    
    scene.view_layers[0].material_override = wire_mat
    scene.render.filepath = p['wireframe_output']
    bpy.ops.render.render(write_still=True)

    normal_mat = bpy.data.materials.new(name="NormalMaterial")
    normal_mat.use_nodes = True
    nodes = normal_mat.node_tree.nodes
    nodes.clear()
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    geo_node = nodes.new(type='ShaderNodeNewGeometry')
    vec_transform_node = nodes.new(type='ShaderNodeVectorTransform')
    vec_math_scale = nodes.new(type='ShaderNodeVectorMath')
    vec_math_add = nodes.new(type='ShaderNodeVectorMath')
    vec_transform_node.vector_type = 'NORMAL'
    vec_transform_node.convert_from = 'WORLD'
    vec_transform_node.convert_to = 'OBJECT'
    vec_math_scale.operation = 'SCALE'
    vec_math_scale.inputs[1].default_value = (0.5, 0.5, 0.5)
    vec_math_add.operation = 'ADD'
    vec_math_add.inputs[1].default_value = (0.5, 0.5, 0.5)
    links = normal_mat.node_tree.links
    links.new(geo_node.outputs['Normal'], vec_transform_node.inputs['Vector'])
    links.new(vec_transform_node.outputs['Vector'], vec_math_scale.inputs[0])
    links.new(vec_math_scale.outputs['Vector'], vec_math_add.inputs[0])
    links.new(vec_math_add.outputs['Vector'], output_node.inputs['Surface'])
    
    scene.view_layers[0].material_override = normal_mat
    scene.render.filepath = p['normal_output']
    bpy.ops.render.render(write_still=True)
    
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(blender_script)

            _run_blender_script(script_path)

            material_img = self._load_image_to_tensor(material_output_path)
            wireframe_img = self._load_image_to_tensor(wireframe_output_path)
            normal_img = self._load_image_to_tensor(normal_output_path)
            
            batched_images = torch.cat((material_img, wireframe_img, normal_img), 0)

            return (batched_images,)
