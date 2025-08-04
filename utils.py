import os
import subprocess
import sys
import tempfile
import trimesh as trimesh_loader
import folder_paths
import numpy as np

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
p = {{ {", ".join(f'"{k}": {v}' for k, v in script_params.items())} }}
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