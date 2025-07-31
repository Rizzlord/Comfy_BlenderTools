import os
import subprocess
import tempfile
import trimesh as tm  # Import with an alias to prevent name conflicts
import numpy as np

def get_blender_path():
    """
    Finds the Blender executable path from the BLENDER_EXE environment variable
    with a fallback to a default path. Logs the result.
    """
    blender_path = os.environ.get("BLENDER_EXE")
    
    if blender_path and os.path.isfile(blender_path):
        print(f"INFO: Found Blender executable via BLENDER_EXE environment variable: {blender_path}")
        return blender_path
    
    if blender_path:
        print(f"WARNING: BLENDER_EXE environment variable was found, but the path '{blender_path}' is not a valid file.")
    else:
        print(f"INFO: BLENDER_EXE environment variable not set. Falling back to default path.")

    # Fallback path if environment variable is not set or invalid
    fallback_path = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
    
    if not os.path.isfile(fallback_path):
        raise FileNotFoundError(
            f"Blender executable not found at the default path: {fallback_path}. "
            "Please set the BLENDER_EXE environment variable to the correct path of your blender.exe."
        )
    
    print(f"INFO: Using fallback Blender executable path: {fallback_path}")
    return fallback_path

class MeshCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the Mesh Cleanup node.
        """
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "apply_smoothing": ("BOOLEAN", {"default": False}),
                "factor": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.1}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 100}),
                "axis_x": ("BOOLEAN", {"default": True}),
                "axis_y": ("BOOLEAN", {"default": True}),
                "axis_z": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "cleanup_and_smooth_mesh"
    CATEGORY = "Comfy_BlenderTools"

    def cleanup_and_smooth_mesh(self, trimesh, apply_smoothing, factor, repeat, axis_x, axis_y, axis_z):
        """
        This function safely isolates the largest connected component of a mesh.
        Optionally, it can then apply Blender's Smooth modifier to it.
        """
        # Use the alias 'tm' to refer to the trimesh library
        if not isinstance(trimesh, tm.Trimesh) or len(trimesh.vertices) == 0 or len(trimesh.faces) == 0:
            raise ValueError("Input to Mesh Cleanup is not a valid or non-empty mesh. Aborting.")

        # 1. Use trimesh to safely find the largest connected component (island)
        # This is a method on the mesh object, so it's correct as `trimesh.split`
        components = trimesh.split(only_watertight=False)
        
        if not components:
            raise ValueError("Mesh could not be split into components. It might be empty or invalid.")
            
        # Find the largest component by the number of faces
        largest_component = max(components, key=lambda component: len(component.faces))

        # If smoothing is not requested, the cleanup is done. Return the result.
        if not apply_smoothing:
            return (largest_component,)

        # 2. If smoothing is requested, send the largest component to Blender.
        blender_path = get_blender_path()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "input_mesh.obj")
            output_mesh_path = os.path.join(temp_dir, "output_mesh.obj")
            script_path = os.path.join(temp_dir, "cleanup_script.py")
            largest_component.export(file_obj=input_mesh_path)

            # The Blender script only runs if smoothing is needed.
            script = f"""
import bpy
import sys

params = {{
    'input_mesh': r'{input_mesh_path}',
    'output_mesh': r'{output_mesh_path}',
    'factor': {factor},
    'repeat': {repeat},
    'axis_x': {axis_x},
    'axis_y': {axis_y},
    'axis_z': {axis_z},
}}

try:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=params['input_mesh'])
    imported_obj = bpy.context.view_layer.objects.active

    if imported_obj and imported_obj.type == 'MESH' and imported_obj.data.polygons:
        smooth_mod = imported_obj.modifiers.new(name="Smooth", type='SMOOTH')
        smooth_mod.factor = params['factor']
        smooth_mod.iterations = params['repeat']
        smooth_mod.use_x = params['axis_x']
        smooth_mod.use_y = params['axis_y']
        smooth_mod.use_z = params['axis_z']
        bpy.ops.object.modifier_apply(modifier=smooth_mod.name)

    bpy.ops.wm.obj_export(filepath=params['output_mesh'], export_uv=True, export_normals=True, export_materials=False)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)

            try:
                subprocess.run(
                    [blender_path, '--factory-startup', '--background', '--python', script_path],
                    check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Blender execution failed. Stderr: {e.stderr}")
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred while running Blender: {e}")

            # 3. Load the result and perform a robust safety check.
            # Use the alias 'tm' to refer to the trimesh library
            loaded_geom = tm.load(output_mesh_path, process=False)
            
            if isinstance(loaded_geom, tm.Scene):
                if not loaded_geom.geometry:
                    raise ValueError("Blender process resulted in an empty scene.")
                # Use the alias 'tm' to refer to the trimesh library
                final_mesh = tm.util.concatenate(list(loaded_geom.geometry.values()))
            else:
                final_mesh = loaded_geom
            
            # Use the alias 'tm' to refer to the trimesh library
            if not isinstance(final_mesh, tm.Trimesh) or len(final_mesh.vertices) == 0 or len(final_mesh.faces) == 0:
                raise ValueError("Mesh Cleanup process resulted in an empty or invalid mesh.")

            return (final_mesh,)