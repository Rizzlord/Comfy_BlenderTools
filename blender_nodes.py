import os
import tempfile
import subprocess
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path
import math
from .utils import _run_blender_script, get_blender_clean_mesh_func_script, get_mof_path, _run_mof_command

class MinistryOfFlatUnwrap:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input parameters for the Ministry of Flat Unwrap node
        based on the provided documentation and settings image.
        """
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "texture_resolution": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 256}),
                "separate_hard_edges": ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "display": "number"}),
                "use_normals": ("BOOLEAN", {"default": False}),
                "udims": ("INT", {"default": 1, "min": 1, "max": 100}),
                "overlap_identical": ("BOOLEAN", {"default": False}),
                "overlap_mirrored": ("BOOLEAN", {"default": False}),
                "world_space_uvs": ("BOOLEAN", {"default": False}),
                "texture_density": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "island_margin": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.001, "display": "number"}),
                "refine_with_minimum_stretch": ("BOOLEAN", {"default": False}),
                "min_stretch_iterations": ("INT", {"default": 10, "min": 0, "max": 256}),
                "export_uv_layout": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("trimesh", "uv_layout_preview")
    FUNCTION = "unwrap"
    CATEGORY = "Comfy_BlenderTools"

    def unwrap(self, trimesh, texture_resolution, separate_hard_edges, aspect_ratio, use_normals, udims, overlap_identical, overlap_mirrored, world_space_uvs, texture_density, island_margin, refine_with_minimum_stretch, min_stretch_iterations, export_uv_layout):
        mof_exe_path = get_mof_path()
        if not mof_exe_path:
            print("Ministry of Flat executable not found. Please set the MOF_EXE environment variable.")
            empty_image = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
            return (trimesh, empty_image)

        with tempfile.TemporaryDirectory() as temp_dir:
            initial_input_path = os.path.join(temp_dir, "initial_i.obj")
            cleaned_input_path = os.path.join(temp_dir, "cleaned_i.obj")
            mof_output_path = os.path.join(temp_dir, "mof_o.obj")
            final_output_path = os.path.join(temp_dir, "final_o.obj")
            
            trimesh.export(file_obj=initial_input_path)

            clean_mesh_func_script = get_blender_clean_mesh_func_script()
            pre_script_path = os.path.join(temp_dir, "pre_clean.py")
            pre_script = f"""
{clean_mesh_func_script}
import bpy, sys
p = {{'in_obj': r'{initial_input_path}', 'out_obj': r'{cleaned_input_path}', 'merge_dist': 0.0001}}
try:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.wm.obj_import(filepath=p['in_obj'])
    obj = bpy.context.view_layer.objects.active
    if obj:
        clean_mesh(obj, p['merge_dist'])
        bpy.ops.wm.obj_export(filepath=p['out_obj'], export_uv=True, export_normals=True, export_materials=False)
    sys.exit(0)
except Exception as e: 
    print(f"Blender pre-clean script failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            with open(pre_script_path, 'w') as f: f.write(pre_script)
            _run_blender_script(pre_script_path)

            if not os.path.exists(cleaned_input_path) or os.path.getsize(cleaned_input_path) == 0:
                raise RuntimeError("Blender pre-clean script failed to produce a valid output OBJ file.")

            command = [
                mof_exe_path,
                cleaned_input_path,
                mof_output_path,
                "-RESOLUTION", str(texture_resolution),
                "-SEPARATE", str(separate_hard_edges).upper(),
                "-ASPECT", str(aspect_ratio),
                "-NORMALS", str(use_normals).upper(),
                "-UDIMS", str(udims),
                "-OVERLAP", str(overlap_identical).upper(),
                "-MIRROR", str(overlap_mirrored).upper(),
                "-WORLDSCALE", str(world_space_uvs).upper(),
                "-DENSITY", str(texture_density),
                "-SUPRESS", "TRUE"
            ]
            
            mof_exe_dir = os.path.dirname(mof_exe_path)
            result = subprocess.run(
                command,
                capture_output=True, text=True,
                cwd=mof_exe_dir
            )

            if result.stdout:
                print(f"Ministry of Flat stdout: {result.stdout}")
            if result.stderr:
                print(f"Ministry of Flat stderr: {result.stderr}")

            if not os.path.exists(mof_output_path) or os.path.getsize(mof_output_path) == 0:
                raise RuntimeError("Ministry of Flat failed to create an output file. Check logs for details.")

            post_script_path = os.path.join(temp_dir, "post_clean.py")
            post_script = f"""
{clean_mesh_func_script}
import bpy, sys, os
p = {{'in_obj': r'{mof_output_path}', 'out_obj': r'{final_output_path}', 'merge_dist': 0.0001, 'island_margin': {island_margin}, 'refine_stretch': {refine_with_minimum_stretch}, 'stretch_iterations': {min_stretch_iterations}}}
try:
    if not os.path.exists(p['in_obj']) or os.path.getsize(p['in_obj']) == 0:
        print(f"MoF output file is missing or empty, cannot perform post-clean.", file=sys.stderr)
        sys.exit(1)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.wm.obj_import(filepath=p['in_obj'])
    obj = bpy.context.view_layer.objects.active
    if obj:
        clean_mesh(obj, p['merge_dist'])
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT')

        if p['refine_stretch']:
            bpy.ops.uv.seams_from_islands()
            bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', iterations=p['stretch_iterations'])

        bpy.ops.uv.pack_islands(margin=p['island_margin'])
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.wm.obj_export(filepath=p['out_obj'], export_uv=True, export_normals=True, export_materials=False)
    sys.exit(0)
except Exception as e: 
    print(f"Blender post-clean script failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            with open(post_script_path, 'w') as f: f.write(post_script)
            _run_blender_script(post_script_path)

            if not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0:
                raise RuntimeError("Blender post-clean script failed to produce a valid final output file.")

            processed_mesh = trimesh_loader.load(final_output_path, process=False)
            
            uv_preview = self.generate_uv_preview(processed_mesh, texture_resolution, texture_resolution, export_uv_layout)
            
            return (processed_mesh, uv_preview)

    def generate_uv_preview(self, mesh, res_x, res_y, export_layout):
        uv_layout_image = torch.zeros((1, res_y, res_x, 3), dtype=torch.float32)
        if export_layout and hasattr(mesh.visual, 'uv') and len(mesh.visual.uv) > 0:
            img = Image.new('RGB', (res_x, res_y), 'black')
            draw = ImageDraw.Draw(img)
            if mesh.faces.shape[1] == 3:
                for face in mesh.faces:
                    points = [(mesh.visual.uv[i][0] * res_x, (1 - mesh.visual.uv[i][1]) * res_y) for i in face]
                    points.append(points[0])
                    draw.line(points, fill='white', width=1)
            uv_layout_image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
        return uv_layout_image
    
class BlenderDecimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "apply_decimation": ("BOOLEAN", {"default": True}),
                "max_face_count": ("INT", {"default": 10000, "min": 100, "max": 10000000, "step": 100}),
                "triangulate": ("BOOLEAN", {"default": True}),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "decimate"
    CATEGORY = "Comfy_BlenderTools"

    def decimate(self, trimesh, apply_decimation, max_face_count, triangulate, merge_distance):
        current_faces = len(trimesh.faces)
        ratio = max_face_count / current_faces if current_faces > 0 else 1.0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.glb")
            output_mesh_path = os.path.join(temp_dir, "o.glb")
            script_path = os.path.join(temp_dir, "s.py")
            
            trimesh.export(file_obj=input_mesh_path)

            params = {
                'i': input_mesh_path, 'o': output_mesh_path, 
                'apply_dec': apply_decimation, 'ratio': min(1.0, ratio), 
                'tri': triangulate, 'merge_dist': merge_distance
            }

            clean_mesh_func_script = get_blender_clean_mesh_func_script()

            script = f"""
{clean_mesh_func_script}
import bpy, sys
p = {{
    'i': r"{params['i']}", 'o': r"{params['o']}", 'apply_dec': {params['apply_dec']}, 
    'ratio': {params['ratio']}, 'tri': {params['tri']}, 'merge_dist': {params['merge_dist']}
}}
try:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
        
    bpy.ops.import_scene.gltf(filepath=p['i'])
    
    mesh_obj = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break
    
    if mesh_obj is None:
        raise Exception("Script Error: No mesh was found after importing the GLB.")

    obj = mesh_obj
    clean_mesh(obj, p['merge_dist'])

    if p['apply_dec'] and p['ratio'] < 1.0:
        mod = obj.modifiers.new(name="DecimateMod", type='DECIMATE')
        mod.ratio = p['ratio']
        mod.use_collapse_triangulate = p['tri']
        bpy.ops.object.modifier_apply(modifier=mod.name)
    
    bpy.ops.export_scene.gltf(filepath=p['o'], export_format='GLB', use_selection=True)
    
    sys.exit(0)
except Exception as e: 
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f: f.write(script)
            _run_blender_script(script_path)
            
            processed_mesh = trimesh_loader.load(output_mesh_path, force="mesh")
            return (processed_mesh,)

class BlenderUnwrap:
    UNWRAP_METHODS = ["XAtlas UV Atlas", "Smart UV Project", "Cube Projection"]
    TEXTURE_RESOLUTIONS = ["512", "768", "1024", "1536", "2048", "4096", "8192"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "unwrap_method": (cls.UNWRAP_METHODS, {"default": "Smart UV Project"}),
                "export_uv_layout": ("BOOLEAN", {"default": True}),
                "texture_resolution": (cls.TEXTURE_RESOLUTIONS, {"default": "1024"}),
                "pixel_margin": ("INT", {"default": 0, "min": 0, "max": 64}),
                "angle_limit": ("FLOAT", {"default": 66.0, "min": 0.0, "max": 90.0, "step": 0.1}),
                "refine_with_minimum_stretch": ("BOOLEAN", {"default": False}),
                "min_stretch_iterations": ("INT", {"default": 10, "min": 0, "max": 256}),
                "final_merge_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
                "correct_aspect": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("trimesh", "uv_layout_preview")
    FUNCTION = "unwrap"
    CATEGORY = "Comfy_BlenderTools"

    def unwrap(self, trimesh, **kwargs):
        unwrap_method = kwargs.get("unwrap_method")

        if unwrap_method == "XAtlas UV Atlas":
            return self.process_with_xatlas(trimesh, **kwargs)
        else:
            return self.process_with_blender(trimesh, **kwargs)

    def _get_post_process_script_block(self):
        return """
    if p['refine_s']:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.seams_from_islands()
        bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', iterations=p['min_stretch_i'], correct_aspect=p['correct_a'], margin=p['margin'])
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.pack_islands(margin=p['margin'])
        bpy.ops.object.mode_set(mode='OBJECT')

    if p['final_merge_dist'] > 0.0:
        clean_mesh(obj, p['final_merge_dist'])
"""

    def process_with_blender(self, trimesh, **p):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.obj")
            output_mesh_path = os.path.join(temp_dir, "o.obj")
            script_path = os.path.join(temp_dir, "s.py")
            trimesh.export(file_obj=input_mesh_path)
            
            post_process_script = self._get_post_process_script_block()
            clean_mesh_func_script = get_blender_clean_mesh_func_script()
            
            params = {
                'i': input_mesh_path, 'o': output_mesh_path, 
                'unwrap_m': p.get('unwrap_method'), 'refine_s': p.get('refine_with_minimum_stretch'),
                'angle_l': p.get('angle_limit', 66.0) * (math.pi / 180.0), 
                'margin': p.get('pixel_margin', 0) / int(p.get('texture_resolution', 1024)),
                'correct_a': p.get('correct_aspect'), 'min_stretch_i': p.get('min_stretch_iterations'), 
                'final_merge_dist': p.get('final_merge_distance')
            }

            script = f"""
{clean_mesh_func_script}
import bpy, sys
p = {{
    'i': r"{params['i']}", 'o': r"{params['o']}", 'unwrap_m': "{params['unwrap_m']}", 
    'refine_s': {params['refine_s']}, 'angle_l': {params['angle_l']}, 'margin': {params['margin']},
    'correct_a': {params['correct_a']}, 'min_stretch_i': {params['min_stretch_i']}, 
    'final_merge_dist': {params['final_merge_dist']}
}}
try:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.obj_import(filepath=p['i'])
    obj = bpy.context.view_layer.objects.active; 
    bpy.context.view_layer.objects.active = obj; 
    obj.select_set(True); m = obj.data

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.mark_sharp(clear=True); 
    bpy.ops.mesh.mark_seam(clear=True)

    if p['unwrap_m'] == 'Smart UV Project':
        bpy.ops.uv.smart_project(angle_limit=p['angle_l'], 
        island_margin=p['margin'], 
        correct_aspect=p['correct_a'], 
        scale_to_bounds=True)

    elif p['unwrap_m'] == 'Cube Projection':
        bpy.ops.uv.cube_project(cube_size=1.0, 
        correct_aspect=p['correct_a'], 
        scale_to_bounds=False)

    bpy.ops.uv.select_all(action='SELECT'); bpy.ops.uv.pack_islands(margin=p['margin'])
    bpy.ops.object.mode_set(mode='OBJECT')
    {post_process_script}
    bpy.ops.wm.obj_export(filepath=p['o'], 
    export_uv=True, export_normals=True, 
    export_materials=False)
    sys.exit(0)
except Exception as e: 
    print(f"Blender script failed: {{e}}", 
    file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f: f.write(script)
            _run_blender_script(script_path)

            processed_mesh = trimesh_loader.load(output_mesh_path, process=False)
            uv_preview = self.generate_uv_preview(processed_mesh, int(p.get('texture_resolution')), int(p.get('texture_resolution')), p.get('export_uv_layout'))
            return (processed_mesh, uv_preview)

    def process_with_xatlas(self, trimesh, **p):
        try:
            import xatlas
        except ImportError:
            raise ImportError("xatlas not installed. Please run: pip install xatlas")

        with tempfile.TemporaryDirectory() as temp_dir:
            vmapping, indices, uvs = xatlas.parametrize(trimesh.vertices, trimesh.faces)
            unwrapped_mesh = trimesh_loader.Trimesh(
                vertices=trimesh.vertices[vmapping], 
                faces=indices, 
                visual=trimesh_loader.visual.TextureVisuals(uv=uvs), 
                process=False
            )
            
            if not p.get('refine_with_minimum_stretch') and not p.get('final_merge_dist') > 0.0:
                uv_preview = self.generate_uv_preview(unwrapped_mesh, int(p.get('texture_resolution')), int(p.get('texture_resolution')), p.get('export_uv_layout'))
                return (unwrapped_mesh, uv_preview)
            
            refine_input = os.path.join(temp_dir, "refine_in.obj")
            final_out = os.path.join(temp_dir, "o.obj")
            unwrapped_mesh.export(file_obj=refine_input)

            post_process_script = self._get_post_process_script_block()
            clean_mesh_func_script = get_blender_clean_mesh_func_script()
            
            post_params = {
                'i': refine_input, 'o': final_out, 
                'refine_s': p.get('refine_with_minimum_stretch'),
                'margin': p.get('pixel_margin', 0) / int(p.get('texture_resolution', 1024)), 
                'correct_a': p.get('correct_aspect'), 'min_stretch_i': p.get('min_stretch_iterations'), 
                'final_merge_dist': p.get('final_merge_distance')
            }
            post_script = f"""
{clean_mesh_func_script}
import bpy, sys
p = {{
    'i': r"{post_params['i']}", 'o': r"{post_params['o']}", 'refine_s': {post_params['refine_s']},
    'margin': {post_params['margin']}, 'correct_a': {post_params['correct_a']}, 
    'min_stretch_i': {post_params['min_stretch_i']}, 'final_merge_dist': {post_params['final_merge_dist']}
}}
try:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.obj_import(filepath=p['i'])
    obj = bpy.context.view_layer.objects.active
    {post_process_script}
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.obj_export(filepath=p['o'], export_uv=True, export_normals=True, export_materials=False)
    sys.exit(0)
except Exception as e: 
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            script_path = os.path.join(temp_dir, "post.py")
            with open(script_path, 'w') as f: f.write(post_script)
            _run_blender_script(script_path)

            final_mesh = trimesh_loader.load(final_out, process=False)
            uv_preview = self.generate_uv_preview(final_mesh, int(p.get('texture_resolution')), int(p.get('texture_resolution')), p.get('export_uv_layout'))
            return (final_mesh, uv_preview)

    def generate_uv_preview(self, mesh, res_x, res_y, export_layout):
        uv_layout_image = torch.zeros((1, res_y, res_x, 3), dtype=torch.float32)
        if export_layout and hasattr(mesh.visual, 'uv') and len(mesh.visual.uv) > 0:
            img = Image.new('RGB', (res_x, res_y), 'black')
            draw = ImageDraw.Draw(img)
            if mesh.faces.shape[1] == 3:
                for face in mesh.faces:
                    points = [(mesh.visual.uv[i][0] * res_x, (1 - mesh.visual.uv[i][1]) * res_y) for i in face]
                    points.append(points[0])
                    draw.line(points, fill='white', width=1)
            uv_layout_image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
        return uv_layout_image

class BlenderExportGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "output_path": ("STRING", {"default": "blender_exports"}),
                "filename_prefix": ("STRING", {"default": "ExportedMesh"}),
                "merge_distance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "display": "number"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export_glb"
    CATEGORY = "Comfy_BlenderTools"
    OUTPUT_NODE = True

    def export_glb(self, trimesh, output_path, filename_prefix, merge_distance):
        if not os.path.isabs(output_path):
            final_output_dir = os.path.join(folder_paths.get_output_directory(), output_path)
        else:
            final_output_dir = output_path
        
        os.makedirs(final_output_dir, exist_ok=True)
        
        base_filepath = os.path.join(final_output_dir, filename_prefix)
        counter = 1
        final_glb_path = f"{base_filepath}_{counter:05}_.glb"
        while os.path.exists(final_glb_path):
            counter += 1
            final_glb_path = f"{base_filepath}_{counter:05}_.glb"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_glb_path = os.path.join(temp_dir, "temp.glb")
            trimesh.export(temp_glb_path)

            script_path = os.path.join(temp_dir, "clean_and_export.py")
            clean_mesh_func_script = get_blender_clean_mesh_func_script()
            
            script = f"""
{clean_mesh_func_script}
import bpy, sys
p = {{'in_glb': r'{temp_glb_path}', 'out_glb': r'{final_glb_path}', 'merge_dist': {merge_distance}}}
try:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
        
    bpy.ops.import_scene.gltf(filepath=p['in_glb'])
    
    mesh_obj = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break
            
    if mesh_obj is None:
        raise Exception("Script Error: No mesh object was found after importing the GLB.")
        
    clean_mesh(mesh_obj, p['merge_dist'])
    
    bpy.ops.export_scene.gltf(filepath=p['out_glb'], export_format='GLB', use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr); sys.exit(1)
"""
            with open(script_path, 'w') as f: f.write(script)
            _run_blender_script(script_path)

        return (final_glb_path,)