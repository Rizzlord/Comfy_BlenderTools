import os
import tempfile
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path
import math
from .utils import _run_blender_script, get_blender_clean_mesh_func_script

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

class ApplyTexturesToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "trimesh": ("TRIMESH",), },
            "optional": {
                "baked_maps": ("BAKED_MAPS",),
                "albedo_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "ao_map": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "apply_textures"
    CATEGORY = "Comfy_BlenderTools"

    def _generate_checkerboard_texture(self, size=1024, square_size=64):
        c = np.indices((size, size)) // square_size
        pattern = (c[0] + c[1]) % 2
        img_array = np.zeros((size, size, 3), dtype=np.uint8)
        black = np.array([0, 0, 0], dtype=np.uint8)
        magenta = np.array([255, 0, 255], dtype=np.uint8)
        img_array[pattern == 0] = black
        img_array[pattern == 1] = magenta
        checker_pil = Image.fromarray(img_array, 'RGB')
        checker_np = np.array(checker_pil).astype(np.float32) / 255.0
        return torch.from_numpy(checker_np)[None,]

    def apply_textures(self, trimesh, baked_maps=None, albedo_map=None, normal_map=None, ao_map=None):
        final_maps = {}
        if baked_maps:
            final_maps.update(baked_maps)
        
        if albedo_map is not None: final_maps['albedo'] = albedo_map
        if normal_map is not None: final_maps['normal'] = normal_map
        if ao_map is not None: final_maps['ao'] = ao_map

        textured_mesh = trimesh.copy()

        if not isinstance(textured_mesh.visual, trimesh_loader.visual.TextureVisuals):
            uvs = getattr(textured_mesh.visual, 'uv', None)
            textured_mesh.visual = trimesh_loader.visual.TextureVisuals(uv=uvs)

        if not hasattr(textured_mesh.visual, 'material') or not textured_mesh.visual.material:
            textured_mesh.visual.material = trimesh_loader.visual.material.PBRMaterial()
        
        material = textured_mesh.visual.material

        has_existing_albedo = hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None
        if 'albedo' not in final_maps and not has_existing_albedo:
            print("INFO: No albedo map provided or found. Generating a fallback checkerboard texture.")
            final_maps['albedo'] = self._generate_checkerboard_texture()

        if not final_maps:
            return (textured_mesh,)

        def tensor_to_pil(tensor):
            if tensor is None: return None
            return Image.fromarray((tensor[0].cpu().numpy() * 255).astype(np.uint8))

        if 'albedo' in final_maps: material.baseColorTexture = tensor_to_pil(final_maps['albedo'])
        if 'normal' in final_maps: material.normalTexture = tensor_to_pil(final_maps['normal'])
        if 'ao' in final_maps: material.occlusionTexture = tensor_to_pil(final_maps['ao'])
        
        return (textured_mesh,)

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