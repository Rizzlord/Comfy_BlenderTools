import os
import tempfile
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path
import math
from .utils import _run_blender_script

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

            script = f"""
import bpy, sys
p = {{
    'i': r"{params['i']}", 'o': r"{params['o']}", 'apply_dec': {params['apply_dec']}, 
    'ratio': {params['ratio']}, 'tri': {params['tri']}, 'merge_dist': {params['merge_dist']}
}}
try:
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=p['i'])
    
    mesh_obj = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break
    
    if mesh_obj is None:
        raise Exception("Script Error: No mesh was found after importing the GLB.")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    obj = mesh_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if p['merge_dist'] > 0.0:
        bpy.ops.mesh.remove_doubles(threshold=p['merge_dist'])
    
    bpy.ops.object.mode_set(mode='OBJECT')
    if p['apply_dec'] and p['ratio'] < 1.0:
        mod = obj.modifiers.new(name="DecimateMod", type='DECIMATE')
        mod.ratio = p['ratio']
        mod.use_collapse_triangulate = p['tri']
        bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.ops.object.shade_smooth()
    
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
    if p['refine_s'] or p['final_merge_dist'] > 0.0:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        if p['refine_s']:
            bpy.ops.uv.seams_from_islands()
            bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', iterations=p['min_stretch_i'], correct_aspect=p['correct_a'], margin=p['margin'])
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.pack_islands(margin=p['margin'])
        if p['final_merge_dist'] > 0.0:
            try:
                bpy.ops.uv.seams_from_islands()
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.mesh.select_mode(type='EDGE')
                bpy.ops.mesh.select_similar(type='EDGE_SEAM')
                bpy.ops.mesh.select_mode(type='VERT')
                bpy.ops.mesh.select_all(action='INVERT')
                bpy.ops.mesh.remove_doubles(threshold=p['final_merge_dist'], use_unselected=True)
            except RuntimeError as e:
                print(f"Warning: Could not perform UV-aware final merge. Reason: {e}")
            finally:
                bpy.ops.mesh.select_all(action='SELECT')
"""

    def process_with_blender(self, trimesh, **p):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "i.obj")
            output_mesh_path = os.path.join(temp_dir, "o.obj")
            script_path = os.path.join(temp_dir, "s.py")
            trimesh.export(file_obj=input_mesh_path)
            
            post_process_script = self._get_post_process_script_block()
            
            params = {
                'i': input_mesh_path, 'o': output_mesh_path, 
                'unwrap_m': p.get('unwrap_method'), 'refine_s': p.get('refine_with_minimum_stretch'),
                'angle_l': p.get('angle_limit', 66.0) * (math.pi / 180.0), 
                'margin': p.get('pixel_margin', 0) / int(p.get('texture_resolution', 1024)),
                'correct_a': p.get('correct_aspect'), 'min_stretch_i': p.get('min_stretch_iterations'), 
                'final_merge_dist': p.get('final_merge_distance')
            }

            script = f"""
import bpy, sys
p = {{
    'i': r"{params['i']}", 'o': r"{params['o']}", 'unwrap_m': "{params['unwrap_m']}", 
    'refine_s': {params['refine_s']}, 'angle_l': {params['angle_l']}, 'margin': {params['margin']},
    'correct_a': {params['correct_a']}, 'min_stretch_i': {params['min_stretch_i']}, 
    'final_merge_dist': {params['final_merge_dist']}
}}
try:
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=p['i'])
    obj = bpy.context.view_layer.objects.active; bpy.context.view_layer.objects.active = obj; obj.select_set(True); m = obj.data

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.mark_sharp(clear=True); bpy.ops.mesh.mark_seam(clear=True)

    if p['unwrap_m'] == 'Smart UV Project':
        bpy.ops.uv.smart_project(angle_limit=p['angle_l'], island_margin=p['margin'], correct_aspect=p['correct_a'], scale_to_bounds=True)
    elif p['unwrap_m'] == 'Cube Projection':
        bpy.ops.uv.cube_project(cube_size=1.0, correct_aspect=p['correct_a'], scale_to_bounds=False)

    bpy.ops.uv.select_all(action='SELECT'); bpy.ops.uv.pack_islands(margin=p['margin'])
    {post_process_script}
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.obj_export(filepath=p['o'], export_uv=True, export_normals=True, export_materials=False)
    sys.exit(0)
except Exception as e: 
    print(f"Blender script failed: {{e}}", file=sys.stderr)
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
            
            if not p.get('refine_with_minimum_stretch') and not p.get('final_merge_distance') > 0.0:
                uv_preview = self.generate_uv_preview(unwrapped_mesh, int(p.get('texture_resolution')), int(p.get('texture_resolution')), p.get('export_uv_layout'))
                return (unwrapped_mesh, uv_preview)
            
            refine_input = os.path.join(temp_dir, "refine_in.obj")
            final_out = os.path.join(temp_dir, "o.obj")
            unwrapped_mesh.export(file_obj=refine_input)

            post_process_script = self._get_post_process_script_block()
            post_params = {
                'i': refine_input, 'o': final_out, 
                'refine_s': p.get('refine_with_minimum_stretch'),
                'margin': p.get('pixel_margin', 0) / int(p.get('texture_resolution', 1024)), 
                'correct_a': p.get('correct_aspect'), 'min_stretch_i': p.get('min_stretch_iterations'), 
                'final_merge_dist': p.get('final_merge_distance')
            }
            post_script = f"""
import bpy, sys
p = {{
    'i': r"{post_params['i']}", 'o': r"{post_params['o']}", 'refine_s': {post_params['refine_s']},
    'margin': {post_params['margin']}, 'correct_a': {post_params['correct_a']}, 
    'min_stretch_i': {post_params['min_stretch_i']}, 'final_merge_dist': {post_params['final_merge_dist']}
}}
try:
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=p['i'])
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

class TextureBake:
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "low_poly_mesh": ("TRIMESH",),
                "bake_normal": ("BOOLEAN", {"default": True}),
                "bake_ao": ("BOOLEAN", {"default": True}),
                "resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "cage_extrusion": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("BAKED_MAPS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("baked_maps", "normal_map_preview", "ao_map_preview")
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools"

    def bake(self, high_poly_mesh, low_poly_mesh, bake_normal, bake_ao, resolution, cage_extrusion, margin):
        if not bake_normal and not bake_ao:
            raise ValueError("No bake type selected. Please enable Normal or AO baking.")

        with tempfile.TemporaryDirectory() as temp_dir:
            high_poly_path = os.path.join(temp_dir, "high.glb")
            low_poly_path = os.path.join(temp_dir, "low.glb")
            script_path = os.path.join(temp_dir, "s.py")

            high_poly_mesh.export(file_obj=high_poly_path)
            low_poly_mesh.export(file_obj=low_poly_path)

            params = {
                'high_poly_path': high_poly_path,
                'low_poly_path': low_poly_path,
                'temp_dir': temp_dir,
                'bake_normal': bake_normal,
                'bake_ao': bake_ao,
                'resolution': int(resolution),
                'cage_extrusion': cage_extrusion,
                'margin': margin
            }

            script = f"""
import bpy, sys, os, traceback
p = {{
    'high_poly_path': r"{params['high_poly_path']}",
    'low_poly_path': r"{params['low_poly_path']}",
    'temp_dir': r"{params['temp_dir']}",
    'bake_normal': {params['bake_normal']},
    'bake_ao': {params['bake_ao']},
    'resolution': {params['resolution']},
    'cage_extrusion': {params['cage_extrusion']},
    'margin': {params['margin']}
}}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.import_scene.gltf(filepath=p['high_poly_path'])
    high_obj = next((obj for obj in bpy.context.selected_objects if obj.type == 'MESH'), None)
    if not high_obj: raise Exception("No mesh found in high-poly file.")
    high_obj.name = "HighPoly"

    bpy.ops.import_scene.gltf(filepath=p['low_poly_path'])
    low_obj = next((obj for obj in bpy.context.selected_objects if obj.type == 'MESH'), None)
    if not low_obj: raise Exception("No mesh found in low-poly file.")
    low_obj.name = "LowPoly"

    if not low_obj.data.uv_layers:
        raise Exception("Low-poly mesh has no UV map. Please use the BlenderUnwrap node first.")

    mat = bpy.data.materials.new(name="BakeMaterial")
    mat.use_nodes = True
    if len(low_obj.data.materials) > 0:
        low_obj.data.materials[0] = mat
    else:
        low_obj.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    tex_node = nodes.new('ShaderNodeTexImage')
    nodes.active = tex_node

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'

    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True)
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj

    def do_bake(bake_type_internal, image_name):
        bake_res = p['resolution']
        bake_image = bpy.data.images.new(name=image_name, width=bake_res, height=bake_res, alpha=True)
        tex_node.image = bake_image

        bpy.ops.object.bake(
            type=bake_type_internal,
            use_selected_to_active=True,
            margin=p['margin'],
            cage_extrusion=p['cage_extrusion'],
            use_clear=True,
            normal_space='TANGENT' if bake_type_internal == 'NORMAL' else 'OBJECT'
        )

        output_path = os.path.join(p['temp_dir'], image_name + ".png")
        bake_image.filepath_raw = output_path
        bake_image.file_format = 'PNG'
        bake_image.save()
        bpy.data.images.remove(bake_image)

    if p['bake_normal']:
        do_bake('NORMAL', 'normal_map')

    if p['bake_ao']:
        do_bake('AO', 'ao_map')
    
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, 'w') as f:
                f.write(script)
            
            _run_blender_script(script_path)
            
            baked_maps = {}
            dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            normal_map_tensor = dummy_image
            ao_map_tensor = dummy_image

            def load_image_as_tensor(path):
                if not os.path.exists(path):
                    return None
                img = Image.open(path).convert('RGB')
                return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

            if bake_normal:
                normal_path = os.path.join(temp_dir, "normal_map.png")
                tensor = load_image_as_tensor(normal_path)
                if tensor is not None:
                    baked_maps['normal'] = tensor
                    normal_map_tensor = tensor
            
            if bake_ao:
                ao_path = os.path.join(temp_dir, "ao_map.png")
                tensor = load_image_as_tensor(ao_path)
                if tensor is not None:
                    baked_maps['ao'] = tensor
                    ao_map_tensor = tensor

            return (baked_maps, normal_map_tensor, ao_map_tensor)


class ApplyAndExportMaps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "TexturedMesh"}),
            },
            "optional": {
                "baked_maps": ("BAKED_MAPS",),
                "albedo_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "ao_map": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "glb_path")
    FUNCTION = "apply_and_export"
    CATEGORY = "Comfy_BlenderTools"
    OUTPUT_NODE = True

    def apply_and_export(self, trimesh, filename_prefix, baked_maps=None, albedo_map=None, normal_map=None, ao_map=None):
        final_maps = {}
        if baked_maps:
            final_maps.update(baked_maps)
        
        if albedo_map is not None:
            final_maps['albedo'] = albedo_map
        if normal_map is not None:
            final_maps['normal'] = normal_map
        if ao_map is not None:
            final_maps['ao'] = ao_map

        if not final_maps:
            raise ValueError("No maps provided to apply. Connect either baked_maps or individual map inputs.")

        textured_mesh = trimesh.copy()

        if not isinstance(textured_mesh.visual, trimesh_loader.visual.TextureVisuals):
            uvs = getattr(trimesh.visual, 'uv', None)
            textured_mesh.visual = trimesh_loader.visual.TextureVisuals(uv=uvs)

        if textured_mesh.visual.material is None:
            textured_mesh.visual.material = trimesh_loader.visual.material.PBRMaterial()
        
        material = textured_mesh.visual.material

        def tensor_to_pil(tensor):
            if tensor is None: return None
            return Image.fromarray((tensor[0].cpu().numpy() * 255).astype(np.uint8))

        if 'albedo' in final_maps:
            material.baseColorTexture = tensor_to_pil(final_maps['albedo'])
        if 'normal' in final_maps:
            material.normalTexture = tensor_to_pil(final_maps['normal'])
        if 'ao' in final_maps:
            material.occlusionTexture = tensor_to_pil(final_maps['ao'])

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path_str = os.path.join(full_output_folder, f"{filename}_{counter:05}.glb")
        
        textured_mesh.export(output_glb_path_str)
        
        return (textured_mesh, str(Path(subfolder) / f"{filename}_{counter:05}.glb"))

class BlendFBX_Export:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "directory": ("STRING", {"default": "output/exported_models"}),
                "original_filename": ("STRING", {"default": "model.glb"}),
            }
        }
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY, OUTPUT_NODE = ("STRING",), ("MODEL_FOLDER_PATH",), "export_model_and_textures_separately", "Comfy_BlenderTools", True

    def export_model_and_textures_separately(self, trimesh, directory, original_filename):
        if trimesh is None:
            return ("Error: No mesh provided.",)

        model_base_name = os.path.splitext(os.path.basename(original_filename))[0]
        model_output_folder = os.path.join(directory, model_base_name)
        counter = 1
        while os.path.exists(model_output_folder):
            numbered_folder_name = f"{model_base_name}_{counter}"
            model_output_folder = os.path.join(directory, numbered_folder_name)
            counter += 1
        os.makedirs(model_output_folder)

        try:
            if hasattr(trimesh, 'geometry') and trimesh.geometry:
                for geom in trimesh.geometry.values():
                    if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                        mat = geom.visual.material
                        texture_map = {
                            "baseColorTexture": "texture_basecolor.png",
                            "metallicRoughnessTexture": "texture_metallic-roughness.png",
                            "normalTexture": "texture_normal.png",
                            "occlusionTexture": "texture_ao.png",
                            "emissiveTexture": "texture_emissive.png",
                        }
                        for prop, filename in texture_map.items():
                            if hasattr(mat, prop) and getattr(mat, prop) is not None:
                                texture_image = getattr(mat, prop)
                                texture_path = os.path.join(model_output_folder, filename)
                                texture_image.save(texture_path)
        except Exception as e:
            pass

        final_fbx_filename = f"{os.path.basename(model_output_folder)}.fbx"
        final_fbx_path = os.path.join(model_output_folder, final_fbx_filename)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_obj_path = os.path.join(temp_dir, "temp_mesh.obj")
            try:
                trimesh.export(temp_obj_path, include_texture=False, include_color=False)
            except Exception as e:
                return (f"Error exporting intermediate mesh file: {e}",)
            
            script_path = os.path.join(temp_dir, "convert_script.py")
            script = f"""
import bpy, sys
import addon_utils
in_obj, out_fbx = r'{temp_obj_path}', r'{final_fbx_path}'
try:
    addon_utils.enable('io_scene_fbx', default_set=True, persistent=False)
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=in_obj)
    bpy.ops.export_scene.fbx(filepath=out_fbx, use_selection=True)
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr); sys.exit(1)
"""
            with open(script_path, 'w') as f: f.write(script)
            _run_blender_script(script_path)

        return (model_output_folder,)
