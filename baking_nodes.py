import os
import tempfile
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image
from .utils import _run_blender_script, get_blender_clean_mesh_func_script

class TextureBake:
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "low_poly_mesh": ("TRIMESH",),
                "resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "bake_normal": ("BOOLEAN", {"default": True}),
                "bake_ao": ("BOOLEAN", {"default": True}),
                "ao_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "bake_thickness": ("BOOLEAN", {"default": True}),
                "thickness_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "bake_cavity": ("BOOLEAN", {"default": True}),
                "cavity_contrast": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "cage_extrusion": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_ray_distance": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "BAKED_MAPS", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("low_poly_mesh", "baked_maps", "albedo_map", "rm_map", "normal_map", "ao_map", "thickness_map", "cavity_map", "ATC_map")
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools"

    def bake(self, high_poly_mesh, low_poly_mesh, resolution, bake_normal, bake_ao, ao_strength,
             bake_thickness, thickness_strength, bake_cavity, cavity_contrast,
             cage_extrusion, max_ray_distance, margin):

        albedo_map_tensor = None
        rm_map_tensor = None
        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        if hasattr(low_poly_mesh, 'visual') and hasattr(low_poly_mesh.visual, 'material'):
            mat = low_poly_mesh.visual.material
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                pil_img = mat.baseColorTexture.convert('RGB')
                img_array = np.array(pil_img).astype(np.float32) / 255.0
                albedo_map_tensor = torch.from_numpy(img_array)[None,]
            
            if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                pil_mr_img = mat.metallicRoughnessTexture.convert('RGB')
                mr_array = np.array(pil_mr_img).astype(np.float32) / 255.0
                rm_map_tensor = torch.from_numpy(mr_array)[None,]

        original_material = low_poly_mesh.visual.material if hasattr(low_poly_mesh.visual, 'material') else None

        with tempfile.TemporaryDirectory() as temp_dir:
            high_poly_path = os.path.join(temp_dir, "high.obj")
            low_poly_path = os.path.join(temp_dir, "low.obj")
            final_low_poly_path = os.path.join(temp_dir, "final_low.glb")
            script_path = os.path.join(temp_dir, "s.py")

            high_poly_mesh.export(file_obj=high_poly_path)
            low_poly_mesh.export(file_obj=low_poly_path)

            params = {
                'high_poly_path': high_poly_path, 'low_poly_path': low_poly_path,
                'final_low_poly_path': final_low_poly_path, 'temp_dir': temp_dir,
                'bake_normal': bake_normal, 
                'bake_ao': bake_ao, 'ao_strength': ao_strength, 'bake_thickness': bake_thickness, 
                'thickness_strength': thickness_strength, 'bake_cavity': bake_cavity, 
                'cavity_contrast': cavity_contrast, 'resolution': int(resolution), 
                'cage_extrusion': cage_extrusion, 'max_ray_distance': max_ray_distance, 'margin': margin
            }

            clean_mesh_func_script = get_blender_clean_mesh_func_script()

            script = f"""
{clean_mesh_func_script}
import bpy, sys, os, traceback
p = {{ {", ".join(f'"{k}": r"{v}"' if isinstance(v, str) else f'"{k}": {v}' for k, v in params.items())} }}

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_meshes():
    bpy.ops.wm.obj_import(filepath=p['high_poly_path'])
    high_obj = bpy.context.selected_objects[0]; high_obj.name = "HighPoly"
    clean_mesh(high_obj, 0.0001)

    bpy.ops.wm.obj_import(filepath=p['low_poly_path'])
    low_obj = bpy.context.selected_objects[0]; low_obj.name = "LowPoly"
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

def process_and_save_map(image, output_path, invert=False, strength=1.0, white_bg=False):
    comp_scene = bpy.context.scene
    comp_scene.use_nodes = True
    tree, links = comp_scene.node_tree, comp_scene.node_tree.links
    tree.nodes.clear()

    img_node = tree.nodes.new('CompositorNodeImage'); img_node.image = image
    
    current_link = img_node.outputs['Image']
    
    if invert:
        invert_node = tree.nodes.new('CompositorNodeInvert')
        links.new(current_link, invert_node.inputs['Color'])
        current_link = invert_node.outputs['Color']

    if strength != 1.0:
        strength_node = tree.nodes.new('CompositorNodeMath')
        strength_node.operation = 'POWER'
        strength_node.inputs[1].default_value = strength
        links.new(current_link, strength_node.inputs[0])
        current_link = strength_node.outputs['Value']

    if white_bg:
        alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
        alpha_over_node.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)
        links.new(img_node.outputs['Alpha'], alpha_over_node.inputs['Fac'])
        links.new(current_link, alpha_over_node.inputs[2])
        current_link = alpha_over_node.outputs['Image']

    composite_node = tree.nodes.new('CompositorNodeComposite')
    links.new(current_link, composite_node.inputs['Image'])
    
    comp_scene.render.resolution_x, comp_scene.render.resolution_y = image.size
    comp_scene.render.image_settings.file_format = 'PNG'
    comp_scene.render.image_settings.color_mode = 'RGB'
    comp_scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    tree.nodes.clear()

def execute_bake(bake_type, tex_node):
    res = p['resolution']
    bake_image = bpy.data.images.new(name=f"{{bake_type}}_BakeImage", width=res, height=res, alpha=True)
    tex_node.image = bake_image
    
    bake_kwargs = {{
        'type': bake_type, 'use_selected_to_active': True,
        'margin': p['margin'], 'cage_extrusion': p['cage_extrusion'], 'use_clear': True,
    }}
    if p['max_ray_distance'] > 0.0:
        bake_kwargs['max_ray_distance'] = p['max_ray_distance']
    if bake_type == 'NORMAL':
        bake_kwargs['normal_space'] = 'TANGENT'

    bpy.ops.object.bake(**bake_kwargs)
    return bake_image

try:
    setup_scene()
    high_obj, low_obj = import_meshes()
    low_poly_mat, tex_node = setup_lowpoly_material(low_obj)

    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True); low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    
    if p['bake_normal']:
        normal_image = execute_bake('NORMAL', tex_node)
        output_path = os.path.join(p['temp_dir'], "normal_map.png")
        normal_image.filepath_raw = output_path
        normal_image.file_format = 'PNG'; normal_image.save()
        bpy.data.images.remove(normal_image)

    if p['bake_ao']:
        ao_image = execute_bake('AO', tex_node)
        process_and_save_map(ao_image, os.path.join(p['temp_dir'], "ao_map.png"), 
                             strength=p['ao_strength'], white_bg=True)
        bpy.data.images.remove(ao_image)

    if p['bake_thickness']:
        thickness_image = execute_bake('AO', tex_node)
        process_and_save_map(thickness_image, os.path.join(p['temp_dir'], "thickness_map.png"), 
                             invert=True, strength=p['thickness_strength'], white_bg=True)
        bpy.data.images.remove(thickness_image)

    if p['bake_cavity']:
        original_high_mat = high_obj.data.materials[0] if high_obj.data.materials else None
        cavity_mat = bpy.data.materials.new(name="CavityMat"); cavity_mat.use_nodes = True
        nodes, links = cavity_mat.node_tree.nodes, cavity_mat.node_tree.links; nodes.clear()
        geo = nodes.new(type='ShaderNodeNewGeometry')
        cr = nodes.new(type='ShaderNodeValToRGB')
        contrast = p['cavity_contrast']
        cr.color_ramp.elements[0].position = max(0.0, 0.5 - (0.5 / contrast))
        cr.color_ramp.elements[1].position = min(1.0, 0.5 + (0.5 / contrast))
        emission = nodes.new('ShaderNodeEmission'); output = nodes.new('ShaderNodeOutputMaterial')
        links.new(geo.outputs['Pointiness'], cr.inputs['Fac'])
        links.new(cr.outputs['Color'], emission.inputs['Color'])
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
        if high_obj.data.materials: high_obj.data.materials[0] = cavity_mat
        else: high_obj.data.materials.append(cavity_mat)
        
        cavity_image = execute_bake('EMIT', tex_node)
        output_path = os.path.join(p['temp_dir'], "cavity_map.png")
        cavity_image.filepath_raw = output_path
        cavity_image.file_format = 'PNG'; cavity_image.save()
        bpy.data.images.remove(cavity_image)
        
        high_obj.data.materials.clear()
        if original_high_mat: high_obj.data.materials.append(original_high_mat)

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

            baked_maps = {}
            res_int = int(resolution)
            
            if albedo_map_tensor is not None:
                baked_maps['albedo'] = albedo_map_tensor
            
            if rm_map_tensor is not None:
                baked_maps['rm_map'] = rm_map_tensor

            def load_image_as_tensor(path):
                if not os.path.exists(path): return None
                img = Image.open(path).convert('RGB')
                return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

            normal_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "normal_map.png"))
            if normal_map_tensor is not None: baked_maps['normal'] = normal_map_tensor

            ao_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "ao_map.png"))
            if ao_map_tensor is not None: baked_maps['ao'] = ao_map_tensor
            
            thickness_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "thickness_map.png"))
            if thickness_map_tensor is not None: baked_maps['thickness'] = thickness_map_tensor
            
            cavity_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "cavity_map.png"))
            if cavity_map_tensor is not None: baked_maps['cavity'] = cavity_map_tensor

            atc_map_tensor = None
            if any([bake_ao, bake_thickness, bake_cavity]):
                black_channel = torch.zeros((1, res_int, res_int, 1), dtype=torch.float32)
                
                r_channel = ao_map_tensor[:, :, :, 0:1] if ao_map_tensor is not None else black_channel
                g_channel = thickness_map_tensor[:, :, :, 0:1] if thickness_map_tensor is not None else black_channel
                b_channel = cavity_map_tensor[:, :, :, 0:1] if cavity_map_tensor is not None else black_channel
                
                atc_map_tensor = torch.cat([r_channel, g_channel, b_channel], dim=-1)
                baked_maps['atc'] = atc_map_tensor
                if ao_map_tensor is not None:
                     baked_maps['ao'] = atc_map_tensor

            final_mesh = trimesh_loader.load(final_low_poly_path, force="mesh")
            if original_material:
                final_mesh.visual.material = original_material

            return (final_mesh, baked_maps,
                    albedo_map_tensor if albedo_map_tensor is not None else dummy_image,
                    rm_map_tensor if rm_map_tensor is not None else dummy_image,
                    normal_map_tensor if normal_map_tensor is not None else dummy_image,
                    ao_map_tensor if ao_map_tensor is not None else dummy_image,
                    thickness_map_tensor if thickness_map_tensor is not None else dummy_image,
                    cavity_map_tensor if cavity_map_tensor is not None else dummy_image,
                    atc_map_tensor if atc_map_tensor is not None else dummy_image)