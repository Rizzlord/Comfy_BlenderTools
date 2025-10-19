import os
import sys
import tempfile
import traceback
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageFilter
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
                "bake_albedo": ("BOOLEAN", {"default": True}),
                "bake_mr_map": ("BOOLEAN", {"default": True}),
                "bake_normal": ("BOOLEAN", {"default": True}),
                "bake_ao": ("BOOLEAN", {"default": True}),
                "ao_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "bake_thickness": ("BOOLEAN", {"default": True}),
                "thickness_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "bake_cavity": ("BOOLEAN", {"default": True}),
                "cavity_contrast": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "cage_extrusion": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_ray_distance": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
                "margin": ("INT", {"default": 1024, "min": 0, "max": 2048}),
                "use_high_poly_textures": ("BOOLEAN", {"default": False}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "blur_atc": ("BOOLEAN", {"default": True}),
                "atc_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 150.0, "step": 0.1}),
                "blur_mr": ("BOOLEAN", {"default": False}),
                "mr_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 150.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("low_poly_mesh", "albedo_map", "normal_map", "mr_map", "ao_map", "thickness_map", "cavity_map", "atc_map")
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools"

    def bake(self, high_poly_mesh, low_poly_mesh, resolution, bake_albedo, bake_mr_map, bake_normal, bake_ao, ao_strength,
             bake_thickness, thickness_strength, bake_cavity, cavity_contrast,
             cage_extrusion, max_ray_distance, margin, use_high_poly_textures, use_gpu,
             blur_atc, atc_blur_strength, blur_mr, mr_blur_strength):

        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        def get_material(mesh):
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                return mesh.visual.material
            return None

        high_material = get_material(high_poly_mesh)
        low_material = get_material(low_poly_mesh)

        def tensor_from_material_texture(material, attr):
            if material is None or not hasattr(material, attr):
                return None
            texture = getattr(material, attr)
            if texture is None:
                return None
            pil_img = texture.convert('RGB')
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(img_array)[None,]

        # Preserve existing textures for output, preferring low-poly then high-poly
        albedo_map_tensor = tensor_from_material_texture(low_material, 'baseColorTexture')
        if albedo_map_tensor is None:
            albedo_map_tensor = tensor_from_material_texture(high_material, 'baseColorTexture')

        mr_map_tensor = tensor_from_material_texture(low_material, 'metallicRoughnessTexture')
        if mr_map_tensor is None:
            mr_map_tensor = tensor_from_material_texture(high_material, 'metallicRoughnessTexture')

        def select_texture_image(attr):
            material_order = [high_material, low_material] if use_high_poly_textures else [low_material, high_material]
            for material in material_order:
                if material is None or not hasattr(material, attr):
                    continue
                texture = getattr(material, attr)
                if texture is None:
                    continue
                return texture.convert('RGB')
            return None

        selected_albedo_image = select_texture_image('baseColorTexture')
        selected_mr_image = select_texture_image('metallicRoughnessTexture')

        original_material = low_poly_mesh.visual.material if hasattr(low_poly_mesh.visual, 'material') else None

        with tempfile.TemporaryDirectory() as temp_dir:
            high_poly_path = os.path.join(temp_dir, "high.obj")
            low_poly_path = os.path.join(temp_dir, "low.obj")
            final_low_poly_path = os.path.join(temp_dir, "final_low.glb")
            script_path = os.path.join(temp_dir, "s.py")

            high_poly_mesh.export(file_obj=high_poly_path)
            low_poly_mesh.export(file_obj=low_poly_path)

            albedo_texture_path = None
            if selected_albedo_image is not None:
                albedo_texture_path = os.path.join(temp_dir, "source_albedo.png")
                selected_albedo_image.save(albedo_texture_path)

            mr_texture_path = None
            if selected_mr_image is not None:
                mr_texture_path = os.path.join(temp_dir, "source_mr.png")
                selected_mr_image.save(mr_texture_path)

            params = {
                'high_poly_path': high_poly_path, 'low_poly_path': low_poly_path,
                'final_low_poly_path': final_low_poly_path, 'temp_dir': temp_dir,
                'bake_albedo': bake_albedo, 'bake_mr': bake_mr_map, 'bake_normal': bake_normal, 
                'bake_ao': bake_ao, 'ao_strength': ao_strength, 'bake_thickness': bake_thickness, 
                'thickness_strength': thickness_strength, 'bake_cavity': bake_cavity, 
                'cavity_contrast': cavity_contrast, 'resolution': int(resolution), 
                'cage_extrusion': cage_extrusion, 'max_ray_distance': max_ray_distance, 'margin': margin,
                'use_high_poly_textures': use_high_poly_textures,
                'use_gpu': use_gpu,
                'albedo_texture_path': albedo_texture_path,
                'mr_texture_path': mr_texture_path
            }

            clean_mesh_func_script = get_blender_clean_mesh_func_script()

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

def load_image_resource(path, colorspace):
    if not path or not os.path.exists(path):
        return None
    try:
        image = bpy.data.images.load(path, check_existing=True)
    except Exception:
        image = bpy.data.images.load(path)
    image.colorspace_settings.name = colorspace
    return image

def apply_highpoly_textures(high_obj):
    albedo_image = load_image_resource(p.get('albedo_texture_path'), 'sRGB')
    mr_image = load_image_resource(p.get('mr_texture_path'), 'Non-Color')
    if albedo_image is None and mr_image is None:
        return

    if not high_obj.data.materials:
        mat = bpy.data.materials.new(name="TextureBakeHigh")
        high_obj.data.materials.append(mat)

    for i, mat in enumerate(high_obj.data.materials):
        if mat is None:
            mat = bpy.data.materials.new(name=f"TextureBakeHigh_{{i}}")
            high_obj.data.materials[i] = mat
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

        if albedo_image:
            albedo_node = nodes.new('ShaderNodeTexImage')
            albedo_node.image = albedo_image
            albedo_node.interpolation = 'Smart'
            albedo_node.name = "TextureBake_Albedo"
            links.new(albedo_node.outputs['Color'], principled.inputs['Base Color'])

        if mr_image:
            mr_node = nodes.new('ShaderNodeTexImage')
            mr_node.image = mr_image
            mr_node.image.colorspace_settings.name = 'Non-Color'
            mr_node.interpolation = 'Smart'
            mr_node.name = "TextureBake_MetallicRoughness"
            separate = nodes.new('ShaderNodeSeparateRGB')
            links.new(mr_node.outputs['Color'], separate.inputs['Image'])
            links.new(separate.outputs['G'], principled.inputs['Roughness'])
            links.new(separate.outputs['B'], principled.inputs['Metallic'])

def enable_emission_bake(obj):
    overrides = []
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not output_node:
            continue
        surface_socket = output_node.inputs.get('Surface')
        if surface_socket is None:
            continue
        surface_links = [l for l in links if l.to_node == output_node and l.to_socket == surface_socket]
        if not surface_links:
            continue
        principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        source_socket = None
        default_color = (1.0, 1.0, 1.0, 1.0)
        if principled:
            base_color_input = principled.inputs.get('Base Color')
            if base_color_input is not None:
                if base_color_input.is_linked:
                    source_socket = base_color_input.links[0].from_socket
                else:
                    default_color = tuple(base_color_input.default_value)
        emission = nodes.new('ShaderNodeEmission')
        emission.name = 'TextureBakeEmission'
        emission.inputs['Strength'].default_value = 1.0
        if source_socket:
            links.new(source_socket, emission.inputs['Color'])
        else:
            emission.inputs['Color'].default_value = default_color
        stored_links = []
        for link in surface_links:
            stored_links.append((link.from_socket, surface_socket))
            links.remove(link)
        links.new(emission.outputs['Emission'], surface_socket)
        overrides.append({{'material': mat, 'emission': emission, 'links': stored_links}})
    return overrides

def enable_mr_emission_bake(obj):
    overrides = []
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not output_node:
            continue
        surface_socket = output_node.inputs.get('Surface')
        if surface_socket is None:
            continue
        surface_links = [l for l in links if l.to_node == output_node and l.to_socket == surface_socket]
        if not surface_links:
            continue

        mr_node = nodes.get('TextureBake_MetallicRoughness')
        if mr_node is None or not getattr(mr_node, 'image', None):
            continue

        emission = nodes.new('ShaderNodeEmission')
        emission.name = 'TextureBakeEmissionMR'
        emission.inputs['Strength'].default_value = 1.0
        links.new(mr_node.outputs['Color'], emission.inputs['Color'])

        stored_links = []
        for link in surface_links:
            stored_links.append((link.from_socket, surface_socket))
            links.remove(link)
        links.new(emission.outputs['Emission'], surface_socket)
        overrides.append({{'material': mat, 'emission': emission, 'links': stored_links}})
    return overrides

def restore_emission_bake(overrides):
    if overrides is None:
        return
    for data in overrides:
        mat = data.get('material')
        emission = data.get('emission')
        stored_links = data.get('links', [])
        if not mat or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for link in list(links):
            if link.from_node == emission or link.to_node == emission:
                links.remove(link)
        try:
            nodes.remove(emission)
        except Exception:
            emission_node = nodes.get(getattr(emission, 'name', None))
            if emission_node:
                nodes.remove(emission_node)
        for from_socket, to_socket in stored_links:
            try:
                links.new(from_socket, to_socket)
            except Exception as exc:
                print("Warning: failed to restore material link: " + str(exc))

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
    apply_highpoly_textures(high_obj)
    low_poly_mat, tex_node = setup_lowpoly_material(low_obj)

    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True); low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    
    if p['bake_albedo']:
        emission_overrides = enable_emission_bake(high_obj)
        diffuse_image = None
        try:
            diffuse_image = execute_bake('EMIT', tex_node)
            output_path = os.path.join(p['temp_dir'], "albedo_map.png")
            diffuse_image.filepath_raw = output_path
            diffuse_image.file_format = 'PNG'
            diffuse_image.save()
        finally:
            if diffuse_image is not None:
                bpy.data.images.remove(diffuse_image)
            restore_emission_bake(emission_overrides)

    if p['bake_mr']:
        mr_overrides = enable_mr_emission_bake(high_obj)
        mr_image = None
        try:
            if mr_overrides:
                mr_image = execute_bake('EMIT', tex_node)
                output_path = os.path.join(p['temp_dir'], "mr_map.png")
                mr_image.filepath_raw = output_path
                mr_image.file_format = 'PNG'
                mr_image.save()
        finally:
            if mr_image is not None:
                bpy.data.images.remove(mr_image)
            restore_emission_bake(mr_overrides)

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

            res_int = int(resolution)

            def load_image_as_tensor(path):
                if not os.path.exists(path): return None
                img = Image.open(path).convert('RGB')
                return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

            albedo_bake_tensor = load_image_as_tensor(os.path.join(temp_dir, "albedo_map.png"))
            if albedo_bake_tensor is not None:
                albedo_map_tensor = albedo_bake_tensor

            mr_bake_tensor = load_image_as_tensor(os.path.join(temp_dir, "mr_map.png"))
            if mr_bake_tensor is not None:
                mr_map_tensor = mr_bake_tensor
            if blur_mr and mr_map_tensor is not None:
                mr_map_tensor = self._blur_image_tensor(mr_map_tensor, mr_blur_strength)

            normal_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "normal_map.png"))
            ao_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "ao_map.png"))
            thickness_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "thickness_map.png"))
            cavity_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "cavity_map.png"))

            atc_map_tensor = None
            if any([bake_ao, bake_thickness, bake_cavity]):
                black_channel = torch.zeros((1, res_int, res_int, 1), dtype=torch.float32)
                
                r_channel = ao_map_tensor[:, :, :, 0:1] if ao_map_tensor is not None else black_channel
                g_channel = thickness_map_tensor[:, :, :, 0:1] if thickness_map_tensor is not None else black_channel
                b_channel = cavity_map_tensor[:, :, :, 0:1] if cavity_map_tensor is not None else black_channel
                
                atc_map_tensor = torch.cat([r_channel, g_channel, b_channel], dim=-1)
            if atc_map_tensor is not None and blur_atc:
                atc_map_tensor = self._blur_image_tensor(atc_map_tensor, atc_blur_strength)

            final_mesh = trimesh_loader.load(final_low_poly_path, force="mesh")
            if original_material:
                final_mesh.visual.material = original_material

            return (final_mesh,
                    albedo_map_tensor if albedo_map_tensor is not None else dummy_image,
                    normal_map_tensor if normal_map_tensor is not None else dummy_image,
                    mr_map_tensor if mr_map_tensor is not None else dummy_image,
                    ao_map_tensor if ao_map_tensor is not None else dummy_image,
                    thickness_map_tensor if thickness_map_tensor is not None else dummy_image,
                    cavity_map_tensor if cavity_map_tensor is not None else dummy_image,
                    atc_map_tensor if atc_map_tensor is not None else dummy_image)

    @staticmethod
    def _blur_image_tensor(image_tensor, strength):
        if image_tensor is None:
            return None
        if strength is None or strength <= 0:
            return image_tensor
        if not torch.is_tensor(image_tensor):
            return image_tensor
        if image_tensor.ndim != 4 or image_tensor.shape[0] == 0:
            return image_tensor

        device = image_tensor.device
        dtype = image_tensor.dtype
        radius = float(strength)

        blurred_batches = []
        for i in range(image_tensor.shape[0]):
            img_array = image_tensor[i].detach().cpu().numpy()
            img_array = np.clip(img_array, 0.0, 1.0)
            pil_img = Image.fromarray((img_array * 255.0).astype(np.uint8))
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            blurred_np = np.array(blurred_pil).astype(np.float32) / 255.0
            if blurred_np.ndim == 2:
                blurred_np = blurred_np[:, :, None]
            blurred_batches.append(torch.from_numpy(blurred_np))

        result = torch.stack(blurred_batches, dim=0)
        return result.to(device=device, dtype=dtype)

class ApplyMaterial:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "albedo_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "mr_map": ("IMAGE",),
                "ao_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "apply"
    CATEGORY = "Comfy_BlenderTools"

    def apply(self, mesh, albedo_map=None, normal_map=None, mr_map=None, ao_map=None):
        new_mesh = mesh.copy()
        if not hasattr(new_mesh, 'visual') or not hasattr(new_mesh.visual, 'uv'):
            raise Exception("Mesh must have UV coordinates to apply materials. Use the BlenderUnwrap node.")

        material = trimesh_loader.visual.material.PBRMaterial()

        def tensor_to_pil(tensor):
            if tensor is None:
                return None
            i = 255. * tensor[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            return img.convert("RGB")

        def create_checker_map_pil(size=512, checker_size=64):
            indices = np.arange(size)
            x_checkers = np.floor_divide(indices, checker_size)
            y_checkers = np.floor_divide(indices, checker_size)[:, np.newaxis]
            pattern = (x_checkers + y_checkers) % 2
            
            black = [0, 0, 0]
            purple = [255, 0, 255]
            
            img_array = np.array([black, purple], dtype=np.uint8)[pattern]
            return Image.fromarray(img_array, 'RGB')

        base_color_texture = None
        if albedo_map is not None:
            base_color_texture = tensor_to_pil(albedo_map)
        else:
            base_color_texture = create_checker_map_pil()

        material.baseColorTexture = base_color_texture
        material.normalTexture = tensor_to_pil(normal_map)
        material.metallicRoughnessTexture = tensor_to_pil(mr_map)
        material.occlusionTexture = tensor_to_pil(ao_map)

        new_mesh.visual = trimesh_loader.visual.texture.TextureVisuals(uv=new_mesh.visual.uv, material=material)

        return (new_mesh,)

class ExtractMaterial:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo_map", "normal_map", "mr_map", "ao_map")
    FUNCTION = "extract"
    CATEGORY = "Comfy_BlenderTools"

    def extract(self, mesh):
        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        def pil_to_tensor(pil_img):
            if pil_img is None:
                return dummy_image
            
            pil_img = pil_img.convert('RGB')
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(img_array)[None,]

        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'material'):
            return (dummy_image, dummy_image, dummy_image, dummy_image)

        mat = mesh.visual.material
        
        albedo_map = pil_to_tensor(getattr(mat, 'baseColorTexture', None))
        normal_map = pil_to_tensor(getattr(mat, 'normalTexture', None))
        mr_map = pil_to_tensor(getattr(mat, 'metallicRoughnessTexture', None))
        ao_map = pil_to_tensor(getattr(mat, 'occlusionTexture', None))

        return (albedo_map, normal_map, mr_map, ao_map)

class SaveMultiviewImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"default": "multiview_bake"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "Comfy_BlenderTools"
    OUTPUT_NODE = True

    def save(self, images, path):
        output_dir = folder_paths.get_output_directory()
        full_path = os.path.join(output_dir, path)

        counter = 1
        while os.path.exists(f"{full_path}_{counter:05}_"):
            counter += 1
        
        final_path = f"{full_path}_{counter:05}_"
        os.makedirs(final_path, exist_ok=True)

        for i, image_tensor in enumerate(images):
            img_pil = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
            img_pil.save(os.path.join(final_path, f"MV_{i+1}.png"))
            
        return (final_path,)

class LoadMultiviewImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load"
    CATEGORY = "Comfy_BlenderTools"

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        image_files = sorted(
            [f for f in os.listdir(path) if f.startswith("MV_") and f.endswith(".png")],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        if not image_files:
            raise FileNotFoundError(f"No multiview images (MV_*.png) found in {path}")

        images = []
        for file_name in image_files:
            img_path = os.path.join(path, file_name)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            images.append(img_tensor)
        
        return (torch.stack(images),)
