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

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("low_poly_mesh", "albedo_map", "normal_map", "rm_map", "ao_map", "thickness_map", "cavity_map", "ATC_map")
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

            res_int = int(resolution)

            def load_image_as_tensor(path):
                if not os.path.exists(path): return None
                img = Image.open(path).convert('RGB')
                return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

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

            final_mesh = trimesh_loader.load(final_low_poly_path, force="mesh")
            if original_material:
                final_mesh.visual.material = original_material

            return (final_mesh,
                    albedo_map_tensor if albedo_map_tensor is not None else dummy_image,
                    normal_map_tensor if normal_map_tensor is not None else dummy_image,
                    rm_map_tensor if rm_map_tensor is not None else dummy_image,
                    ao_map_tensor if ao_map_tensor is not None else dummy_image,
                    thickness_map_tensor if thickness_map_tensor is not None else dummy_image,
                    cavity_map_tensor if cavity_map_tensor is not None else dummy_image,
                    atc_map_tensor if atc_map_tensor is not None else dummy_image)
        
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
                "metallic_roughness_map": ("IMAGE",),
                "ao_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "apply"
    CATEGORY = "Comfy_BlenderTools"

    def apply(self, mesh, albedo_map=None, normal_map=None, metallic_roughness_map=None, ao_map=None):
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
        material.metallicRoughnessTexture = tensor_to_pil(metallic_roughness_map)
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
    RETURN_NAMES = ("albedo_map", "normal_map", "metallic_roughness_map", "ao_map")
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
    
class VertexBake:
    RESOLUTIONS = ["512", "1024", "2048", "4096", "8192"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly_mesh": ("TRIMESH",),
                "low_poly_mesh": ("TRIMESH",),
                "multiview_images": ("IMAGE",),
                "camera_config": ("HY3DCAMERA",),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "resolution": (cls.RESOLUTIONS, {"default": "2048"}),
                "cage_extrusion": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_ray_distance": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.001}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TRIMESH",)
    RETURN_NAMES = ("baked_texture", "debug_high_poly",)
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools"

    def bake(self, high_poly_mesh, low_poly_mesh, multiview_images, camera_config, use_gpu, resolution, cage_extrusion, max_ray_distance, margin):
        if not multiview_images.shape[0] > 0:
            raise ValueError("No images provided in multiview_images")
        if multiview_images.shape[-1] not in [3, 4]:
            raise ValueError("Images must be RGB or RGBA")
        if not isinstance(camera_config, dict):
            raise ValueError("camera_config must be a dictionary")
        required_keys = ["selected_camera_azims", "selected_camera_elevs", "camera_distance", "ortho_scale"]
        if not all(key in camera_config for key in required_keys):
            raise ValueError(f"camera_config missing required keys: {required_keys}")
        if len(camera_config["selected_camera_azims"]) != len(camera_config["selected_camera_elevs"]):
            raise ValueError("Mismatch between camera azimuths and elevations")
        if len(multiview_images) != len(camera_config["selected_camera_azims"]):
            raise ValueError("Number of images must match number of camera azimuths/elevations")

        with tempfile.TemporaryDirectory() as temp_dir:
            high_poly_path = os.path.join(temp_dir, "high.obj")
            low_poly_path = os.path.join(temp_dir, "low.obj")
            final_high_poly_path = os.path.join(temp_dir, "final_high.glb")
            high_poly_mesh.export(file_obj=high_poly_path)
            low_poly_mesh.export(file_obj=low_poly_path)

            image_paths = []
            for i, image_tensor in enumerate(multiview_images):
                try:
                    image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
                    if image_array.size == 0 or image_array.shape[0] == 0 or image_array.shape[1] == 0:
                        raise ValueError(f"Image {i} is empty or invalid")
                    image_pil = Image.fromarray(image_array).convert("RGB")
                    image_path = os.path.join(temp_dir, f"view_{i}.png")
                    image_pil.save(image_path)
                    image_paths.append(image_path)
                except Exception as e:
                    print(f"Failed to save image {i}: {e}", file=sys.stderr)
                    raise

            baked_texture_path = os.path.join(temp_dir, "baked_texture.png")
            script_path = os.path.join(temp_dir, "bake_script.py")

            params = {
                'high_poly_path': high_poly_path,
                'low_poly_path': low_poly_path,
                'baked_texture_path': baked_texture_path,
                'final_high_poly_path': final_high_poly_path,
                'image_paths': image_paths,
                'camera_azimuths': camera_config["selected_camera_azims"],
                'camera_elevations': camera_config["selected_camera_elevs"],
                'camera_distance': camera_config["camera_distance"],
                'ortho_scale': camera_config["ortho_scale"],
                'use_gpu': use_gpu,
                'resolution': int(resolution),
                'cage_extrusion': cage_extrusion,
                'max_ray_distance': max_ray_distance,
                'margin': margin,
            }

            script = f"""
import bpy
import math
import mathutils
import sys
import traceback

p = {params}

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

def import_meshes():
    bpy.ops.wm.obj_import(filepath=p['high_poly_path'])
    high_obj = bpy.context.selected_objects[0]
    high_obj.name = "HighPoly"
    
    bpy.ops.wm.obj_import(filepath=p['low_poly_path'])
    low_obj = bpy.context.selected_objects[0]
    low_obj.name = "LowPoly"

    if not low_obj.data.uv_layers:
        raise Exception("Low-poly mesh has no UV map.")

    return high_obj, low_obj

def get_view_matrix(azimuth, elevation, distance):
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    x = distance * math.cos(azimuth_rad) * math.cos(elevation_rad)
    y = distance * math.sin(azimuth_rad) * math.cos(elevation_rad)
    z = distance * math.sin(elevation_rad)
    camera_location = mathutils.Vector((x, y, z))
    direction = -camera_location
    rot_quat = direction.to_track_quat('Z', 'Y')
    mat = rot_quat.to_matrix().to_4x4()
    mat.translation = camera_location
    return mat

def blend_colors_by_loop(high_obj, final_name, temp_name):
    final_layer = high_obj.data.color_attributes[final_name]
    temp_layer = high_obj.data.color_attributes[temp_name]

    for i in range(len(high_obj.data.loops)):
        c_final = final_layer.data[i].color
        c_temp = temp_layer.data[i].color
        
        final_layer.data[i].color = (
            min(1.0, c_final[0] + c_temp[0]),
            min(1.0, c_final[1] + c_temp[1]),
            min(1.0, c_final[2] + c_temp[2]),
            1.0
        )
    
    high_obj.data.color_attributes.remove(temp_layer)

def project_colors_with_baking(high_obj):
    scene = bpy.context.scene
    
    cam_data = bpy.data.cameras.new("ProjectionCam")
    cam_obj = bpy.data.objects.new("ProjectionCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = p['ortho_scale']

    final_vcol_name = "FinalColor"
    temp_vcol_name = "TempColor"

    for layer in high_obj.data.color_attributes:
        high_obj.data.color_attributes.remove(layer)
    
    final_vcol = high_obj.data.color_attributes.new(name=final_vcol_name, type='BYTE_COLOR', domain='CORNER')
    
    proj_mat = bpy.data.materials.new("ProjectionMaterial")
    proj_mat.use_nodes = True
    nodes = proj_mat.node_tree.nodes
    links = proj_mat.node_tree.links
    nodes.clear()
    
    tex_coord_node = nodes.new('ShaderNodeTexCoord')
    tex_img_node = nodes.new('ShaderNodeTexImage')
    emission_node = nodes.new('ShaderNodeEmission')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    
    links.new(tex_coord_node.outputs['Window'], tex_img_node.inputs['Vector'])
    links.new(tex_img_node.outputs['Color'], emission_node.inputs['Color'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
    high_obj.data.materials.append(proj_mat)

    for i, img_path in enumerate(p['image_paths']):
        print(f"Projecting and blending view {{i+1}}/{{len(p['image_paths'])}}...")
        img = bpy.data.images.load(img_path)
        tex_img_node.image = img

        cam_obj.matrix_world = get_view_matrix(
            p['camera_azimuths'][i],
            p['camera_elevations'][i],
            p['camera_distance']
        )
        
        temp_vcol = high_obj.data.color_attributes.new(name=temp_vcol_name, type='BYTE_COLOR', domain='CORNER')
        high_obj.data.color_attributes.active_color = temp_vcol
        
        bpy.ops.object.select_all(action='DESELECT')
        high_obj.select_set(True)
        scene.view_layers[0].objects.active = high_obj
        
        scene.render.bake.target = 'VERTEX_COLORS'
        scene.render.bake.use_selected_to_active = False
        bpy.ops.object.bake(type='EMIT', use_clear=True)

        bpy.data.images.remove(img)
        
        blend_colors_by_loop(high_obj, final_vcol_name, temp_vcol_name)
    
    bpy.data.objects.remove(cam_obj)
    bpy.data.cameras.remove(cam_data)

def bake_to_texture(high_obj, low_obj):
    mat_high = bpy.data.materials.new("HighPolyVColorMat")
    mat_high.use_nodes = True
    high_obj.data.materials.clear()
    high_obj.data.materials.append(mat_high)
    nodes_high = mat_high.node_tree.nodes
    links_high = mat_high.node_tree.links
    nodes_high.clear()
    
    vcol_node = nodes_high.new(type='ShaderNodeAttribute')
    vcol_node.attribute_name = "FinalColor"
    principled_node = nodes_high.new(type='ShaderNodeBsdfPrincipled')
    output_node = nodes_high.new(type='ShaderNodeOutputMaterial')
    links_high.new(vcol_node.outputs['Color'], principled_node.inputs['Base Color'])
    links_high.new(vcol_node.outputs['Color'], principled_node.inputs['Emission Color'])
    links_high.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    mat_low = bpy.data.materials.new("BakeMaterial")
    mat_low.use_nodes = True
    low_obj.data.materials.clear()
    low_obj.data.materials.append(mat_low)
    nodes_low = mat_low.node_tree.nodes
    nodes_low.clear()
    tex_node = nodes_low.new('ShaderNodeTexImage')
    bake_image = bpy.data.images.new("BakedTexture", width=p['resolution'], height=p['resolution'])
    tex_node.image = bake_image
    nodes_low.active = tex_node
    
    cage_obj = low_obj.copy()
    cage_obj.name = "BakeCage"
    cage_obj.data = low_obj.data.copy()
    bpy.context.collection.objects.link(cage_obj)
    
    bpy.context.view_layer.objects.active = cage_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.shrink_fatten(value=p['cage_extrusion'])
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True)
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    
    bpy.context.scene.render.bake.target = 'IMAGE_TEXTURES'
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.ops.object.bake(
        type='EMIT',
        use_cage=True,
        cage_object=cage_obj.name,
        cage_extrusion=p['cage_extrusion'],
        max_ray_distance=p['max_ray_distance'],
        margin=p['margin'],
        use_clear=True
    )

    bake_image.filepath_raw = p['baked_texture_path']
    bake_image.file_format = 'PNG'
    bake_image.save()
    bpy.data.objects.remove(cage_obj)

try:
    setup_scene()
    setup_gpu()
    high_obj, low_obj = import_meshes()
    project_colors_with_baking(high_obj)
    bake_to_texture(high_obj, low_obj)
    
    bpy.ops.object.select_all(action='DESELECT')
    high_obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=p['final_high_poly_path'], use_selection=True, export_colors=True)
    
    print("Bake completed successfully")
    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""

            try:
                with open(script_path, 'w') as f:
                    f.write(script)
                
                _run_blender_script(script_path)
                
                baked_texture_tensor = None
                if os.path.exists(baked_texture_path):
                    baked_texture_pil = Image.open(baked_texture_path).convert('RGB')
                    baked_texture_np = np.array(baked_texture_pil).astype(np.float32) / 255.0
                    baked_texture_tensor = torch.from_numpy(baked_texture_np)[None,]
                else:
                    raise FileNotFoundError(f"Baked texture not found at {baked_texture_path}")

                debug_high_poly_mesh = None
                if os.path.exists(final_high_poly_path):
                    debug_high_poly_mesh = trimesh.load(final_high_poly_path, force='mesh')
                else:
                    raise FileNotFoundError(f"Debug high-poly mesh not found at {final_high_poly_path}")

                return (baked_texture_tensor, debug_high_poly_mesh,)
            
            except Exception as e:
                print(f"VertexBake failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return (torch.zeros((1, int(resolution), int(resolution), 3), dtype=torch.float32), high_poly_mesh,)

def _run_blender_script(script_path):
    import subprocess
    blender_path = os.environ.get("BLENDER_EXE", "blender")
    print(f"Attempting to run Blender using path: {blender_path}")
    result = subprocess.run([blender_path, "-b", "--factory-startup", "-P", script_path], capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print("--- Blender stdout ---")
        print(result.stdout)
        print("--- Blender stderr ---", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Blender execution failed with code {result.returncode}. See console for details.")

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