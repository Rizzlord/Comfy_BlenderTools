import os
import sys
import tempfile
import traceback
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
import re
from PIL import Image, ImageFilter, ImageDraw
from .utils import (
    _run_blender_script,
    generate_uv_preview_shared,
    get_blender_clean_mesh_func_script,
)


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
                "ao_strength": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "bake_thickness": ("BOOLEAN", {"default": True}),
                "thickness_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
                ),
                "bake_cavity": ("BOOLEAN", {"default": True}),
                "cavity_contrast": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "cage_extrusion": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_ray_distance": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "margin": ("INT", {"default": 1024, "min": 0, "max": 2048}),
                "use_high_poly_textures": ("BOOLEAN", {"default": False}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "blur_atc": ("BOOLEAN", {"default": True}),
                "atc_blur_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 150.0, "step": 0.1},
                ),
                "blur_mr": ("BOOLEAN", {"default": False}),
                "mr_blur_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 150.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = (
        "TRIMESH",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "low_poly_mesh",
        "albedo_map",
        "normal_map",
        "mr_map",
        "ao_map",
        "thickness_map",
        "cavity_map",
        "atc_map",
    )
    FUNCTION = "bake"
    CATEGORY = "Comfy_BlenderTools"

    def bake(
        self,
        high_poly_mesh,
        low_poly_mesh,
        resolution,
        bake_albedo,
        bake_mr_map,
        bake_normal,
        bake_ao,
        ao_strength,
        bake_thickness,
        thickness_strength,
        bake_cavity,
        cavity_contrast,
        cage_extrusion,
        max_ray_distance,
        margin,
        use_high_poly_textures,
        use_gpu,
        blur_atc,
        atc_blur_strength,
        blur_mr,
        mr_blur_strength,
    ):

        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        def get_material(mesh):
            if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
                return mesh.visual.material
            return None

        high_material = get_material(high_poly_mesh)
        low_material = get_material(low_poly_mesh)

        material_preference = (
            [high_material, low_material]
            if use_high_poly_textures
            else [low_material, high_material]
        )

        original_low_uv = (
            np.array(low_poly_mesh.visual.uv, copy=True)
            if hasattr(low_poly_mesh, "visual")
            and hasattr(low_poly_mesh.visual, "uv")
            and low_poly_mesh.visual.uv is not None
            else None
        )

        def tensor_from_materials(materials, attr):
            for material in materials:
                if material is None or not hasattr(material, attr):
                    continue
                texture = getattr(material, attr)
                if texture is None:
                    continue
                pil_img = texture.convert("RGB")
                img_array = np.array(pil_img).astype(np.float32) / 255.0
                return torch.from_numpy(img_array)[None,]
            return None

        def select_texture_image(attr):
            for material in material_preference:
                if material is None or not hasattr(material, attr):
                    continue
                texture = getattr(material, attr)
                if texture is None:
                    continue
                return texture.convert("RGB")
            return None

        albedo_map_tensor = tensor_from_materials(
            material_preference, "baseColorTexture"
        )
        mr_map_tensor = tensor_from_materials(
            material_preference, "metallicRoughnessTexture"
        )
        normal_map_tensor = tensor_from_materials(material_preference, "normalTexture")
        ao_map_tensor = tensor_from_materials(material_preference, "occlusionTexture")
        thickness_map_tensor = None
        cavity_map_tensor = None

        selected_albedo_image = select_texture_image("baseColorTexture")
        selected_mr_image = select_texture_image("metallicRoughnessTexture")

        original_material = (
            low_poly_mesh.visual.material
            if hasattr(low_poly_mesh.visual, "material")
            else None
        )

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
                "high_poly_path": high_poly_path,
                "low_poly_path": low_poly_path,
                "final_low_poly_path": final_low_poly_path,
                "temp_dir": temp_dir,
                "bake_albedo": bake_albedo,
                "bake_mr": bake_mr_map,
                "bake_normal": bake_normal,
                "bake_ao": bake_ao,
                "ao_strength": ao_strength,
                "bake_thickness": bake_thickness,
                "thickness_strength": thickness_strength,
                "bake_cavity": bake_cavity,
                "cavity_contrast": cavity_contrast,
                "resolution": int(resolution),
                "cage_extrusion": cage_extrusion,
                "max_ray_distance": max_ray_distance,
                "margin": margin,
                "use_high_poly_textures": use_high_poly_textures,
                "use_gpu": use_gpu,
                "albedo_texture_path": albedo_texture_path,
                "mr_texture_path": mr_texture_path,
            }

            clean_mesh_func_script = get_blender_clean_mesh_func_script()

            script = f"""
{clean_mesh_func_script}
import sys, traceback
try:
    import bpy, os, bmesh, numpy as np
    from mathutils import Vector
except Exception as e:
    print(f"Failed to import Blender dependencies: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
p = {{ {", ".join(f'"{k}": r"{v}"' if isinstance(v, str) else f'"{k}": {v}' for k, v in params.items())} }}
auto_bake_settings = {{"cage_extrusion": None, "max_ray_distance": None}}
shared_bake_cage = None

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
    bpy.context.scene.render.use_compositing = True
    setup_gpu()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_obj_as_single_mesh(filepath, object_name):
    existing_names = {{obj.name for obj in bpy.context.scene.objects}}
    bpy.ops.wm.obj_import(filepath=filepath)
    imported_meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.name not in existing_names and obj.type == 'MESH'
    ]
    if not imported_meshes:
        raise Exception(f"No mesh objects were imported from: {{filepath}}")

    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_meshes[0]
    bpy.context.view_layer.update()

    if len(imported_meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = object_name
    sanitize_for_bake(obj, 0.0001)
    return obj

def get_world_bounds(obj):
    corners = np.array(
        [[coord for coord in (obj.matrix_world @ Vector(corner))] for corner in obj.bound_box],
        dtype=np.float64,
    )
    return corners.min(axis=0), corners.max(axis=0)

def get_object_diagonal(obj):
    bounds_min, bounds_max = get_world_bounds(obj)
    return float(np.linalg.norm(bounds_max - bounds_min))

def align_meshes(high_obj, low_obj):
    low_obj.rotation_mode = high_obj.rotation_mode
    low_obj.location = high_obj.location.copy()
    if hasattr(high_obj, "rotation_quaternion") and high_obj.rotation_mode == 'QUATERNION':
        low_obj.rotation_quaternion = high_obj.rotation_quaternion.copy()
    else:
        low_obj.rotation_euler = high_obj.rotation_euler.copy()
    low_obj.scale = high_obj.scale.copy()
    bpy.context.view_layer.update()

    high_min, high_max = get_world_bounds(high_obj)
    low_min, low_max = get_world_bounds(low_obj)
    high_center = (high_min + high_max) * 0.5
    low_center = (low_min + low_max) * 0.5
    offset = high_center - low_center
    if float(np.linalg.norm(offset)) > 1e-6:
        low_obj.location.x += float(offset[0])
        low_obj.location.y += float(offset[1])
        low_obj.location.z += float(offset[2])
        bpy.context.view_layer.update()

    high_min, high_max = get_world_bounds(high_obj)
    low_min, low_max = get_world_bounds(low_obj)
    print(
        "Bake alignment centers high="
        + str(((high_min + high_max) * 0.5).tolist())
        + " low="
        + str(((low_min + low_max) * 0.5).tolist())
    )

def sample_surface_distances(source_obj, target_obj, sample_limit=2048):
    distances = []
    source_vertices = getattr(getattr(source_obj, "data", None), "vertices", None)
    if source_vertices is None or len(source_vertices) == 0:
        return distances

    vertex_count = len(source_vertices)
    sample_count = min(vertex_count, int(sample_limit))
    if sample_count <= 0:
        return distances

    if sample_count == vertex_count:
        sample_indices = range(vertex_count)
    else:
        sample_indices = np.linspace(0, vertex_count - 1, num=sample_count, dtype=np.int32)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_eval = target_obj.evaluated_get(depsgraph)
    target_world_inv = target_eval.matrix_world.inverted()

    for idx in sample_indices:
        try:
            source_vertex = source_vertices[int(idx)]
            world_co = source_obj.matrix_world @ source_vertex.co
            target_local_co = target_world_inv @ world_co
            hit_result = target_eval.closest_point_on_mesh(target_local_co)
            if not hit_result:
                continue
            hit = bool(hit_result[0])
            if not hit:
                continue
            hit_location_world = target_eval.matrix_world @ hit_result[1]
            distances.append(float((world_co - hit_location_world).length))
        except Exception:
            continue

    return distances

def estimate_bake_settings(high_obj, low_obj):
    diag = max(get_object_diagonal(high_obj), get_object_diagonal(low_obj), 1e-4)
    min_cage = max(diag * 0.001, 0.0005)
    min_ray = max(diag * 0.0015, min_cage * 1.25)

    sampled_distances = sample_surface_distances(low_obj, high_obj)
    if sampled_distances:
        sampled_array = np.asarray(sampled_distances, dtype=np.float64)
        p90 = float(np.quantile(sampled_array, 0.90))
        p98 = float(np.quantile(sampled_array, 0.98))
        auto_cage = max(min_cage, p90 * 1.1, p98 * 1.02)
        auto_ray = max(min_ray, p98 * 1.35, auto_cage * 1.2)
        print(
            "Auto bake settings from sampled distances: "
            + f"p90={{p90:.6f}}, p98={{p98:.6f}}, "
            + f"cage={{auto_cage:.6f}}, ray={{auto_ray:.6f}}"
        )
    else:
        auto_cage = max(min_cage, diag * 0.0025)
        auto_ray = max(min_ray, auto_cage * 1.5)
        print(
            "Auto bake settings fallback from bounds: "
            + f"diag={{diag:.6f}}, cage={{auto_cage:.6f}}, ray={{auto_ray:.6f}}"
        )

    if float(p['cage_extrusion']) > 0.0:
        auto_cage = float(p['cage_extrusion'])
    if float(p['max_ray_distance']) > 0.0:
        auto_ray = float(p['max_ray_distance'])

    return {{
        "cage_extrusion": float(auto_cage),
        "max_ray_distance": float(auto_ray),
    }}

def import_meshes():
    high_obj = import_obj_as_single_mesh(p['high_poly_path'], "HighPoly")
    low_obj = import_obj_as_single_mesh(p['low_poly_path'], "LowPoly")
    align_meshes(high_obj, low_obj)

    if not low_obj.data.uv_layers:
        raise Exception("Low-poly mesh has no UV map. Please use the BlenderUnwrap node first.")

    return high_obj, low_obj

def setup_lowpoly_material(low_obj):
    mat = bpy.data.materials.new(name="BakeMaterial")
    mat.use_nodes = True
    if low_obj.data.materials: low_obj.data.materials[0] = mat
    else: low_obj.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.name = "TextureBake_Target"
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
            try:
                separate = nodes.new('ShaderNodeSeparateColor')
                separate.mode = 'RGB'
            except Exception:
                separate = nodes.new('ShaderNodeSeparateRGB')
            links.new(mr_node.outputs['Color'], separate.inputs[0])
            links.new(separate.outputs[1], principled.inputs['Roughness'])
            links.new(separate.outputs[2], principled.inputs['Metallic'])

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

def get_compositor_tree(scene):
    # Blender 5.0 removed scene.node_tree; compositor lives in compositing_node_group.
    if hasattr(scene, "compositing_node_group"):
        tree = getattr(scene, "compositing_node_group", None)
        if tree is None:
            tree = bpy.data.node_groups.new("TextureBakeCompositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        if hasattr(scene, "use_nodes"):
            scene.use_nodes = True
        return tree
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        return scene.node_tree
    return None

def sanitize_for_bake(obj, merge_distance):
    # Clean mesh for baking: merge, drop non-UV attributes, clear split normals, smooth.
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.object.mode_set(mode='EDIT')
        if merge_distance > 0.0:
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
                bmesh.update_edit_mesh(obj.data)
            except Exception as exc_merge:
                print(f"Merge by distance failed: {{exc_merge}}")
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except Exception as exc_norm:
            print(f"Failed to clear split normals: {{exc_norm}}")
        try:
            uv_names = {{uv.name for uv in getattr(obj.data, "uv_layers", [])}}
            for attr in list(getattr(obj.data, "attributes", [])):
                if getattr(attr, "is_internal", False):
                    continue
                if attr.name in uv_names:
                    continue
                if attr.name in {{"position", "normal", "material_index"}}:
                    continue
                try:
                    obj.data.attributes.remove(attr)
                except Exception as exc_attr:
                    print(f"Failed to remove attribute {{attr.name}}: {{exc_attr}}")
        except Exception as exc_attrblock:
            print(f"Attribute cleanup failed: {{exc_attrblock}}")
    finally:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    try:
        bpy.ops.object.shade_smooth()
    except Exception as exc_smooth:
        print(f"Shade smooth failed: {{exc_smooth}}")

def duplicate_clean_for_normal(obj, name_suffix, merge_distance=0.0001):
    try:
        dup = obj.copy()
        if obj.data:
            dup.data = obj.data.copy()
        dup.name = name_suffix
        bpy.context.collection.objects.link(dup)
    except Exception as exc:
        print(f"Failed to duplicate {{name_suffix}} for normal bake: {{exc}}")
        return None

    sanitize_for_bake(dup, merge_distance)
    return dup

def configure_normal_bake_mesh(obj, clear_sharp=False):
    if obj is None:
        return

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        if clear_sharp:
            try:
                bpy.ops.mesh.mark_sharp(clear=True)
            except Exception as exc_sharp:
                print(f"Failed to clear sharp edges on {{obj.name}}: {{exc_sharp}}")
        try:
            bpy.ops.mesh.normals_make_consistent(inside=False)
        except Exception as exc_normals:
            print(f"Failed to recalculate normals on {{obj.name}}: {{exc_normals}}")
    finally:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    data = getattr(obj, "data", None)
    if data is not None:
        if hasattr(data, "use_auto_smooth"):
            data.use_auto_smooth = True
        if hasattr(data, "auto_smooth_angle"):
            data.auto_smooth_angle = float(np.pi)

    try:
        bpy.ops.object.shade_smooth_by_angle(angle=float(np.pi), keep_sharp_edges=True)
    except Exception:
        try:
            bpy.ops.object.shade_smooth()
        except Exception as exc_smooth:
            print(f"Failed to smooth shading on {{obj.name}}: {{exc_smooth}}")

def apply_object_modifier(obj, modifier_name):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    bpy.ops.object.modifier_apply(modifier=modifier_name)

def create_explicit_bake_cage(low_source_obj, high_source_obj, name_suffix, cage_offset):
    if low_source_obj is None or high_source_obj is None:
        return None

    cage = duplicate_clean_for_normal(low_source_obj, name_suffix, 0.0)
    if cage is None:
        return None

    cage.hide_render = True
    try:
        cage.display_type = 'WIRE'
    except Exception:
        pass

    configure_normal_bake_mesh(cage, clear_sharp=False)

    project_modifier = None
    try:
        project_modifier = cage.modifiers.new(name="BakeCageProject", type='SHRINKWRAP')
        project_modifier.target = high_source_obj
        if hasattr(project_modifier, "wrap_method"):
            try:
                project_modifier.wrap_method = 'PROJECT'
            except Exception:
                project_modifier.wrap_method = 'NEAREST_SURFACEPOINT'
        if hasattr(project_modifier, "wrap_mode"):
            try:
                project_modifier.wrap_mode = 'OUTSIDE'
            except Exception:
                pass
        for axis_name in ("x", "y", "z"):
            attr = f"use_project_{{axis_name}}"
            if hasattr(project_modifier, attr):
                setattr(project_modifier, attr, True)
        if hasattr(project_modifier, "use_positive_direction"):
            project_modifier.use_positive_direction = True
        if hasattr(project_modifier, "use_negative_direction"):
            project_modifier.use_negative_direction = True
        if hasattr(project_modifier, "cull_face"):
            try:
                project_modifier.cull_face = 'OFF'
            except Exception:
                pass
        if hasattr(project_modifier, "offset"):
            project_modifier.offset = max(float(cage_offset) * 0.15, 0.0002)
        apply_object_modifier(cage, project_modifier.name)
    except Exception as exc_project:
        print(f"Failed to project bake cage {{name_suffix}}: {{exc_project}}")

    push_amount = max(float(cage_offset) * 0.85, 0.0005)
    try:
        bpy.ops.object.select_all(action='DESELECT')
        cage.select_set(True)
        bpy.context.view_layer.objects.active = cage
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.transform.shrink_fatten(value=push_amount, use_even_offset=True)
        except TypeError:
            bpy.ops.transform.shrink_fatten(value=push_amount)
    except Exception as exc_push:
        print(f"Failed to inflate bake cage {{name_suffix}}: {{exc_push}}")
    finally:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    configure_normal_bake_mesh(cage, clear_sharp=False)
    return cage

def ensure_bake_target_node(target_obj, bake_image):
    if target_obj is None:
        raise RuntimeError("Bake target object is missing.")

    if not target_obj.data.materials:
        target_mat, target_tex_node = setup_lowpoly_material(target_obj)
    else:
        target_mat = target_obj.data.materials[0]
        if target_mat is None:
            target_mat, target_tex_node = setup_lowpoly_material(target_obj)
        else:
            target_mat.use_nodes = True
            nodes = target_mat.node_tree.nodes
            links = target_mat.node_tree.links

            output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if output is None:
                output = nodes.new('ShaderNodeOutputMaterial')

            principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if principled is None:
                principled = nodes.new('ShaderNodeBsdfPrincipled')

            has_surface_link = any(
                link.from_node == principled
                and link.to_node == output
                and link.to_socket == output.inputs['Surface']
                for link in links
            )
            if not has_surface_link:
                links.new(principled.outputs['BSDF'], output.inputs['Surface'])

            target_tex_node = nodes.get("TextureBake_Target")
            if target_tex_node is None or target_tex_node.type != 'TEX_IMAGE':
                target_tex_node = nodes.new('ShaderNodeTexImage')
                target_tex_node.name = "TextureBake_Target"

    target_tex_node.image = bake_image
    target_mat.node_tree.nodes.active = target_tex_node
    return target_tex_node

def process_and_save_map(image, output_path, invert=False, strength=1.0, white_bg=False):
    width, height = image.size[0], image.size[1]
    channels = getattr(image, "channels", 4)
    pixels = np.array(image.pixels[:], dtype=np.float32)
    if pixels.size != width * height * channels:
        raise RuntimeError("Unexpected pixel buffer size while saving baked map.")
    pixels = pixels.reshape((height, width, channels))

    rgb = pixels[..., :3]
    alpha = pixels[..., 3:4] if channels > 3 else np.ones_like(rgb[..., :1])

    if invert:
        rgb = 1.0 - rgb
    if strength != 1.0:
        rgb = np.clip(rgb, 0.0, 1.0) ** float(strength)
    if white_bg:
        rgb = rgb * alpha + (1.0 - alpha)

    rgb = np.clip(rgb, 0.0, 1.0)
    rgba = np.concatenate([rgb, alpha], axis=2)
    out_img = bpy.data.images.new("ProcessedBake", width=width, height=height, alpha=True)
    out_img.pixels = rgba.astype(np.float32).reshape(-1).tolist()
    out_img.filepath_raw = output_path
    out_img.file_format = "PNG"
    out_img.save()
    bpy.data.images.remove(out_img)

def execute_bake(bake_type, tex_node, source_obj=None, target_obj=None, cage_obj=None):
    res = p['resolution']
    bake_image = bpy.data.images.new(name=f"{{bake_type}}_BakeImage", width=res, height=res, alpha=True)

    tgt = target_obj if target_obj is not None else low_obj
    src = source_obj if source_obj is not None else high_obj
    effective_cage = cage_obj if cage_obj is not None else shared_bake_cage
    effective_cage_extrusion = auto_bake_settings.get('cage_extrusion') or float(p['cage_extrusion'])
    effective_max_ray_distance = auto_bake_settings.get('max_ray_distance') or float(p['max_ray_distance'])
    ensure_bake_target_node(tgt, bake_image)
    bpy.ops.object.select_all(action='DESELECT')
    if src is not None:
        src.select_set(True)
    if tgt is not None:
        tgt.select_set(True)
        bpy.context.view_layer.objects.active = tgt
    bpy.context.view_layer.update()

    if src is not None and tgt is not None:
        src_min, src_max = get_world_bounds(src)
        tgt_min, tgt_max = get_world_bounds(tgt)
        src_center = (src_min + src_max) * 0.5
        tgt_center = (tgt_min + tgt_max) * 0.5
        print(
            f"Executing {{bake_type}} bake. Source center={{src_center.tolist()}}, "
            f"target center={{tgt_center.tolist()}}"
        )
    
    bake_kwargs = {{
        'type': bake_type, 'use_selected_to_active': True,
        'margin': p['margin'], 'use_clear': True,
    }}
    if effective_cage is not None:
        bake_kwargs['use_cage'] = True
        bake_kwargs['cage_object'] = effective_cage.name
    else:
        bake_kwargs['cage_extrusion'] = effective_cage_extrusion
    if effective_max_ray_distance > 0.0 and effective_cage is None:
        bake_kwargs['max_ray_distance'] = effective_max_ray_distance
    if bake_type == 'NORMAL':
        bake_kwargs['normal_space'] = 'TANGENT'

    bpy.ops.object.bake(**bake_kwargs)
    return bake_image

try:
    setup_scene()
    high_obj, low_obj = import_meshes()
    apply_highpoly_textures(high_obj)
    auto_bake_settings = estimate_bake_settings(high_obj, low_obj)
    shared_bake_cage = create_explicit_bake_cage(
        low_obj,
        high_obj,
        "LowPoly_BakeCage",
        auto_bake_settings['cage_extrusion'],
    )
    low_poly_mat, tex_node = setup_lowpoly_material(low_obj)

    if p['bake_albedo']:
        emission_overrides = enable_emission_bake(high_obj)
        diffuse_image = None
        try:
            diffuse_image = execute_bake('EMIT', tex_node, source_obj=high_obj, target_obj=low_obj)
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
                mr_image = execute_bake('EMIT', tex_node, source_obj=high_obj, target_obj=low_obj)
                output_path = os.path.join(p['temp_dir'], "mr_map.png")
                mr_image.filepath_raw = output_path
                mr_image.file_format = 'PNG'
                mr_image.save()
        finally:
            if mr_image is not None:
                bpy.data.images.remove(mr_image)
            restore_emission_bake(mr_overrides)

    if p['bake_normal']:
        normal_high_obj = duplicate_clean_for_normal(high_obj, "HighPoly_NormalBake", 0.0001)
        normal_low_obj = duplicate_clean_for_normal(low_obj, "LowPoly_NormalBake", 0.0)
        normal_source_obj = normal_high_obj or high_obj
        normal_target_obj = normal_low_obj or low_obj
        configure_normal_bake_mesh(normal_source_obj, clear_sharp=False)
        configure_normal_bake_mesh(normal_target_obj, clear_sharp=False)
        normal_cage_obj = create_explicit_bake_cage(
            normal_target_obj,
            normal_source_obj,
            "LowPoly_NormalCage",
            max(float(auto_bake_settings['cage_extrusion']), 0.001),
        )
        cleanup_objs = [
            o for o in (normal_high_obj, normal_low_obj, normal_cage_obj) if o is not None
        ]
        try:
            normal_image = execute_bake(
                'NORMAL',
                tex_node,
                source_obj=normal_source_obj,
                target_obj=normal_target_obj,
                cage_obj=normal_cage_obj,
            )
            output_path = os.path.join(p['temp_dir'], "normal_map.png")
            normal_image.filepath_raw = output_path
            normal_image.file_format = 'PNG'; normal_image.save()
            bpy.data.images.remove(normal_image)
        finally:
            for obj in cleanup_objs:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception as exc:
                    print(f"Failed to remove normal bake helper {{obj.name}}: {{exc}}")

    if p['bake_ao']:
        ao_image = execute_bake('AO', tex_node, source_obj=high_obj, target_obj=low_obj)
        process_and_save_map(ao_image, os.path.join(p['temp_dir'], "ao_map.png"), 
                             strength=p['ao_strength'], white_bg=True)
        bpy.data.images.remove(ao_image)

    if p['bake_thickness']:
        thickness_image = execute_bake('AO', tex_node, source_obj=high_obj, target_obj=low_obj)
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
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj

    log_path = os.path.join(p['temp_dir'], "export_debug.txt")

    def log(msg):
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(str(msg) + "\\n")
        except Exception:
            pass

    log('Scene objects: ' + ', '.join([o.name for o in bpy.context.scene.objects]))
    log('Selected objects: ' + ', '.join([o.name for o in bpy.context.selected_objects]))
    log('Active object: ' + str(getattr(bpy.context.view_layer.objects.active, 'name', None)))
    log('Current mode: ' + str(bpy.context.mode))

    os.makedirs(os.path.dirname(p['final_low_poly_path']), exist_ok=True)

    def do_export(use_selection=True):
        try:
            res = bpy.ops.export_scene.gltf(
                filepath=p['final_low_poly_path'],
                export_format='GLB',
                use_selection=use_selection
            )
        except Exception as e:
            log(f"Export exception (use_selection={{use_selection}}): {{e}}")
            return None
        log(f"Export result (use_selection={{use_selection}}): {{res}}")
        return res

    export_result = do_export(True)
    if (not export_result or 'FINISHED' not in export_result) or (not os.path.exists(p['final_low_poly_path'])):
        log("Primary export missing or failed; retrying with use_selection=False")
        export_result = do_export(False)

    if not export_result or 'FINISHED' not in export_result:
        log('Temp dir listing: ' + ', '.join(os.listdir(p['temp_dir'])))
        raise RuntimeError(f"glTF export failed with result: {{export_result}}")
    if not os.path.exists(p['final_low_poly_path']):
        log('Temp dir listing: ' + ', '.join(os.listdir(p['temp_dir'])))
        raise RuntimeError(f"glTF export did not produce file: {{p['final_low_poly_path']}}")

    log('Exported GLB: ' + p['final_low_poly_path'])

    if shared_bake_cage is not None:
        try:
            bpy.data.objects.remove(shared_bake_cage, do_unlink=True)
        except Exception as exc:
            print(f"Failed to remove shared bake cage: {{exc}}")

    sys.exit(0)
except Exception as e:
    print(f"Blender script failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
            with open(script_path, "w") as f:
                f.write(script)

            _run_blender_script(script_path)

            # Try to read Blender-side export debug log if present
            log_path = os.path.join(temp_dir, "export_debug.txt")
            blender_export_log = None
            try:
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8") as lf:
                        blender_export_log = lf.read()
            except Exception:
                blender_export_log = None

            temp_listing = []
            try:
                temp_listing = os.listdir(temp_dir)
            except Exception:
                temp_listing = []

            mesh_path = final_low_poly_path
            if not os.path.exists(mesh_path):
                alt_glb = next(
                    (os.path.join(temp_dir, f) for f in temp_listing if f.lower().endswith(".glb")),
                    None,
                )
                alt_gltf = next(
                    (os.path.join(temp_dir, f) for f in temp_listing if f.lower().endswith(".gltf")),
                    None,
                )
                if alt_glb and os.path.exists(alt_glb):
                    mesh_path = alt_glb
                    print(f"Primary GLB missing, using alternative: {mesh_path}")
                elif alt_gltf and os.path.exists(alt_gltf):
                    mesh_path = alt_gltf
                    print(f"Primary GLB missing, using GLTF: {mesh_path}")
                else:
                    print("Temp dir contents after Blender:", temp_listing)
                    if blender_export_log:
                        print("Blender export log:\n", blender_export_log)
                    raise RuntimeError(f"Blender did not produce the expected file: {final_low_poly_path}")

            res_int = int(resolution)

            def load_image_as_tensor(path):
                if not os.path.exists(path):
                    return None
                img = Image.open(path).convert("RGB")
                return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

            albedo_bake_tensor = load_image_as_tensor(
                os.path.join(temp_dir, "albedo_map.png")
            )
            if albedo_bake_tensor is not None:
                albedo_map_tensor = albedo_bake_tensor

            mr_bake_tensor = load_image_as_tensor(os.path.join(temp_dir, "mr_map.png"))
            if mr_bake_tensor is not None:
                mr_map_tensor = mr_bake_tensor
            if blur_mr and mr_map_tensor is not None:
                mr_map_tensor = self._blur_image_tensor(mr_map_tensor, mr_blur_strength)

            normal_map_tensor = load_image_as_tensor(
                os.path.join(temp_dir, "normal_map.png")
            )
            ao_map_tensor = load_image_as_tensor(os.path.join(temp_dir, "ao_map.png"))
            thickness_map_tensor = load_image_as_tensor(
                os.path.join(temp_dir, "thickness_map.png")
            )
            cavity_map_tensor = load_image_as_tensor(
                os.path.join(temp_dir, "cavity_map.png")
            )

            atc_map_tensor = None
            if any([bake_ao, bake_thickness, bake_cavity]):
                black_channel = torch.zeros(
                    (1, res_int, res_int, 1), dtype=torch.float32
                )

                r_channel = (
                    ao_map_tensor[:, :, :, 0:1]
                    if ao_map_tensor is not None
                    else black_channel
                )
                g_channel = (
                    thickness_map_tensor[:, :, :, 0:1]
                    if thickness_map_tensor is not None
                    else black_channel
                )
                b_channel = (
                    cavity_map_tensor[:, :, :, 0:1]
                    if cavity_map_tensor is not None
                    else black_channel
                )

                atc_map_tensor = torch.cat([r_channel, g_channel, b_channel], dim=-1)
            if atc_map_tensor is not None and blur_atc:
                atc_map_tensor = self._blur_image_tensor(
                    atc_map_tensor, atc_blur_strength
                )

            final_mesh = trimesh_loader.load(mesh_path, force="mesh", process=False)
            uv_data = final_mesh.visual.uv if hasattr(final_mesh.visual, "uv") else None
            if uv_data is None and original_low_uv is not None:
                if original_low_uv.shape[0] == final_mesh.vertices.shape[0]:
                    uv_data = original_low_uv

            def tensor_to_pil(tensor):
                if tensor is None:
                    return None
                array = np.clip(tensor[0].cpu().numpy() * 255.0, 0, 255).astype(
                    np.uint8
                )
                return Image.fromarray(array)

            final_material = (
                original_material
                if original_material
                else trimesh_loader.visual.material.PBRMaterial()
            )

            base_color_texture = tensor_to_pil(albedo_map_tensor)
            metallic_roughness_texture = tensor_to_pil(mr_map_tensor)
            normal_texture = tensor_to_pil(normal_map_tensor)
            ao_texture = tensor_to_pil(ao_map_tensor)

            if base_color_texture is not None:
                final_material.baseColorTexture = base_color_texture
            if metallic_roughness_texture is not None:
                final_material.metallicRoughnessTexture = metallic_roughness_texture
            if normal_texture is not None:
                final_material.normalTexture = normal_texture
            if ao_texture is not None:
                final_material.occlusionTexture = ao_texture

            if uv_data is not None:
                final_mesh.visual = trimesh_loader.visual.texture.TextureVisuals(
                    uv=uv_data, material=final_material
                )
            else:
                final_mesh.visual.material = final_material

            return (
                final_mesh,
                albedo_map_tensor if albedo_map_tensor is not None else dummy_image,
                normal_map_tensor if normal_map_tensor is not None else dummy_image,
                mr_map_tensor if mr_map_tensor is not None else dummy_image,
                ao_map_tensor if ao_map_tensor is not None else dummy_image,
                (
                    thickness_map_tensor
                    if thickness_map_tensor is not None
                    else dummy_image
                ),
                cavity_map_tensor if cavity_map_tensor is not None else dummy_image,
                atc_map_tensor if atc_map_tensor is not None else dummy_image,
            )

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
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "apply"
    CATEGORY = "Comfy_BlenderTools"

    def apply(self, mesh, albedo_map=None, normal_map=None, mr_map=None, ao_map=None):
        # Keep a safe copy of UVs; bail out early with a clear error if missing
        uv_data = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uv_data = np.array(mesh.visual.uv, copy=True)
        if uv_data is None:
            raise Exception(
                "Mesh must have UV coordinates to apply materials. Use the BlenderUnwrap node first."
            )

        new_mesh = mesh.copy()

        # Copy material if possible, otherwise start with a fresh PBR material
        material = None
        try:
            if (
                hasattr(mesh, "visual")
                and hasattr(mesh.visual, "material")
                and mesh.visual.material is not None
                and hasattr(mesh.visual.material, "copy")
            ):
                material = mesh.visual.material.copy()
        except Exception:
            material = None
        if material is None:
            material = trimesh_loader.visual.material.PBRMaterial()

        def tensor_to_pil(tensor):
            if tensor is None:
                return None
            # Accept (B,H,W,C) or (H,W,C); always use first batch
            arr = tensor
            if tensor.ndim == 4:
                arr = tensor[0]
            i = 255.0 * arr.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # Ensure we always hand back RGB data so PBRMaterial can serialize
            return img.convert("RGB")

        def create_checker_map_pil(size=512, checker_size=64):
            indices = np.arange(size)
            x_checkers = np.floor_divide(indices, checker_size)
            y_checkers = np.floor_divide(indices, checker_size)[:, np.newaxis]
            pattern = (x_checkers + y_checkers) % 2

            black = [0, 0, 0]
            purple = [255, 0, 255]

            img_array = np.array([black, purple], dtype=np.uint8)[pattern]
            return Image.fromarray(img_array, "RGB")

        if albedo_map is not None:
            material.baseColorTexture = tensor_to_pil(albedo_map)
        elif (
            not hasattr(material, "baseColorTexture")
            or material.baseColorTexture is None
        ):
            material.baseColorTexture = create_checker_map_pil()

        if normal_map is not None:
            material.normalTexture = tensor_to_pil(normal_map)

        if mr_map is not None:
            material.metallicRoughnessTexture = tensor_to_pil(mr_map)

        if ao_map is not None:
            material.occlusionTexture = tensor_to_pil(ao_map)

        # Re-attach UVs explicitly to avoid losing them on the copy
        new_mesh.visual = trimesh_loader.visual.texture.TextureVisuals(
            uv=uv_data, material=material
        )

        return (new_mesh,)


class ExtractMaterial:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "uv_preview_resolution": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 256}),
                "export_uv_layout": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo_map", "normal_map", "mr_map", "ao_map", "uv_preview")
    FUNCTION = "extract"
    CATEGORY = "Comfy_BlenderTools"

    def generate_uv_preview(self, mesh, res_x, res_y, export_layout):
        return generate_uv_preview_shared(mesh, res_x, res_y, export_layout)

    def extract(self, mesh, uv_preview_resolution=1024, export_uv_layout=True):
        dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        def pil_to_tensor(pil_img):
            if pil_img is None:
                return dummy_image

            pil_img = pil_img.convert("RGB")
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(img_array)[None,]

        if not hasattr(mesh, "visual") or not hasattr(mesh.visual, "material"):
            return (dummy_image, dummy_image, dummy_image, dummy_image, dummy_image)

        mat = mesh.visual.material

        albedo_map = pil_to_tensor(getattr(mat, "baseColorTexture", None))
        normal_map = pil_to_tensor(getattr(mat, "normalTexture", None))
        mr_map = pil_to_tensor(getattr(mat, "metallicRoughnessTexture", None))
        ao_map = pil_to_tensor(getattr(mat, "occlusionTexture", None))

        uv_preview = self.generate_uv_preview(mesh, uv_preview_resolution, uv_preview_resolution, export_uv_layout)

        return (albedo_map, normal_map, mr_map, ao_map, uv_preview)
