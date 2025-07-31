import os
import subprocess
import sys
import tempfile
import trimesh as trimesh_loader
import folder_paths
import numpy as np
import torch
from PIL import Image
import math

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

# --- Custom Node: Render Depth Map (Unchanged) ---
class BlenderRenderDepthMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "resolution": (["512", "1024", "2048"], {"default": "1024"}),
                "camera_distance": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 200.0, "step": 1.0}),
                "camera_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "camera_elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "camera_fov": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 120.0, "step": 1.0}),
                "depth_range": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "model_rotation_x": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "model_rotation_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "model_rotation_z": ("FLOAT", {"default": 180.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "multi_view": ("BOOLEAN", {"default": False}),
                "camera_count": (["2", "6", "10"], {"default": "2"}),
            }
        }
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("IMAGE", "IMAGE"), ("depth_map", "multi_view_images"), "render_depth", "Comfy_BlenderTools"

    def render_depth(self, trimesh, use_gpu, resolution, camera_distance, camera_rotation, camera_elevation, camera_fov, depth_range, depth_strength, model_rotation_x, model_rotation_y, model_rotation_z, multi_view, camera_count):
        camera_count_int = int(camera_count)
        blender_path = get_blender_path()
        res_int = int(resolution)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "input.obj")
            script_path = os.path.join(temp_dir, "depth_render_script.py")
            
            trimesh.export(file_obj=input_mesh_path)

            script = f"""
import bpy, math, sys, os, mathutils
params={{'input_mesh':r'{input_mesh_path}','use_gpu':{use_gpu},'resolution':{res_int},'cam_dist':{camera_distance},'cam_rot':{camera_rotation},'cam_elev':{camera_elevation},'cam_fov':{camera_fov},'depth_range':{depth_range},'depth_strength':{depth_strength},'model_rot_x':{model_rotation_x},'model_rot_y':{model_rotation_y},'model_rot_z':{model_rotation_z},'multi_view':{multi_view},'camera_count':{camera_count_int}}}
try:
    scene=bpy.context.scene; scene.render.engine='CYCLES'
    if params['use_gpu']:
        scene.cycles.device='GPU'; prefs=bpy.context.preferences.addons['cycles'].preferences; backend_preference=['OPTIX','HIP','METAL','ONEAPI','CUDA']; chosen_backend=''
        for backend in backend_preference:
            if prefs.get_devices_for_type(backend): prefs.compute_device_type=backend; chosen_backend=backend; break
        if chosen_backend:
            prefs.get_devices()
            for device in prefs.devices: device.use = device.type != 'CPU'
    else: scene.cycles.device='CPU'
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene.render.resolution_x=params['resolution']; scene.render.resolution_y=params['resolution']; scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=True
    bpy.ops.wm.obj_import(filepath=params['input_mesh']); obj=bpy.context.view_layer.objects.active
    if obj and obj.type=='MESH':
        bpy.context.view_layer.objects.active=obj; obj.select_set(True); bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS'); obj.location=(0,0,0)
        z_dimension = obj.dimensions.z; obj.location.z = -(z_dimension * 0.4)
        obj.rotation_euler.x=math.radians(params['model_rot_x']); obj.rotation_euler.y=math.radians(params['model_rot_y']); obj.rotation_euler.z=math.radians(params['model_rot_z']); obj.select_set(False)
    camera_configs=[]
    if params['multi_view']:
        base_elev=params['cam_elev']
        if params['camera_count']==2: camera_configs=[(90,base_elev),(270,base_elev)]
        elif params['camera_count']==6: camera_configs=[(90,base_elev),(270,base_elev),(0,base_elev),(180,base_elev),(0,90),(0,-90)]
        elif params['camera_count']==10: camera_configs=[(90,base_elev),(270,base_elev),(0,base_elev),(180,base_elev),(0,90),(0,-90),(45,base_elev),(135,base_elev),(225,base_elev),(315,base_elev)]
    else: camera_configs=[(params['cam_rot']+90,params['cam_elev'])]
    if'--'in sys.argv: argv=sys.argv[sys.argv.index('--')+1:]; temp_dir_from_args=argv[0]; params['output_image_base']=os.path.join(temp_dir_from_args,"depth_map")
    else: print("Error: Temp dir not passed to Blender.",file=sys.stderr); sys.exit(1)
    for i,(azimuth_deg,elevation_deg) in enumerate(camera_configs):
        if"DepthCam" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["DepthCam"],do_unlink=True)
        azimuth_rad=math.radians(azimuth_deg); elevation_rad=math.radians(elevation_deg)
        cam_data=bpy.data.cameras.new(name="DepthCam"); cam_obj=bpy.data.objects.new("DepthCam",cam_data)
        cam_obj.location=(params['cam_dist']*math.cos(azimuth_rad)*math.cos(elevation_rad),params['cam_dist']*math.sin(azimuth_rad)*math.cos(elevation_rad),params['cam_dist']*math.sin(elevation_rad))
        cam_data.lens_unit='FOV'; cam_data.angle=math.radians(params['cam_fov']); half_range=params['depth_range']/2.0; cam_data.clip_start=max(0.01,params['cam_dist']-half_range); cam_data.clip_end=params['cam_dist']+half_range
        look_at_point = mathutils.Vector((0,0, obj.location.z + z_dimension * 0.5))
        direction = look_at_point - cam_obj.location
        rot_quat=direction.to_track_quat('-Z','Y'); cam_obj.rotation_euler=rot_quat.to_euler(); scene.collection.objects.link(cam_obj); scene.camera=cam_obj
        scene.use_nodes=True; scene.view_layers["ViewLayer"].use_pass_z=True; tree=scene.node_tree
        for node in tree.nodes: tree.nodes.remove(node)
        render_layers=tree.nodes.new('CompositorNodeRLayers'); map_range_node=tree.nodes.new('CompositorNodeMapRange'); map_range_node.inputs[1].default_value=cam_data.clip_start; map_range_node.inputs[2].default_value=cam_data.clip_end; map_range_node.inputs[3].default_value=1.0; map_range_node.inputs[4].default_value=0.0
        math_node=tree.nodes.new('CompositorNodeMath'); math_node.operation='POWER'; math_node.inputs[1].default_value=params['depth_strength']
        mix_node=tree.nodes.new('CompositorNodeMixRGB'); mix_node.inputs[1].default_value=(0,0,0,1)
        composite_node=tree.nodes.new('CompositorNodeComposite')
        tree.links.new(render_layers.outputs['Depth'],map_range_node.inputs['Value']); tree.links.new(map_range_node.outputs['Value'],math_node.inputs[0]); tree.links.new(math_node.outputs['Value'],mix_node.inputs[2]); tree.links.new(render_layers.outputs['Alpha'],mix_node.inputs[0]); tree.links.new(mix_node.outputs['Image'],composite_node.inputs['Image'])
        current_output_path=f"{{params['output_image_base']}}_{{i:02d}}.png"; scene.render.filepath=current_output_path; bpy.ops.render.render(write_still=True)
    sys.exit(0)
except Exception as e: print(f"Blender script failed: {{e}}",file=sys.stderr); sys.exit(1)
"""
            with open(script_path, 'w') as f: f.write(script)
            subprocess.run([blender_path, '--factory-startup', '--background', '--python', script_path, '--', temp_dir], check=True, capture_output=True, text=True)
            output_images = []
            num_images_to_load = camera_count_int if multi_view else 1
            for i in range(num_images_to_load):
                img_path = os.path.join(temp_dir, f"depth_map_{i:02d}.png")
                if not os.path.exists(img_path): continue
                img_pil = Image.open(img_path).convert("RGB")
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(img_np)[None,])
            if not output_images: raise RuntimeError("Blender produced no output images.")
            multi_view_tensor = torch.cat(output_images, dim=0) if output_images else torch.empty(0)
            return (output_images[0], multi_view_tensor)

# --- Custom Node: Texture Projection Bake (with AutoUnwrap option) ---
class BlenderTextureProjection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "images_to_project": ("IMAGE",),
                "AutoUnwrap": ("BOOLEAN", {"default": True}), # New conditional input
                "bake_resolution": (["1024", "2048", "4096", "8192"], {"default": "2048"}),
                "projection_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "camera_distance": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 200.0, "step": 1.0}),
                "camera_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "camera_elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "camera_fov": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 120.0, "step": 1.0}),
                "model_rotation_x": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "model_rotation_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "model_rotation_z": ("FLOAT", {"default": 180.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "multi_view": ("BOOLEAN", {"default": False}),
                "camera_count": (["2", "6", "10"], {"default": "2"}),
            }
        }
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("TRIMESH", "IMAGE"), ("trimesh", "baked_texture"), "project_and_bake", "Comfy_BlenderTools"

    def project_and_bake(self, trimesh, images_to_project, AutoUnwrap, bake_resolution, projection_blend, use_gpu, camera_distance, camera_rotation, camera_elevation, camera_fov, model_rotation_x, model_rotation_y, model_rotation_z, multi_view, camera_count):
        blender_path = get_blender_path()
        camera_count_int = int(camera_count)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_mesh_path = os.path.join(temp_dir, "input.obj")
            output_mesh_path = os.path.join(temp_dir, "unwrapped.obj") 
            output_texture_path = os.path.join(temp_dir, "baked_texture.png")
            script_path = os.path.join(temp_dir, "bake_script.py")
            
            trimesh.export(file_obj=input_mesh_path)
            
            num_images = images_to_project.shape[0]
            for i in range(num_images):
                img_tensor = images_to_project[i]
                pil_image = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
                pil_image.save(os.path.join(temp_dir, f"projection_image_{i:02d}.png"))

            bake_res_int = int(bake_resolution)

            script = f"""
import bpy, math, os, sys, mathutils
params={{'input_mesh':r'{input_mesh_path}','output_mesh':r'{output_mesh_path}','temp_dir':r'{temp_dir}','output_texture':r'{output_texture_path}','bake_res':{bake_res_int},'blend_factor':{projection_blend},'num_images':{num_images},'use_gpu':{use_gpu},'cam_dist':{camera_distance},'cam_rot':{camera_rotation},'cam_elev':{camera_elevation},'cam_fov':{camera_fov},'model_rot_x':{model_rotation_x},'model_rot_y':{model_rotation_y},'model_rot_z':{model_rotation_z},'multi_view':{multi_view},'camera_count':{camera_count_int}, 'auto_unwrap': {AutoUnwrap}}}
try:
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=16
    if params['use_gpu']:
        scene.cycles.device='GPU';prefs=bpy.context.preferences.addons['cycles'].preferences;backend_preference=['OPTIX','HIP','METAL','ONEAPI','CUDA'];chosen_backend=''
        for backend in backend_preference:
            if prefs.get_devices_for_type(backend):prefs.compute_device_type=backend;chosen_backend=backend;break
        if chosen_backend:
            prefs.get_devices()
            for device in prefs.devices:device.use=device.type!='CPU'
    else:scene.cycles.device='CPU'
    bpy.ops.wm.obj_import(filepath=params['input_mesh']);obj=bpy.context.view_layer.objects.active;obj.name="TargetObject"
    if obj and obj.type=='MESH':
        bpy.context.view_layer.objects.active=obj;obj.select_set(True);bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS');obj.location=(0,0,0)
        z_dimension = obj.dimensions.z; obj.location.z = -(z_dimension * 0.4)
        obj.rotation_euler.x=math.radians(params['model_rot_x']);obj.rotation_euler.y=math.radians(params['model_rot_y']);obj.rotation_euler.z=math.radians(params['model_rot_z']);obj.select_set(False)
    projection_bake_image=bpy.data.images.new("ProjectionBake",width=params['bake_res'],height=params['bake_res'],alpha=True)
    final_bake_image=bpy.data.images.new("FinalBake",width=params['bake_res'],height=params['bake_res'],alpha=True)
    mat=bpy.data.materials.new(name="BakeMaterial");obj.data.materials.append(mat);mat.use_nodes=True;nodes=mat.node_tree.nodes;links=mat.node_tree.links;nodes.clear()
    proj_bake_tex_node=nodes.new(type='ShaderNodeTexImage');proj_bake_tex_node.image=projection_bake_image
    principled_bsdf=nodes.new(type='ShaderNodeBsdfPrincipled');output_node=nodes.new(type='ShaderNodeOutputMaterial');links.new(principled_bsdf.outputs['BSDF'],output_node.inputs['Surface'])
    camera_configs=[]
    if params['multi_view']:
        base_elev=params['cam_elev']
        if params['camera_count']==2:camera_configs=[(90,base_elev),(270,base_elev)]
        elif params['camera_count']==6:camera_configs=[(90,base_elev),(270,base_elev),(0,base_elev),(180,base_elev),(0,90),(0,-90)]
        elif params['camera_count']==10:camera_configs=[(90,base_elev),(270,base_elev),(0,base_elev),(180,base_elev),(0,90),(0,-90),(45,base_elev),(135,base_elev),(225,base_elev),(315,base_elev)]
    else:camera_configs=[(params['cam_rot']+90,params['cam_elev'])]
    if len(camera_configs)!=params['num_images']:print(f"Warning: Cam count ({{len(camera_configs)}}) != image count ({{params['num_images']}}).",file=sys.stderr)
    if"ProjectionUV"in obj.data.uv_layers:obj.data.uv_layers.remove(obj.data.uv_layers["ProjectionUV"])
    projection_uv_map=obj.data.uv_layers.new(name="ProjectionUV")
    for i,(azimuth_deg,elevation_deg)in enumerate(camera_configs):
        if i>=params['num_images']:break
        cam_data=bpy.data.cameras.new(name=f"ProjectionCam_{{i}}");cam_obj=bpy.data.objects.new(f"ProjectionCam_{{i}}",cam_data)
        cam_obj.location=(params['cam_dist']*math.cos(math.radians(azimuth_deg))*math.cos(math.radians(elevation_deg)),params['cam_dist']*math.sin(math.radians(azimuth_deg))*math.cos(math.radians(elevation_deg)),params['cam_dist']*math.sin(math.radians(elevation_deg)))
        cam_data.lens_unit='FOV';cam_data.angle=math.radians(params['cam_fov'])
        look_at_point = mathutils.Vector((0, 0, obj.location.z + z_dimension * 0.5))
        direction = look_at_point - cam_obj.location
        rot_quat=direction.to_track_quat('-Z','Y');cam_obj.rotation_euler=rot_quat.to_euler();scene.collection.objects.link(cam_obj)
        obj.data.uv_layers["ProjectionUV"].active=True;bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj;bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
        for window in bpy.context.window_manager.windows:
            screen=window.screen
            for area in screen.areas:
                if area.type=='VIEW_3D':
                    for region in area.regions:
                        if region.type=='WINDOW':
                            override={{'window':window,'screen':screen,'area':area,'region':region,'scene':bpy.context.scene}}
                            with bpy.context.temp_override(**override):
                                area.spaces[0].region_3d.view_perspective='CAMERA';area.spaces[0].camera=cam_obj
                                bpy.context.view_layer.update()
                                bpy.ops.uv.project_from_view(camera_bounds=False,correct_aspect=True,scale_to_bounds=False)
                            break
                    break
            break
        bpy.ops.object.mode_set(mode='OBJECT');img_path=os.path.join(params['temp_dir'],f"projection_image_{{i:02d}}.png");current_proj_image=bpy.data.images.load(img_path);uv_map_node=nodes.new(type='ShaderNodeUVMap');uv_map_node.uv_map="ProjectionUV";current_proj_tex_node=nodes.new(type='ShaderNodeTexImage');current_proj_tex_node.image=current_proj_image;mix_node=nodes.new(type='ShaderNodeMixRGB');mix_node.inputs['Fac'].default_value=params['blend_factor']
        links.new(uv_map_node.outputs['UV'],current_proj_tex_node.inputs['Vector']);links.new(proj_bake_tex_node.outputs['Color'],mix_node.inputs[1]);links.new(current_proj_tex_node.outputs['Color'],mix_node.inputs[2]);links.new(mix_node.outputs['Color'],principled_bsdf.inputs['Base Color'])
        nodes.active=proj_bake_tex_node;bpy.ops.object.bake(type='DIFFUSE',pass_filter={{'COLOR'}},target='IMAGE_TEXTURES')
        nodes.remove(uv_map_node);nodes.remove(current_proj_tex_node);nodes.remove(mix_node);bpy.data.images.remove(current_proj_image);bpy.data.objects.remove(cam_obj)
    
    # --- Conditional Unwrapping and Final Bake ---
    if params['auto_unwrap']:
        print("Auto-unwrapping with Smart UV Project and baking...")
        if"FinalUV"in obj.data.uv_layers:obj.data.uv_layers.remove(obj.data.uv_layers["FinalUV"])
        final_uv_map=obj.data.uv_layers.new(name="FinalUV");final_uv_map.active=True
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.uv.smart_project(angle_limit=66,island_margin=0.02);bpy.ops.object.mode_set(mode='OBJECT')
        links.new(proj_bake_tex_node.outputs['Color'],principled_bsdf.inputs['Base Color']);final_bake_tex_node=nodes.new(type='ShaderNodeTexImage');final_bake_tex_node.image=final_bake_image;nodes.active=final_bake_tex_node
        bpy.ops.object.bake(type='DIFFUSE',pass_filter={{'COLOR'}},target='IMAGE_TEXTURES',save_mode='INTERNAL')
    else:
        print("Baking to existing UV map...")
        if not obj.data.uv_layers: raise Exception("Mesh has no UV maps to bake to.")
        original_uv_map = obj.data.uv_layers[0]
        original_uv_map.active_render = True
        source_uv_node = nodes.new(type='ShaderNodeUVMap'); source_uv_node.uv_map = "ProjectionUV"
        links.new(source_uv_node.outputs['UV'], proj_bake_tex_node.inputs['Vector'])
        links.new(proj_bake_tex_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
        final_bake_tex_node = nodes.new(type='ShaderNodeTexImage'); final_bake_tex_node.image = final_bake_image; nodes.active = final_bake_tex_node
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={{'COLOR'}}, target='IMAGE_TEXTURES', save_mode='INTERNAL')

    final_bake_image.filepath_raw=params['output_texture'];final_bake_image.file_format='PNG';final_bake_image.save()
    print("Applying final 180-degree Z-axis rotation...")
    obj.rotation_euler.z += math.radians(180)
    print("Exporting mesh with new UVs and corrected orientation...")
    bpy.context.view_layer.objects.active=obj;obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=params['output_mesh'],forward_axis='NEGATIVE_Z',up_axis='Y',export_uv=True,export_normals=True,export_materials=False,path_mode='COPY')
    sys.exit(0)
except Exception as e:print(f"Blender script failed: {{e}}",file=sys.stderr);sys.exit(1)
"""
            with open(script_path, 'w') as f: f.write(script)

            try:
                subprocess.run([blender_path, '--factory-startup', '--background', '--python', script_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Blender execution failed. Stderr: {e.stderr}")

            baked_texture_pil = Image.open(output_texture_path).convert("RGB")
            baked_texture_np = np.array(baked_texture_pil).astype(np.float32) / 255.0
            baked_texture_tensor = torch.from_numpy(baked_texture_np)[None,]
            
            unwrapped_mesh = trimesh_loader.load(output_mesh_path, process=False)

            unwrapped_mesh.visual = trimesh_loader.visual.TextureVisuals(
                uv=unwrapped_mesh.visual.uv, 
                material=trimesh_loader.visual.material.PBRMaterial(baseColorTexture=baked_texture_pil)
            )

            return (unwrapped_mesh, baked_texture_tensor)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "BlenderRenderDepthMap": BlenderRenderDepthMap,
    "BlenderTextureProjection": BlenderTextureProjection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderRenderDepthMap": "Render Depth Map (Blender)",
    "BlenderTextureProjection": "Texture Projection Bake (Blender)",
}