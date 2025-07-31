ComfyUI Blender Tools
Welcome to ComfyUI Blender Tools, a collection of custom nodes designed to integrate Blender's powerful 3D processing capabilities directly into your ComfyUI workflows. These nodes allow you to perform mesh decimation, UV unwrapping, texture projection, depth map rendering, and mesh cleanup without leaving ComfyUI.

🚀 Installation
Ensure Blender is Installed: These nodes require a working Blender installation (version 4.0 or newer is recommended).

Set BLENDER_EXE Environment Variable:
You must set an environment variable named BLENDER_EXE to the full path of your Blender executable (e.g., C:\Program Files\Blender Foundation\Blender 4.5\blender.exe on Windows or /Applications/Blender.app/Contents/MacOS/Blender on macOS).

Windows:

Search for "Environment Variables" in the Start Menu.

Click "Edit the system environment variables".

Click "Environment Variables..." button.

Under "System variables", click "New...".

Set "Variable name" to BLENDER_EXE and "Variable value" to your Blender executable path.

Click OK on all windows.

Linux/macOS (for current session):

export BLENDER_EXE="/path/to/your/blender/executable"

For permanent setting, add this line to your ~/.bashrc, ~/.zshrc, or ~/.profile file.

Install xatlas (Optional, for advanced unwrapping):
If you plan to use the "XAtlas UV Atlas" unwrap method, you need to install xatlas:

pip install xatlas

Navigate to your custom_nodes directory:

cd ComfyUI/custom_nodes

Clone this repository:

git clone https://github.com/YourGitHubUsername/ComfyUI-BlenderTools.git

(Replace YourGitHubUsername with your actual GitHub username if you've forked this repo, or the original author's if you're cloning directly).

Restart ComfyUI:
Your new nodes should now appear in the node selection menu under the "Comfy_BlenderTools" category.

✨ Nodes Overview
This section provides a detailed look at each node included in the ComfyUI Blender Tools collection.

Comfy_BlenderTools/Mesh Processing
Blender Decimate
Reduces the face count of a 3D mesh using Blender's decimate modifier. It can also triangulate the mesh and merge close vertices.

Category: Mesh Processing

Function: decimate

Inputs:

trimesh (TRIMESH): The input mesh to be decimated.

apply_decimation (BOOLEAN): Whether to apply the decimation modifier. Default: True

max_face_count (INT): The target maximum number of faces after decimation. Default: 10000 (Min: 100, Max: 10,000,000)

triangulate (BOOLEAN): Whether to triangulate the mesh during decimation. Default: True

merge_distance (FLOAT): Distance threshold for merging vertices (to remove doubles). Default: 0.0001 (Min: 0.0, Max: 1.0)

Outputs:

TRIMESH (TRIMESH): The decimated and cleaned mesh.

Mesh Cleanup
Safely isolates the largest connected component of a mesh and optionally applies Blender's Smooth modifier.

Category: Mesh Processing

Function: cleanup_and_smooth_mesh

Inputs:

trimesh (TRIMESH): The input mesh for cleanup.

apply_smoothing (BOOLEAN): Whether to apply the smooth modifier. Default: False

factor (FLOAT): The smoothing factor. Default: 0.5 (Min: -10.0, Max: 10.0)

repeat (INT): Number of smoothing iterations. Default: 1 (Min: 1, Max: 100)

axis_x (BOOLEAN): Apply smoothing along the X-axis. Default: True

axis_y (BOOLEAN): Apply smoothing along the Y-axis. Default: True

axis_z (BOOLEAN): Apply smoothing along the Z-axis. Default: True

Outputs:

trimesh (TRIMESH): The cleaned and optionally smoothed mesh.

Comfy_BlenderTools/UV & Texture
Blender Unwrap
Performs UV unwrapping on a 3D mesh using various Blender methods or XAtlas. It can also export a preview of the UV layout.

Category: UV & Texture

Function: unwrap

Inputs:

trimesh (TRIMESH): The input mesh to be unwrapped.

unwrap_method (ENUM): The UV unwrapping algorithm to use. Options: XAtlas UV Atlas, Smart UV Project, Unwrap (Angle Based), Unwrap (Conformal), Cube Projection. Default: Smart UV Project

export_uv_layout (BOOLEAN): Whether to generate and export a preview image of the UV layout. Default: True

texture_resolution (ENUM): The resolution for the UV layout preview and baking operations. Options: 512, 768, 1024, 1536, 2048, 4096, 8192. Default: 1024

pixel_margin (INT): Margin in pixels between UV islands. Default: 0 (Min: 0, Max: 64)

angle_limit (FLOAT): Angle limit for Smart UV Project and Angle Based Unwrap. Default: 66.0 (Min: 0.0, Max: 90.0)

refine_with_minimum_stretch (BOOLEAN): Applies a minimum stretch refinement after unwrapping. Default: False

min_stretch_iterations (INT): Iterations for minimum stretch refinement. Default: 10 (Min: 0, Max: 256)

final_merge_distance (FLOAT): Distance threshold for a final UV-aware vertex merge. Default: 0.0 (Min: 0.0, Max: 1.0)

correct_aspect (BOOLEAN): Correct aspect ratio for UV projection. Default: True

Outputs:

trimesh (TRIMESH): The mesh with new UV coordinates.

uv_layout_preview (IMAGE): A torch tensor image representing the UV layout.

Texture Projection Bake (Blender)
Projects images onto a 3D mesh from multiple camera angles and bakes them into a single texture. Can optionally auto-unwrap the mesh first.

Category: UV & Texture

Function: project_and_bake

Inputs:

trimesh (TRIMESH): The input mesh to project textures onto.

images_to_project (IMAGE): A batch of images to be projected.

AutoUnwrap (BOOLEAN): If True, the mesh will be unwrapped using Smart UV Project before baking. If False, it bakes to existing UVs. Default: True

bake_resolution (ENUM): Resolution of the baked texture. Options: 1024, 2048, 4096, 8192. Default: 2048

projection_blend (FLOAT): Blending factor for multiple image projections. Default: 0.5 (Min: 0.0, Max: 1.0)

use_gpu (BOOLEAN): Use GPU for Blender rendering. Default: True

camera_distance (FLOAT): Distance of the projection cameras from the model. Default: 50.0 (Min: 1.0, Max: 200.0)

camera_rotation (FLOAT): Base rotation of the projection camera(s) around the Z-axis. Default: 0.0 (Min: -180.0, Max: 180.0)

camera_elevation (FLOAT): Elevation of the projection camera(s). Default: 0.0 (Min: -90.0, Max: 90.0)

camera_fov (FLOAT): Field of View for the projection cameras. Default: 60.0 (Min: 0.0, Max: 120.0)

model_rotation_x (FLOAT): Rotation of the model around its X-axis before projection. Default: 90.0 (Min: -360.0, Max: 360.0)

model_rotation_y (FLOAT): Rotation of the model around its Y-axis before projection. Default: 0.0 (Min: -360.0, Max: 360.0)

model_rotation_z (FLOAT): Rotation of the model around its Z-axis before projection. Default: 180.0 (Min: -360.0, Max: 360.0)

multi_view (BOOLEAN): Use multiple projection cameras. Default: False

camera_count (ENUM): Number of cameras for multi-view projection. Options: 2, 6, 10. Default: 2

Outputs:

trimesh (TRIMESH): The mesh with the new UVs (if AutoUnwrap was true) and corrected orientation.

baked_texture (IMAGE): The baked texture image.

Apply Texture and Export GLB
Applies a given texture to a mesh (assuming it has UVs) and exports the textured mesh as a GLB file.

Category: UV & Texture

Function: apply_and_export

Inputs:

trimesh (TRIMESH): The mesh to apply the texture to.

texture (IMAGE): The texture image to apply.

filename_prefix (STRING): Prefix for the exported GLB filename. Default: "TexturedMesh"

Outputs:

trimesh (TRIMESH): The textured mesh object.

glb_path (STRING): The path to the exported GLB file.

Comfy_BlenderTools/Rendering & Export
Render Depth Map (Blender)
Renders a depth map of a 3D mesh from a specified camera perspective using Blender's Cycles renderer. Supports single or multi-view rendering.

Category: Rendering & Export

Function: render_depth

Inputs:

trimesh (TRIMESH): The input mesh to render.

use_gpu (BOOLEAN): Use GPU for Blender rendering. Default: True

resolution (ENUM): Resolution of the rendered depth map. Options: 512, 1024, 2048. Default: 1024

camera_distance (FLOAT): Distance of the camera from the model. Default: 50.0 (Min: 1.0, Max: 200.0)

camera_rotation (FLOAT): Rotation of the camera around the Z-axis. Default: 0.0 (Min: -180.0, Max: 180.0)

camera_elevation (FLOAT): Elevation of the camera. Default: 0.0 (Min: -90.0, Max: 90.0)

camera_fov (FLOAT): Field of View for the camera. Default: 60.0 (Min: 0.0, Max: 120.0)

depth_range (FLOAT): The range of depth values to map to the output image. Default: 2.0 (Min: 0.1, Max: 10.0)

depth_strength (FLOAT): Controls the intensity/contrast of the depth map. Default: 1.0 (Min: 0.1, Max: 5.0)

model_rotation_x (FLOAT): Rotation of the model around its X-axis. Default: 90.0 (Min: -360.0, Max: 360.0)

model_rotation_y (FLOAT): Rotation of the model around its Y-axis. Default: 0.0 (Min: -360.0, Max: 360.0)

model_rotation_z (FLOAT): Rotation of the model around its Z-axis. Default: 180.0 (Min: -360.0, Max: 360.0)

multi_view (BOOLEAN): Render depth maps from multiple camera angles. Default: False

camera_count (ENUM): Number of cameras for multi-view rendering. Options: 2, 6, 10. Default: 2

Outputs:

depth_map (IMAGE): The primary depth map (from the first camera if multi-view).

multi_view_images (IMAGE): A batch of depth maps if multi_view is enabled.

Export for Blender (FBX + Textures)
Exports a mesh as an FBX file and extracts any associated textures into a dedicated folder.

Category: Rendering & Export

Function: export_model_and_textures_separately

Inputs:

trimesh (TRIMESH): The mesh to export.

directory (STRING): The base directory for the exported model folder. Default: "output/exported_models"

original_filename (STRING): The original filename of the model, used to name the output folder and FBX. Default: "model.glb"

Outputs:

MODEL_FOLDER_PATH (STRING): The path to the folder containing the exported FBX and textures.

Feel free to contribute or suggest improvements!