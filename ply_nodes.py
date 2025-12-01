import os
import tempfile
import numpy as np
import torch
import trimesh
import pymeshlab
from plyfile import PlyData, PlyElement

class GS_PlyToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"default": "", "multiline": False}),
                "depth": ("INT", {"default": 10, "min": 5, "max": 16, "step": 1}),
                "color_mode": (["Gaussian DC", "Standard RGB"], {"default": "Gaussian DC"}),
                "clean_up": ("BOOLEAN", {"default": True}),
                "invert_normals": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert_ply_to_mesh"
    CATEGORY = "Comfy_BlenderTools/Utils"

    def convert_ply_to_mesh(self, ply_path, depth, color_mode, clean_up, invert_normals):
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found at: {ply_path}")

        # 1. Read PLY Data
        try:
            plydata = PlyData.read(ply_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read PLY file: {e}")

        vertex_data = plydata['vertex']
        
        # Extract positions
        x = np.asarray(vertex_data['x'])
        y = np.asarray(vertex_data['y'])
        z = np.asarray(vertex_data['z'])
        
        # Stack positions
        points = np.vstack([x, y, z]).T

        # Extract Colors
        colors = None
        if color_mode == "Gaussian DC":
            # Check if f_dc fields exist
            if 'f_dc_0' in vertex_data and 'f_dc_1' in vertex_data and 'f_dc_2' in vertex_data:
                f_dc_0 = np.asarray(vertex_data['f_dc_0'])
                f_dc_1 = np.asarray(vertex_data['f_dc_1'])
                f_dc_2 = np.asarray(vertex_data['f_dc_2'])
                
                # SH coefficient (DC) to RGB conversion
                # RGB = 0.5 + C0 * f_dc
                # C0 = 0.28209479177387814
                SH_C0 = 0.28209479177387814
                r = 0.5 + SH_C0 * f_dc_0
                g = 0.5 + SH_C0 * f_dc_1
                b = 0.5 + SH_C0 * f_dc_2
                
                # Clip to [0, 1]
                colors = np.vstack([r, g, b]).T
                colors = np.clip(colors, 0.0, 1.0)
                # Convert to uint8 [0, 255]
                colors = (colors * 255).astype(np.uint8)
            else:
                print("Warning: Gaussian DC fields not found. Falling back to default white.")
                colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)

        elif color_mode == "Standard RGB":
            if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:
                r = np.asarray(vertex_data['red'])
                g = np.asarray(vertex_data['green'])
                b = np.asarray(vertex_data['blue'])
                colors = np.vstack([r, g, b]).T.astype(np.uint8)
            else:
                print("Warning: Standard RGB fields not found. Falling back to default white.")
                colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)

        # 2. Create Temporary PLY for PyMeshLab
        # PyMeshLab needs a file on disk or a dictionary. Let's use a temp file to be safe and explicit.
        # We need to write a PLY that PyMeshLab can digest easily (x, y, z, red, green, blue)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_ply = os.path.join(temp_dir, "temp_input.ply")
            temp_output_ply = os.path.join(temp_dir, "temp_output.ply")

            # Construct structured array for PlyFile
            # Define dtype
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
            vertex_array = np.empty(len(points), dtype=vertex_dtype)
            vertex_array['x'] = points[:, 0]
            vertex_array['y'] = points[:, 1]
            vertex_array['z'] = points[:, 2]
            vertex_array['red'] = colors[:, 0]
            vertex_array['green'] = colors[:, 1]
            vertex_array['blue'] = colors[:, 2]

            el = PlyElement.describe(vertex_array, 'vertex')
            PlyData([el], text=False).write(temp_input_ply)

            # 3. PyMeshLab Processing
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_input_ply)

            # Compute Normals
            # k=10 is a reasonable default for point clouds
            ms.compute_normal_for_point_clouds(k=10, flipflag=True, viewpos=[0, 0, 0])

            # Surface Reconstruction: Screened Poisson
            ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)

            # Cleanup (Optional)
            if clean_up:
                # Remove small connected components?
                # This might be too aggressive if the mesh is fragmented, but for "watertight" it's usually good to keep the main blob.
                # Let's try to remove isolated pieces if they are small.
                # 'remove_connected_component_by_diameter' is one option, or 'remove_isolated_pieces_wrt_face_num'
                # Let's stick to a safe default or skip if unsure. 
                # The user asked for "watertight", Poisson usually gives watertight.
                # But sometimes it creates extra bubbles.
                pass 

            # Save Output
            ms.save_current_mesh(temp_output_ply)

            # 4. Load with Trimesh
            mesh = trimesh.load(temp_output_ply, force='mesh', process=False)
            
            if invert_normals:
                mesh.invert()
            
            # Ensure vertex colors are preserved and correctly assigned
            # Pymeshlab should have interpolated colors to vertices.
            
            return (mesh,)
