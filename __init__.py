from .blender_nodes import (
    BlenderDecimate,
    BlenderUnwrap,
    MinistryOfFlatUnwrap,
    BlenderExportGLB,
    BlenderLoadModel,
    BlenderPreview3D,
)
from server import PromptServer
from aiohttp import web
import os
import folder_paths
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial
from .material_nodes import UpscaleMaterial
from .utils import (
    Voxelize,
    VoxelSettings,
    OtherModesSettings,
    TextureToHeight,
    DisplaceMesh,
    SmoothMesh,
    ProcessMesh,
    MirrorMesh,
    QuadriflowRemesh,
    QuadriflowSettings,
    SubdivisionMesh,
    Pyremesh,
    O3DRemesh,
    InstantMeshes,
    UnwrapColoredMesh,
    BlenderPreview,
    BlenderUVPack,
    BlenderPointsToMesh,
)
from .vertexbake_nodes import (
    VertexToHighPoly,
    MultiviewDisplaceMesh,
    VertexColorBake,
    MultiviewTextureBake,
    AutoBakeTextureFromMV,
    SeqTexCam,
    BakeToModel,
)
from .ply_nodes import GS_PlyToMesh


NODE_CLASS_MAPPINGS = {
    "BlenderDecimate": BlenderDecimate,
    "BlenderUnwrap": BlenderUnwrap,
    "MinistryOfFlatUnwrap": MinistryOfFlatUnwrap,
    "TextureBake": TextureBake,
    "ApplyMaterial": ApplyMaterial,
    "ExtractMaterial": ExtractMaterial,
    "UpscaleMaterial": UpscaleMaterial,
    "BlenderExportGLB": BlenderExportGLB,
    "BlenderLoadModel": BlenderLoadModel,
    "BlenderPreview3D": BlenderPreview3D,
    "Voxelize": Voxelize,
    "VoxelSettings": VoxelSettings,
    "OtherModesSettings": OtherModesSettings,
    "QuadriflowRemesh": QuadriflowRemesh,
    "QuadriflowSettings": QuadriflowSettings,
    "TextureToHeight": TextureToHeight,
    "DisplaceMesh": DisplaceMesh,
    "SmoothMesh": SmoothMesh,
    "VertexToHighPoly": VertexToHighPoly,
    "MultiviewDisplaceMesh": MultiviewDisplaceMesh,
    "MultiviewTextureBake": MultiviewTextureBake,
    "AutoBakeTextureFromMV": AutoBakeTextureFromMV,
    "SeqTexCam": SeqTexCam,
    "BakeToModel": BakeToModel,
    "VertexColorBake": VertexColorBake,
    "ProcessMesh": ProcessMesh,
    "MirrorMesh": MirrorMesh,
    "SubdivisionMesh": SubdivisionMesh,
    "Pyremesh": Pyremesh,
    "O3DRemesh": O3DRemesh,
    "InstantMeshes": InstantMeshes,
    "UnwrapColoredMesh": UnwrapColoredMesh,
    "BlenderPreview": BlenderPreview,
    "GS_PlyToMesh": GS_PlyToMesh,
    "BlenderUVPack": BlenderUVPack,
    "BlenderPointsToMesh": BlenderPointsToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "MinistryOfFlatUnwrap": "Ministry of Flat Unwrap",
    "TextureBake": "Blender Bake Maps",
    "ApplyMaterial": "Apply Material",
    "ExtractMaterial": "Extract Material",
    "UpscaleMaterial": "Upscale Material",
    "BlenderExportGLB": "Blender Export GLB",
    "BlenderLoadModel": "Blender Load Model",
    "BlenderPreview3D": "Blender Preview 3D",
    "Voxelize": "Blender Remesh",
    "VoxelSettings": "Remesh Voxel Settings",
    "OtherModesSettings": "Remesh Other Settings",
    "QuadriflowRemesh": "Blender Quadriflow Remesh",
    "QuadriflowSettings": "Remesh Quadriflow Settings",
    "TextureToHeight": "Texture To Height Map",
    "DisplaceMesh": "Displace Mesh",
    "SmoothMesh": "Smooth Mesh",
    "VertexToHighPoly": "Project Vertex Color (High Poly)",
    "MultiviewDisplaceMesh": "Displace Mesh (Multiview)",
    "MultiviewTextureBake": "Bake Multiview Texture",
    "AutoBakeTextureFromMV": "Auto Bake Texture (Multiview)",
    "SeqTexCam": "SeqTex Cam",
    "BakeToModel": "Bake To Model",
    "VertexColorBake": "Bake Vertex Color",
    "ProcessMesh": "Process Mesh",
    "MirrorMesh": "Mirror Mesh",
    "SubdivisionMesh": "Subdivision Surface",
    "Pyremesh": "Pyremesher",
    "O3DRemesh": "Open3D Remesh",
    "InstantMeshes": "Instant Meshes (Quad Remesher)",
    "UnwrapColoredMesh": "Unwrap Colored Mesh (UVgami)",
    "BlenderPreview": "Blender Preview",
    "GS_PlyToMesh": "GS Ply To Mesh",
    "BlenderUVPack": "Blender UV Pack",
    "BlenderPointsToMesh": "Blender Points To Mesh",
}


WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    routes = PromptServer.instance.routes
except AttributeError:
    routes = None

if routes is not None:
    @routes.get("/blender_tools/list_models")
    async def list_models(request):
        if "path" not in request.rel_url.query:
            return web.json_response({"error": "path not provided"}, status=400)
        
        path = request.rel_url.query["path"]
        
        if not os.path.exists(path) or not os.path.isdir(path):
             return web.json_response({"files": []})
             
        files = [f for f in os.listdir(path) if f.lower().endswith(('.glb', '.obj', '.fbx', '.ply', '.usdz', '.udz'))]
        return web.json_response({"files": sorted(files)})

    @routes.get("/blender_tools/view_model")
    async def view_model(request):
        if "path" not in request.rel_url.query:
            return web.json_response({"error": "path not provided"}, status=400)
            
        path = request.rel_url.query["path"]
        
        print(f"[BlenderTools] Requesting view_model for path: {path}")
        if not os.path.exists(path):
            fixed_path = path.lstrip("/\\")
            if len(fixed_path) > 2 and fixed_path[1] == ":" and os.path.exists(fixed_path):
                path = fixed_path
            else:
                for base in [
                    folder_paths.get_temp_directory(),
                    folder_paths.get_output_directory(),
                    folder_paths.get_input_directory(),
                ]:
                    candidate = os.path.join(base, fixed_path)
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        path = candidate
                        break

        if not os.path.exists(path):
            print(f"[BlenderTools] Error: Path does NOT exist: {path}")
            return web.json_response({"error": "File not found"}, status=404)
            
        if not os.path.isfile(path):
            print(f"[BlenderTools] Error: Path is NOT a file: {path}")
            return web.json_response({"error": "Path is not a file"}, status=404)
            
        return web.FileResponse(path)
