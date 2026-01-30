from .blender_nodes import (
    BlenderDecimate,
    BlenderUnwrap,
    MinistryOfFlatUnwrap,
    BlenderExportGLB,
    BlenderExportGLB,
    BlenderLoadModel,
)
from server import PromptServer
from aiohttp import web
import os
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial
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
    "BlenderExportGLB": BlenderExportGLB,
    "BlenderLoadModel": BlenderLoadModel,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "MinistryOfFlatUnwrap": "Ministry of Flat Unwrap",
    "TextureBake": "Blender Bake Maps",
    "ApplyMaterial": "Apply Material",
    "ExtractMaterial": "Extract Material",
    "BlenderExportGLB": "Blender Export GLB",
    "BlenderLoadModel": "Blender Load Model",
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
}


WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

@PromptServer.instance.routes.get("/blender_tools/list_models")
async def list_models(request):
    if "path" not in request.rel_url.query:
        return web.json_response({"error": "path not provided"}, status=400)
    
    path = request.rel_url.query["path"]
    
    if not os.path.exists(path) or not os.path.isdir(path):
         return web.json_response({"files": []})
         
    files = [f for f in os.listdir(path) if f.lower().endswith(('.glb', '.obj', '.fbx', '.ply'))]
    return web.json_response({"files": sorted(files)})

@PromptServer.instance.routes.get("/blender_tools/view_model")
async def view_model(request):
    if "path" not in request.rel_url.query:
        return web.json_response({"error": "path not provided"}, status=400)
        
    path = request.rel_url.query["path"]
    
    if not os.path.exists(path) or not os.path.isfile(path):
        return web.json_response({"error": "File not found"}, status=404)
        
    return web.FileResponse(path)
