from .blender_nodes import BlenderDecimate, BlenderUnwrap, MinistryOfFlatUnwrap, BlenderExportGLB
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial, SaveMultiviewImages, LoadMultiviewImages
from .utils import Voxelize, VoxelSettings, OtherModesSettings, TextureToHeight, DisplaceMesh, SmoothMesh, ProcessMesh, QuadriflowRemesh, QuadriflowSettings
from .vertexbake_nodes import VertexToHighPoly, VertexToLowPoly
from .vtxbake_nodes import VertexColorBake, DiffuseHighpolyCol


NODE_CLASS_MAPPINGS = {
    "BlenderDecimate": BlenderDecimate,
    "BlenderUnwrap": BlenderUnwrap,
    "MinistryOfFlatUnwrap": MinistryOfFlatUnwrap,
    "TextureBake": TextureBake,
    "SaveMultiviewImages": SaveMultiviewImages,
    "LoadMultiviewImages": LoadMultiviewImages,
    "ApplyMaterial": ApplyMaterial,
    "ExtractMaterial": ExtractMaterial,
    "BlenderExportGLB": BlenderExportGLB,
    "Voxelize": Voxelize,
    "VoxelSettings": VoxelSettings,
    "OtherModesSettings": OtherModesSettings,
    "QuadriflowRemesh": QuadriflowRemesh,
    "QuadriflowSettings": QuadriflowSettings,
    "TextureToHeight": TextureToHeight,
    "DisplaceMesh": DisplaceMesh,
    "SmoothMesh": SmoothMesh,
    "VertexToHighPoly": VertexToHighPoly,
    "VertexToLowPoly": VertexToLowPoly,
    "VertexColorBake": VertexColorBake,
    "DiffuseHighpolyCol": DiffuseHighpolyCol,
    "ProcessMesh": ProcessMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "MinistryOfFlatUnwrap": "Ministry of Flat Unwrap",
    "TextureBake": "Blender Bake Maps",
    "SaveMultiviewImages": "Save Multiview Images",
    "LoadMultiviewImages": "Load Multiview Images",
    "ApplyMaterial": "Apply Material",
    "ExtractMaterial": "Extract Material",
    "BlenderExportGLB": "Blender Export GLB",
    "Voxelize": "Blender Remesh",
    "VoxelSettings": "Remesh Voxel Settings",
    "OtherModesSettings": "Remesh Other Settings",
    "QuadriflowRemesh": "Blender Quadriflow Remesh",
    "QuadriflowSettings": "Remesh Quadriflow Settings",
    "TextureToHeight": "Texture To Height Map",
    "DisplaceMesh": "Displace Mesh",
    "SmoothMesh": "Smooth Mesh",
    "VertexToHighPoly": "Project Vertex Color (High Poly)",
    "VertexToLowPoly": "Bake Vertex to Texture (Low Poly)",
    "VertexColorBake": "Bake Vertex Color",
    "DiffuseHighpolyCol": "Bake Diffuse Color",
    "ProcessMesh": "Process Mesh"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']