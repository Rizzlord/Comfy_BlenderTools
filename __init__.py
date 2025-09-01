from .blender_nodes import BlenderDecimate, BlenderUnwrap, MinistryOfFlatUnwrap, BlenderExportGLB
from .texture_nodes import BlenderTextureProjection, BlenderRenderDepthMap
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial, SaveMultiviewImages, LoadMultiviewImages
from .utils import Voxelize, VoxelSettings, OtherModesSettings, TextureToHeight, DisplaceMesh, SmoothMesh
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
    "BlenderTextureProjection": BlenderTextureProjection,
    "BlenderRenderDepthMap": BlenderRenderDepthMap,
    "Voxelize": Voxelize,
    "VoxelSettings": VoxelSettings,
    "OtherModesSettings": OtherModesSettings,
    "TextureToHeight": TextureToHeight,
    "DisplaceMesh": DisplaceMesh,
    "SmoothMesh": SmoothMesh,
    "VertexToHighPoly": VertexToHighPoly,
    "VertexToLowPoly": VertexToLowPoly,
    "VertexColorBake": VertexColorBake,
    "DiffuseHighpolyCol": DiffuseHighpolyCol
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
    "BlenderTextureProjection": "Texture Projection Bake (Blender)",
    "BlenderRenderDepthMap": "Render Depth Map (Blender)",
    "Voxelize": "Blender Remesh",
    "VoxelSettings": "Remesh Voxel Settings",
    "OtherModesSettings": "Remesh Other Settings",
    "TextureToHeight": "Texture To Height Map",
    "DisplaceMesh": "Displace Mesh",
    "SmoothMesh": "Smooth Mesh",
    "VertexToHighPoly": "Project Vertex Color (High Poly)",
    "VertexToLowPoly": "Bake Vertex to Texture (Low Poly)",
    "VertexColorBake": "Bake Vertex Color",
    "DiffuseHighpolyCol": "Bake Diffuse Color",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
