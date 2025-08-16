from .blender_nodes import BlenderDecimate, BlenderUnwrap, MinistryOfFlatUnwrap, BlenderExportGLB
from .texture_nodes import BlenderTextureProjection, BlenderRenderDepthMap
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial
from .utils import Voxelize, VoxelSettings, OtherModesSettings, TextureToHeight, ImageDisplace, SmoothMesh

NODE_CLASS_MAPPINGS = {
    "BlenderDecimate": BlenderDecimate,
    "BlenderUnwrap": BlenderUnwrap,
    "MinistryOfFlatUnwrap": MinistryOfFlatUnwrap,
    "TextureBake": TextureBake,
    "ApplyMaterial": ApplyMaterial,
    "ExtractMaterial": ExtractMaterial,
    "BlenderExportGLB": BlenderExportGLB,
    "BlenderTextureProjection": BlenderTextureProjection,
    "BlenderRenderDepthMap": BlenderRenderDepthMap,
    "Voxelize": Voxelize,
    "VoxelSettings": VoxelSettings,
    "OtherModesSettings": OtherModesSettings,
    "TextureToHeight": TextureToHeight,
    "ImageDisplace": ImageDisplace,
    "SmoothMesh": SmoothMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "MinistryOfFlatUnwrap": "Ministry of Flat Unwrap",
    "TextureBake": "Blender Bake Maps",
    "ApplyMaterial": "Apply Material",
    "ExtractMaterial": "Extract Material",
    "BlenderExportGLB": "Blender Export GLB",
    "BlenderTextureProjection": "Texture Projection Bake (Blender)",
    "BlenderRenderDepthMap": "Render Depth Map (Blender)",
    "Voxelize": "Blender Remesh",
    "VoxelSettings": "Remesh Voxel Settings",
    "OtherModesSettings": "Remesh Other Settings",
    "TextureToHeight": "Texture To Height Map",
    "ImageDisplace": "Image Displace Mesh",
    "SmoothMesh": "Smooth Mesh",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
