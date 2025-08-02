from .blender_nodes import BlenderDecimate, BlenderUnwrap, ApplyAlbedoAndExport, BlendFBX_Export
from .texture_nodes import BlenderTextureProjection, BlenderRenderDepthMap
from .utils import Voxelize, VoxelSettings, OtherModesSettings

NODE_CLASS_MAPPINGS = {
    "BlenderDecimate": BlenderDecimate,
    "BlenderUnwrap": BlenderUnwrap,
    "ApplyAlbedoAndExport": ApplyAlbedoAndExport,
    "BlendFBX_Export": BlendFBX_Export,
    "BlenderTextureProjection": BlenderTextureProjection,
    "BlenderRenderDepthMap": BlenderRenderDepthMap,
    "Voxelize": Voxelize,
    "VoxelSettings": VoxelSettings,
    "OtherModesSettings": OtherModesSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "ApplyAlbedoAndExport": "Apply Albedo and Export GLB",
    "BlendFBX_Export": "Export for Blender (FBX + Textures)",
    "BlenderTextureProjection": "Texture Projection Bake (Blender)",
    "BlenderRenderDepthMap": "Render Depth Map (Blender)",
    "Voxelize": "Blender Remesh",
    "VoxelSettings": "Remesh Voxel Settings",
    "OtherModesSettings": "Remesh Other Settings",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']