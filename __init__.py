from .blender_nodes import BlenderDecimate, BlenderUnwrap, TextureBake, ApplyTexturesToMesh, BlenderExportGLB
from .texture_nodes import BlenderTextureProjection, BlenderRenderDepthMap
from .utils import Voxelize, VoxelSettings, OtherModesSettings

NODE_CLASS_MAPPINGS = {
    "BlenderDecimate": BlenderDecimate,
    "BlenderUnwrap": BlenderUnwrap,
    "TextureBake": TextureBake,
    "ApplyTexturesToMesh": ApplyTexturesToMesh,
    "BlenderExportGLB": BlenderExportGLB,
    "BlenderTextureProjection": BlenderTextureProjection,
    "BlenderRenderDepthMap": BlenderRenderDepthMap,
    "Voxelize": Voxelize,
    "VoxelSettings": VoxelSettings,
    "OtherModesSettings": OtherModesSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "TextureBake": "Blender Bake Maps",
    "ApplyTexturesToMesh": "Apply Textures to Mesh",
    "BlenderExportGLB": "Blender Export GLB",
    "BlenderTextureProjection": "Texture Projection Bake (Blender)",
    "BlenderRenderDepthMap": "Render Depth Map (Blender)",
    "Voxelize": "Blender Remesh",
    "VoxelSettings": "Remesh Voxel Settings",
    "OtherModesSettings": "Remesh Other Settings",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']