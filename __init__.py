# Import the new nodes and remove the old one
from .blender_nodes import BlenderDecimate, BlenderUnwrap, ApplyTextureAndExport, BlendFBX_Export
from .texture_nodes import BlenderTextureProjection, BlenderRenderDepthMap

# Register the new nodes in the mappings
NODE_CLASS_MAPPINGS = {
    "BlenderDecimate": BlenderDecimate,
    "BlenderUnwrap": BlenderUnwrap,
    "ApplyTextureAndExport": ApplyTextureAndExport,
    "BlendFBX_Export": BlendFBX_Export,
    "BlenderTextureProjection": BlenderTextureProjection,
    "BlenderRenderDepthMap": BlenderRenderDepthMap,
}

# Give the new nodes display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderDecimate": "Blender Decimate",
    "BlenderUnwrap": "Blender Unwrap",
    "ApplyTextureAndExport": "Apply Texture and Export GLB",
    "BlendFBX_Export": "Export for Blender (FBX + Textures)",
    "BlenderTextureProjection": "Texture Projection Bake (Blender)",
    "BlenderRenderDepthMap": "Render Depth Map (Blender)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']