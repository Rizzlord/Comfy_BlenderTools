from .blender_nodes import BlenderDecimate, BlenderUnwrap, MinistryOfFlatUnwrap, BlenderExportGLB
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial, SaveMultiviewImages, LoadMultiviewImages
from .utils import Voxelize, VoxelSettings, OtherModesSettings, TextureToHeight, DisplaceMesh, SmoothMesh, ProcessMesh, MirrorMesh, QuadriflowRemesh, QuadriflowSettings, SubdivisionMesh, SmoothByAngle, Pyremesh, O3DRemesh, InstantMeshes
from .vertexbake_nodes import VertexToHighPoly, VertexColorBake


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
    "VertexColorBake": VertexColorBake,
    "ProcessMesh": ProcessMesh,
    "MirrorMesh": MirrorMesh,
    "SubdivisionMesh": SubdivisionMesh,
    "SmoothByAngle": SmoothByAngle,
    "Pyremesh": Pyremesh,
    "O3DRemesh": O3DRemesh,
    "InstantMeshes": InstantMeshes,
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
    "VertexColorBake": "Bake Vertex Color",
    "ProcessMesh": "Process Mesh",
    "MirrorMesh": "Mirror Mesh",
    "SubdivisionMesh": "Subdivision Surface",
    "SmoothByAngle": "Smooth By Angle",
    "Pyremesh": "Pyremesher",
    "O3DRemesh": "Open3D Remesh",
    "InstantMeshes": "Instant Meshes (Quad Remesher)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']