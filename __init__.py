from .blender_nodes import BlenderDecimate, BlenderUnwrap, MinistryOfFlatUnwrap, BlenderExportGLB
from .baking_nodes import TextureBake, ApplyMaterial, ExtractMaterial, SaveMultiviewImages, LoadMultiviewImages
from .utils import Voxelize, VoxelSettings, OtherModesSettings, TextureToHeight, DisplaceMesh, SmoothMesh, ProcessMesh, MirrorMesh, QuadriflowRemesh, QuadriflowSettings, SubdivisionMesh, Pyremesh, O3DRemesh, InstantMeshes, UnwrapColoredMesh, BlenderPreview
from .vertexbake_nodes import VertexToHighPoly, MultiviewDisplaceMesh, VertexColorBake


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
    "MultiviewDisplaceMesh": MultiviewDisplaceMesh,
    "VertexColorBake": VertexColorBake,
    "ProcessMesh": ProcessMesh,
    "MirrorMesh": MirrorMesh,
    "SubdivisionMesh": SubdivisionMesh,
    "Pyremesh": Pyremesh,
    "O3DRemesh": O3DRemesh,
    "InstantMeshes": InstantMeshes,
    "UnwrapColoredMesh": UnwrapColoredMesh,
    "BlenderPreview": BlenderPreview,
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
    "MultiviewDisplaceMesh": "Displace Mesh (Multiview)",
    "VertexColorBake": "Bake Vertex Color",
    "ProcessMesh": "Process Mesh",
    "MirrorMesh": "Mirror Mesh",
    "SubdivisionMesh": "Subdivision Surface",
    "Pyremesh": "Pyremesher",
    "O3DRemesh": "Open3D Remesh",
    "InstantMeshes": "Instant Meshes (Quad Remesher)",
    "UnwrapColoredMesh": "Unwrap Colored Mesh (UVgami)",
    "BlenderPreview": "Blender Preview",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
