from copy import deepcopy

import numpy as np
from PIL import Image
import torch
import trimesh
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel


class UpscaleMaterial:
    @classmethod
    def INPUT_TYPES(cls):
        resolutions = ["none", "512", "1024", "2048", "4096"]
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "albedo_res": (resolutions, {"default": "none"}),
                "normal_res": (resolutions, {"default": "none"}),
                "orm_res": (resolutions, {"default": "none"}),
                "albedo_output_res": (resolutions, {"default": "none"}),
                "normal_output_res": (resolutions, {"default": "none"}),
                "orm_output_res": (resolutions, {"default": "none"}),
            },
            "optional": {"upscale_model": ("UPSCALE_MODEL",)},
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "upscale"
    CATEGORY = "Comfy_BlenderTools"
    DESCRIPTION = (
        "Resize existing textures to the selected square resolution with Lanczos, "
        "then optionally upscale using a model and resize with Lanczos to output_res. "
        "Output resolution none keeps the upscaled size. "
        "ORM processes the metallic/roughness texture in its existing slot. "
        "Initial resolution none skips resizing and upscaling for that group. "
        "Ambient occlusion texture assignments are always removed."
    )

    def upscale(
        self, mesh, albedo_res, normal_res, orm_res, upscale_model=None,
        albedo_output_res="none", normal_output_res="none", orm_output_res="none",
    ):
        result = deepcopy(mesh)
        slots = {
            "baseColorTexture": (albedo_res, albedo_output_res),
            "image": (albedo_res, albedo_output_res),  # SimpleMaterial's diffuse texture.
            "normalTexture": (normal_res, normal_output_res),
            "metallicRoughnessTexture": (orm_res, orm_output_res),
        }
        processed = {}
        visited = set()

        def update_material(material):
            if material is None or id(material) in visited:
                return
            visited.add(id(material))
            if isinstance(material, trimesh.visual.material.MultiMaterial):
                for child in material.materials:
                    update_material(child)
                return
            if getattr(material, "occlusionTexture", None) is not None:
                material.occlusionTexture = None
            for slot, (resolution, output_resolution) in slots.items():
                texture = getattr(material, slot, None)
                if resolution == "none" or texture is None:
                    continue
                key = (id(texture), resolution, output_resolution)
                if key not in processed:
                    image = self._resize_texture(texture, int(resolution), upscale_model)
                    if output_resolution != "none":
                        size = int(output_resolution)
                        image = image.resize((size, size), Image.Resampling.LANCZOS)
                    processed[key] = (texture, image)
                setattr(material, slot, processed[key][1])

        geometries = (
            [(mesh.geometry[name], geometry) for name, geometry in result.geometry.items()]
            if isinstance(result, trimesh.Scene) else [(mesh, result)]
        )
        material_memo = {}
        for original, geometry in geometries:
            material = getattr(original.visual, "material", None)
            if material is not None:
                # Trimesh.copy copies each texture separately; retain shared slot assignments.
                geometry.visual.material = deepcopy(material, material_memo)
                update_material(geometry.visual.material)
        return (result,)

    @staticmethod
    def _resize_texture(texture, resolution, upscale_model):
        # Palette transparency must survive conversion to the model's RGB input.
        has_alpha = "A" in texture.getbands() or "transparency" in texture.info
        source = texture.convert("RGBA" if has_alpha else "RGB")
        resized = source.resize((resolution, resolution), Image.Resampling.LANCZOS)
        if upscale_model is None:
            return resized.convert(texture.mode) if texture.mode in ("L", "LA") else resized

        pixels = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        output = ImageUpscaleWithModel().upscale(upscale_model, tensor)[0]
        pixels = output[0].detach().float().cpu().numpy()
        image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
        if has_alpha:
            image.putalpha(resized.getchannel("A").resize(image.size, Image.Resampling.LANCZOS))
        return image.convert(texture.mode) if texture.mode in ("L", "LA") else image
