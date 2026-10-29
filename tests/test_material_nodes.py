import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh


# Exercise material handling without initializing ComfyUI's GPU runtime.
backend = types.ModuleType("comfy_extras.nodes_upscale_model")
backend.ImageUpscaleWithModel = unittest.mock.Mock()
spec = importlib.util.spec_from_file_location(
    "material_nodes", Path(__file__).resolve().parents[1] / "material_nodes.py"
)
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {backend.__name__: backend}):
    spec.loader.exec_module(module)


class UpscaleMaterialTests(unittest.TestCase):
    def setUp(self):
        self.node = module.UpscaleMaterial()
        self.orm = Image.new("RGB", (8, 8), (40, 90, 150))
        self.material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGBA", (8, 8), (100, 120, 140, 60)),
            normalTexture=Image.new("RGB", (8, 8), (128, 128, 255)),
            metallicRoughnessTexture=self.orm,
            occlusionTexture=self.orm,
            roughnessFactor=0.3,
            alphaMode="BLEND",
        )
        self.mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            visual=trimesh.visual.TextureVisuals(
                uv=[[0, 0], [1, 0], [0, 1]], material=self.material,
            ),
            process=False,
        )

    def test_none_skips_model(self):
        with patch.object(module, "ImageUpscaleWithModel") as runner:
            result, = self.node.upscale(self.mesh, "none", "none", "none", object())
        self.assertIsNone(result.visual.material.occlusionTexture)
        np.testing.assert_array_equal(result.visual.material.baseColorTexture, self.material.baseColorTexture)
        self.assertIs(self.material.occlusionTexture, self.orm)
        runner.assert_not_called()

    def test_resize_preserves_assignments_and_original(self):
        result, = self.node.upscale(self.mesh, "512", "none", "1024")
        material = result.visual.material
        self.assertEqual(material.baseColorTexture.size, (512, 512))
        self.assertEqual(material.baseColorTexture.getpixel((0, 0))[3], 60)
        self.assertEqual(material.metallicRoughnessTexture.size, (1024, 1024))
        self.assertIsNone(material.occlusionTexture)
        np.testing.assert_array_equal(material.normalTexture, self.material.normalTexture)
        np.testing.assert_array_equal(result.visual.uv, self.mesh.visual.uv)
        np.testing.assert_array_equal(result.faces, self.mesh.faces)
        self.assertEqual(material.roughnessFactor, 0.3)
        self.assertEqual(material.alphaMode, "BLEND")
        self.assertEqual(self.material.baseColorTexture.size, (8, 8))
        self.assertEqual(self.orm.size, (8, 8))

    def test_lanczos_precedes_model_and_alpha_survives(self):
        pixels = np.random.default_rng(7).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        source = Image.fromarray(pixels).convert("RGBA")
        source.putalpha(60)
        self.material.baseColorTexture = source
        model = object()

        def upscale(received_model, tensor):
            self.assertIs(received_model, model)
            expected = source.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
            np.testing.assert_allclose(tensor[0].numpy(), np.asarray(expected) / 255.0, atol=1e-7)
            return (tensor.repeat_interleave(2, 1).repeat_interleave(2, 2),)

        with patch.object(module, "ImageUpscaleWithModel") as runner:
            runner.return_value.upscale.side_effect = upscale
            result, = self.node.upscale(self.mesh, "512", "none", "none", model)
        self.assertEqual(result.visual.material.baseColorTexture.size, (1024, 1024))
        self.assertEqual(result.visual.material.baseColorTexture.getpixel((0, 0))[3], 60)

    def test_multimaterial_scene_and_simple_material(self):
        simple = trimesh.visual.material.SimpleMaterial(image=Image.new("RGB", (8, 8)))
        self.mesh.visual.material = trimesh.visual.material.MultiMaterial([self.material, simple])
        self.mesh.visual.face_materials = np.array([1])
        scene = trimesh.Scene(self.mesh)
        result, = self.node.upscale(scene, "512", "2048", "none")
        geometry = next(iter(result.geometry.values()))
        first, second = geometry.visual.material.materials
        self.assertEqual(first.normalTexture.size, (2048, 2048))
        self.assertIsNone(first.occlusionTexture)
        self.assertEqual(second.image.size, (512, 512))
        np.testing.assert_array_equal(geometry.visual.face_materials, [1])
        self.assertEqual(simple.image.size, (8, 8))

    def test_missing_textures_are_skipped(self):
        self.mesh.visual.material = trimesh.visual.material.PBRMaterial()
        with patch.object(module, "ImageUpscaleWithModel") as runner:
            result, = self.node.upscale(self.mesh, "512", "512", "512", object())
        self.assertIsNone(result.visual.material.baseColorTexture)
        runner.assert_not_called()

    def test_output_resolutions_follow_model(self):
        def upscale(model, tensor):
            self.assertEqual(tuple(tensor.shape), (1, 512, 512, 3))
            return (tensor.repeat_interleave(2, 1).repeat_interleave(2, 2),)

        with patch.object(module, "ImageUpscaleWithModel") as runner:
            runner.return_value.upscale.side_effect = upscale
            result, = self.node.upscale(
                self.mesh, "512", "512", "512", object(),
                albedo_output_res="512", normal_output_res="2048", orm_output_res="1024",
            )
            self.assertEqual(runner.return_value.upscale.call_count, 3)
        material = result.visual.material
        self.assertEqual(material.baseColorTexture.size, (512, 512))
        self.assertEqual(material.baseColorTexture.getpixel((0, 0))[3], 60)
        self.assertEqual(material.normalTexture.size, (2048, 2048))
        self.assertEqual(material.metallicRoughnessTexture.size, (1024, 1024))
        self.assertIsNone(material.occlusionTexture)

    def test_output_resize_uses_lanczos_without_model(self):
        pixels = np.random.default_rng(9).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        self.material.normalTexture = Image.fromarray(pixels)
        result, = self.node.upscale(
            self.mesh, "none", "1024", "none", normal_output_res="512",
        )
        expected = self.material.normalTexture.resize((1024, 1024), Image.Resampling.LANCZOS)
        expected = expected.resize((512, 512), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(result.visual.material.normalTexture, expected)

    def test_initial_none_still_skips_texture(self):
        result, = self.node.upscale(
            self.mesh, "none", "none", "none", object(), albedo_output_res="512",
        )
        self.assertEqual(result.visual.material.baseColorTexture.size, (8, 8))
        self.assertIsNone(result.visual.material.occlusionTexture)

    def test_shared_texture_with_different_output_sizes(self):
        self.material.normalTexture = self.orm
        result, = self.node.upscale(
            self.mesh, "none", "512", "512",
            normal_output_res="1024", orm_output_res="2048",
        )
        material = result.visual.material
        self.assertEqual(material.normalTexture.size, (1024, 1024))
        self.assertEqual(material.metallicRoughnessTexture.size, (2048, 2048))
        self.assertIsNone(material.occlusionTexture)

    def test_ao_removed_from_every_scene_mesh_without_resizing(self):
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh, geom_name="first")
        scene.add_geometry(self.mesh.copy(), geom_name="second")
        with patch.object(module, "ImageUpscaleWithModel") as runner:
            result, = self.node.upscale(scene, "none", "none", "none", object())
        runner.assert_not_called()
        for name, geometry in result.geometry.items():
            self.assertIsNone(geometry.visual.material.occlusionTexture)
            original = scene.geometry[name].visual.material
            self.assertIsNotNone(original.occlusionTexture)
            np.testing.assert_array_equal(
                geometry.visual.material.metallicRoughnessTexture,
                original.metallicRoughnessTexture,
            )


if __name__ == "__main__":
    unittest.main()
