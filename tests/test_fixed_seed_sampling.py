import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import torch

from scripts.sample_fixed_seeds import (
    assert_repeat_equivalence,
    assert_work_group_equivalence,
    build_metadata,
    configure_precision,
    generate_uint8,
    make_checkpoint_id,
    save_mode_outputs,
    seeded_inputs,
    sha256_file,
    write_manifest,
)


class DummyNet(torch.nn.Module):
    img_channels = 3
    img_resolution = 32
    use_fp16 = False

    def round_sigma(self, sigma):
        return sigma

    def forward(self, x, sigma, class_labels=None):
        del sigma, class_labels
        # Deliberately depend on the forward batch composition. The fixed-seed
        # sampler must still be invariant to its outer work-group size.
        batch_term = x.mean(dim=0, keepdim=True) * 0.01
        return torch.tanh((x + batch_term) / 80)


class FixedSeedSamplingTest(unittest.TestCase):
    def setUp(self):
        self.net = DummyNet().eval()
        self.seeds = list(range(64))

    def test_precision_is_explicit_and_reported(self):
        self.assertEqual(configure_precision(self.net, "checkpoint", "cpu"), "fp32")
        self.assertEqual(configure_precision(self.net, "fp32", "cpu"), "fp32")
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            configure_precision(self.net, "fp16", "cpu")

    def test_checkpoint_id_uses_filename_and_sha_prefix(self):
        digest = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
        self.assertEqual(
            make_checkpoint_id("/tmp/edm-cifar10-32x32-uncond-vp.pkl", digest),
            "edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d",
        )

    def test_same_seed_shares_initial_latent_and_deterministic_noise(self):
        shape = (3, 32, 32)
        one_step_latent, _ = seeded_inputs([7], shape, 0)
        two_step_latent, first_noise = seeded_inputs([7], shape, 1)
        repeated_latent, repeated_noise = seeded_inputs([7], shape, 1)
        torch.testing.assert_close(one_step_latent, two_step_latent, rtol=0, atol=0)
        torch.testing.assert_close(two_step_latent, repeated_latent, rtol=0, atol=0)
        torch.testing.assert_close(first_noise[0], repeated_noise[0], rtol=0, atol=0)

    def test_work_groups_are_pixel_identical_for_nfe1_and_nfe2(self):
        for nfe in [1, 2]:
            with self.subTest(nfe=nfe):
                group8 = generate_uint8(self.net, self.seeds, nfe, 0.821, 8, "cpu")
                group16 = generate_uint8(self.net, self.seeds, nfe, 0.821, 16, "cpu")
                self.assertEqual(group8.shape, (64, 3, 32, 32))
                self.assertEqual(group8.dtype, np.uint8)
                np.testing.assert_array_equal(group8, group16)

    def test_repeated_runs_are_pixel_identical_for_nfe1_and_nfe2(self):
        for nfe in [1, 2]:
            with self.subTest(nfe=nfe):
                reference = assert_work_group_equivalence(
                    self.net, self.seeds, nfe, 0.821, 8, 16, "cpu"
                )
                assert_repeat_equivalence(
                    self.net, self.seeds, nfe, 0.821, 8, reference, "cpu"
                )

    def test_repeated_png_sha256_is_identical(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for nfe in [1, 2]:
                with self.subTest(nfe=nfe):
                    first = generate_uint8(self.net, self.seeds, nfe, 0.821, 8, "cpu")
                    second = generate_uint8(self.net, self.seeds, nfe, 0.821, 8, "cpu")
                    first_paths = save_mode_outputs(first, self.seeds, root / "first" / f"nfe{nfe}")
                    second_paths = save_mode_outputs(second, self.seeds, root / "second" / f"nfe{nfe}")
                    self.assertEqual(
                        [sha256_file(path) for path in first_paths],
                        [sha256_file(path) for path in second_paths],
                    )

    def test_protocol_outputs_are_complete_rgb_images(self):
        self.assertEqual(self.seeds, list(range(64)))
        self.assertEqual(len(self.seeds), len(set(self.seeds)))

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for nfe in [1, 2]:
                with self.subTest(nfe=nfe):
                    images = generate_uint8(self.net, self.seeds, nfe, 0.821, 8, "cpu")
                    paths = save_mode_outputs(images, self.seeds, root / f"nfe{nfe}")
                    image_paths = paths[:-1]
                    self.assertEqual(len(image_paths), 64)
                    self.assertEqual(
                        {path.name for path in image_paths},
                        {f"seed{seed:06d}.png" for seed in range(64)},
                    )
                    for path in image_paths:
                        with PIL.Image.open(path) as image:
                            self.assertEqual(image.mode, "RGB")
                            self.assertEqual(image.size, (32, 32))

    def test_seed_zero_and_one_are_distinct(self):
        images = generate_uint8(self.net, [0, 1], 1, 0.821, 8, "cpu")
        self.assertNotEqual(images[0].tobytes(), images[1].tobytes())

    def test_metadata_records_nfe_mid_t_and_required_schema(self):
        args = Namespace(
            network="/tmp/model.pkl",
            precision="fp32",
            device="cpu",
            work_group_size=8,
            verify_work_group_size=16,
        )
        modes = [
            {"name": "nfe1", "nfe": 1, "mid_t": [], "image_count": 64, "elapsed_seconds": 1.0},
            {"name": "nfe2", "nfe": 2, "mid_t": [0.821], "image_count": 64, "elapsed_seconds": 2.0},
        ]
        metadata = build_metadata(
            args=args,
            checkpoint_sha256="a" * 64,
            checkpoint_id="model-aaaaaaaaaaaa",
            run_dir=Path("/tmp/evaluations/model-aaaaaaaaaaaa"),
            net=self.net,
            effective_precision="fp32",
            seeds=self.seeds,
            modes=modes,
            elapsed_seconds=3.0,
        )
        self.assertEqual(metadata["schema_version"], "1.0")
        self.assertEqual(metadata["nfe_modes"], [1, 2])
        self.assertEqual(metadata["mid_t_by_mode"], {"nfe1": [], "nfe2": [0.821]})
        self.assertEqual(metadata["image_count_by_mode"], {"nfe1": 64, "nfe2": 64})
        self.assertEqual(metadata["image_count_total"], 128)
        self.assertEqual(metadata["model_forward_batch_size"], 1)
        self.assertTrue(metadata["determinism_passed"])

    def test_manifest_covers_images_grids_and_metadata(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = []
            for nfe in [1, 2]:
                images = generate_uint8(self.net, self.seeds, nfe, 0.821, 8, "cpu")
                for path in save_mode_outputs(images, self.seeds, root / f"nfe{nfe}"):
                    entries.append((sha256_file(path), path.relative_to(root).as_posix()))
            metadata_path = root / "metadata.json"
            metadata_path.write_text("{}\n", encoding="utf-8")
            entries.append((sha256_file(metadata_path), "metadata.json"))
            manifest_path = root / "sha256_manifest.txt"
            write_manifest(entries, manifest_path)

            names = {
                line.split("  ", 1)[1]
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            }
            expected_images = {
                f"nfe{nfe}/images/seed{seed:06d}.png"
                for nfe in [1, 2]
                for seed in self.seeds
            }
            self.assertTrue(expected_images.issubset(names))
            self.assertTrue({"nfe1/grid_8x8.png", "nfe2/grid_8x8.png", "metadata.json"}.issubset(names))
            self.assertEqual(len(names), 131)


if __name__ == "__main__":
    unittest.main()
