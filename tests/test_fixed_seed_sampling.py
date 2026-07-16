import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import torch

from scripts.sample_fixed_seeds import (
    assert_repeat_equivalence,
    assert_work_group_equivalence,
    configure_precision,
    generate_uint8,
    save_mode_outputs,
    sha256_file,
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
    def test_precision_is_explicit_and_reported(self):
        net = DummyNet().eval()
        self.assertEqual(configure_precision(net, "checkpoint", "cpu"), "fp32")
        self.assertEqual(configure_precision(net, "fp32", "cpu"), "fp32")
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            configure_precision(net, "fp16", "cpu")

    def test_work_groups_are_pixel_identical_for_nfe1_and_nfe2(self):
        net = DummyNet().eval()
        seeds = list(range(64))
        for nfe in [1, 2]:
            with self.subTest(nfe=nfe):
                group8 = generate_uint8(net, seeds, nfe, 0.821, 8, "cpu")
                group16 = generate_uint8(net, seeds, nfe, 0.821, 16, "cpu")
                self.assertEqual(group8.shape, (64, 3, 32, 32))
                self.assertEqual(group8.dtype, np.uint8)
                np.testing.assert_array_equal(group8, group16)

    def test_repeated_runs_are_pixel_identical_for_nfe1_and_nfe2(self):
        net = DummyNet().eval()
        seeds = list(range(64))
        for nfe in [1, 2]:
            with self.subTest(nfe=nfe):
                reference = assert_work_group_equivalence(
                    net, seeds, nfe, 0.821, 8, 16, "cpu"
                )
                assert_repeat_equivalence(
                    net, seeds, nfe, 0.821, 8, reference, "cpu"
                )

    def test_repeated_png_sha256_is_identical(self):
        net = DummyNet().eval()
        seeds = list(range(64))
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for nfe in [1, 2]:
                with self.subTest(nfe=nfe):
                    first = generate_uint8(net, seeds, nfe, 0.821, 8, "cpu")
                    second = generate_uint8(net, seeds, nfe, 0.821, 8, "cpu")
                    first_paths = save_mode_outputs(first, seeds, root / "first" / f"nfe{nfe}")
                    second_paths = save_mode_outputs(second, seeds, root / "second" / f"nfe{nfe}")
                    self.assertEqual(
                        [sha256_file(path) for path in first_paths],
                        [sha256_file(path) for path in second_paths],
                    )

    def test_protocol_outputs_are_complete_rgb_images(self):
        net = DummyNet().eval()
        seeds = list(range(64))
        self.assertEqual(seeds, list(range(64)))
        self.assertEqual(len(seeds), len(set(seeds)))

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for nfe in [1, 2]:
                with self.subTest(nfe=nfe):
                    images = generate_uint8(net, seeds, nfe, 0.821, 8, "cpu")
                    paths = save_mode_outputs(images, seeds, root / f"nfe{nfe}")
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


if __name__ == "__main__":
    unittest.main()
