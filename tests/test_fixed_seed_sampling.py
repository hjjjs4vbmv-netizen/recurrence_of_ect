import unittest

import numpy as np
import torch

from scripts.sample_fixed_seeds import generate_uint8


class DummyNet(torch.nn.Module):
    img_channels = 3
    img_resolution = 32

    def round_sigma(self, sigma):
        return sigma

    def forward(self, x, sigma, class_labels=None):
        del sigma, class_labels
        # Deliberately depend on the forward batch composition. The fixed-seed
        # sampler must still be invariant to its outer work-group size.
        batch_term = x.mean(dim=0, keepdim=True) * 0.01
        return torch.tanh((x + batch_term) / 80)


class FixedSeedSamplingTest(unittest.TestCase):
    def test_batches_are_pixel_identical_for_nfe1_and_nfe2(self):
        net = DummyNet().eval()
        seeds = list(range(64))
        for nfe in [1, 2]:
            with self.subTest(nfe=nfe):
                batch8 = generate_uint8(net, seeds, nfe, 0.821, 8, "cpu")
                batch16 = generate_uint8(net, seeds, nfe, 0.821, 16, "cpu")
                self.assertEqual(batch8.shape, (64, 3, 32, 32))
                self.assertEqual(batch8.dtype, np.uint8)
                np.testing.assert_array_equal(batch8, batch16)


if __name__ == "__main__":
    unittest.main()
