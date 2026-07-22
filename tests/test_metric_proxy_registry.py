"""Unit tests for the frozen 5k proxy metric definitions.

These tests deliberately mock the expensive feature-statistics implementations.
They verify the metric registry and the exact sample budgets used by the quality
evaluation protocol without requiring a GPU or pretrained Inception weights.
"""

from types import SimpleNamespace
import unittest
from unittest import mock

import ct_eval
from metrics import metric_main


class MetricProxyRegistryTest(unittest.TestCase):
    def test_proxy_metric_names_are_registered(self):
        valid_metrics = metric_main.list_valid_metrics()
        self.assertIn("fid5k_proxy", valid_metrics)
        self.assertIn("kid5k_proxy", valid_metrics)
        self.assertTrue(metric_main.is_valid_metric("fid5k_proxy"))
        self.assertTrue(metric_main.is_valid_metric("kid5k_proxy"))

    def test_metric_repeats_is_explicit_and_backwards_compatible(self):
        option = next(param for param in ct_eval.main.params if param.name == "metric_repeats")
        self.assertEqual(option.default, 3)
        self.assertEqual(option.type.min, 1)

    def test_fid5k_proxy_uses_frozen_sample_budgets(self):
        opts = SimpleNamespace(dataset_kwargs={})
        with mock.patch.object(
            metric_main.frechet_inception_distance, "compute_fid", return_value=12.5
        ) as compute_fid:
            result = metric_main.fid5k_proxy(opts)

        self.assertEqual(result, {"fid5k_proxy": 12.5})
        self.assertEqual(opts.dataset_kwargs, {"max_size": 50000, "xflip": False})
        compute_fid.assert_called_once_with(opts, max_real=50000, num_gen=5000)

    def test_kid5k_proxy_uses_frozen_sample_budgets(self):
        opts = SimpleNamespace(dataset_kwargs={})
        with mock.patch.object(
            metric_main.kernel_inception_distance, "compute_kid", return_value=0.0125
        ) as compute_kid:
            result = metric_main.kid5k_proxy(opts)

        self.assertEqual(result, {"kid5k_proxy": 0.0125})
        self.assertEqual(opts.dataset_kwargs, {"max_size": 50000, "xflip": False})
        compute_kid.assert_called_once_with(
            opts,
            max_real=50000,
            num_gen=5000,
            num_subsets=100,
            max_subset_size=1000,
        )


if __name__ == "__main__":
    unittest.main()
