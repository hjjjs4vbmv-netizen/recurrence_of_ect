import contextlib
import io
import math
import pickle
import unittest

import torch

from training.loss import ECMLoss
from training.schedules import get_schedule


def sample_t(device='cpu'):
    return torch.tensor([0.002, 0.02, 0.2, 2.0, 20.0, 80.0], device=device)


def make_loss(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return ECMLoss(adj='adaptive_v2_dualema', **kwargs)


class AdaptiveV2DualEMATest(unittest.TestCase):
    def make_controller(self, **kwargs):
        defaults = dict(
            beta_fast=0.5,
            beta_slow=0.9,
            warmup_updates=1,
            max_adjust=0.1,
        )
        defaults.update(kwargs)
        return get_schedule('adaptive_v2_dualema', **defaults)

    def test_a_warmup_equal_emas_and_zero_adjust_are_exact_baseline(self):
        t = sample_t()
        baseline = get_schedule('sigmoid')
        expected = baseline.compute_r(t=t, stage=3)

        warmup = self.make_controller(warmup_updates=3)
        for loss in [10.0, 9.0]:
            warmup.update_training_signal(loss)
            self.assertTrue(torch.equal(warmup.compute_r(t=t, stage=3), expected))

        equal = self.make_controller(warmup_updates=1)
        equal.update_training_signal(10.0)
        self.assertEqual(equal.fast_loss_ema, equal.slow_loss_ema)
        self.assertEqual(equal.correction(), 0.0)
        self.assertTrue(torch.equal(equal.compute_r(t=t, stage=3), expected))

        zero = self.make_controller(max_adjust=0.0)
        for loss in [10.0, 20.0, 40.0]:
            zero.update_training_signal(loss)
        self.assertTrue(torch.equal(zero.compute_r(t=t, stage=3), expected))

    def test_b_ema_values_match_hand_calculation(self):
        controller = self.make_controller(beta_fast=0.5, beta_slow=0.75)
        losses = [8.0, 4.0, 10.0]
        expected_fast = [8.0, 6.0, 8.0]
        expected_slow = [8.0, 7.0, 7.75]
        for loss, fast, slow in zip(losses, expected_fast, expected_slow):
            self.assertTrue(controller.update_training_signal(loss))
            self.assertAlmostEqual(controller.fast_loss_ema, fast)
            self.assertAlmostEqual(controller.slow_loss_ema, slow)

    def test_c_worsening_raises_rho_and_shrinks_gap(self):
        controller = self.make_controller()
        controller.update_training_signal(10.0)
        controller.update_training_signal(20.0)
        r = controller.compute_r(sample_t(), stage=3)
        metrics = controller.runtime_metrics()
        self.assertGreater(metrics['raw_error'], 0)
        self.assertGreater(metrics['correction'], 0)
        self.assertGreater(metrics['adaptive_rho'], metrics['baseline_rho'])
        self.assertLess(metrics['adaptive_gap'], metrics['baseline_gap'])
        self.assertTrue(torch.isfinite(r).all())

    def test_c_improving_lowers_rho_and_expands_gap(self):
        controller = self.make_controller()
        controller.update_training_signal(10.0)
        controller.update_training_signal(5.0)
        controller.compute_r(sample_t(), stage=3)
        metrics = controller.runtime_metrics()
        self.assertLess(metrics['raw_error'], 0)
        self.assertLess(metrics['correction'], 0)
        self.assertLess(metrics['adaptive_rho'], metrics['baseline_rho'])
        self.assertGreater(metrics['adaptive_gap'], metrics['baseline_gap'])

    def test_d_loss_update_only_affects_a_later_schedule_call(self):
        controller = self.make_controller()
        t = sample_t()
        controller.set_training_iteration(1)
        pair_k = controller.compute_r(t, stage=3)
        controller.update_training_signal(10.0)
        self.assertEqual(controller.correction(), 0.0)
        controller.set_training_iteration(2)
        pair_before_loss_k = controller.compute_r(t, stage=3)
        controller.update_training_signal(20.0)
        self.assertTrue(torch.equal(pair_k, pair_before_loss_k))
        controller.set_training_iteration(3)
        pair_k_plus_1 = controller.compute_r(t, stage=3)
        self.assertFalse(torch.equal(pair_before_loss_k, pair_k_plus_1))
        self.assertEqual(controller.first_nonzero_correction_iteration, 3)
        self.assertEqual(controller.first_adapted_pair_iteration, 3)

    def test_e_extreme_error_is_bounded_and_output_is_legal(self):
        for losses in ([1e-100, 1e100], [1e100, 1e-100]):
            with self.subTest(losses=losses):
                controller = self.make_controller(max_adjust=0.05, min_gap=1e-3)
                for loss in losses:
                    controller.update_training_signal(loss)
                t = torch.tensor([1e-8, 0.1, 1.0, 80.0])
                r = controller.compute_r(t, stage=100)
                self.assertLessEqual(abs(controller.correction()), 0.05)
                self.assertTrue(torch.isfinite(r).all())
                self.assertTrue((r >= 0).all())
                self.assertTrue((r < t).all())

    def test_f_nonfinite_and_nonpositive_signals_do_not_mutate_emas(self):
        controller = self.make_controller()
        controller.update_training_signal(10.0)
        before = (controller.fast_loss_ema, controller.slow_loss_ema,
                  controller.signal_updates, controller.last_correction)
        for value in [float('nan'), float('inf'), -float('inf'), 0.0, -1.0]:
            self.assertFalse(controller.update_training_signal(value))
            self.assertEqual(
                (controller.fast_loss_ema, controller.slow_loss_ema,
                 controller.signal_updates, controller.last_correction),
                before,
            )
            self.assertFalse(controller.runtime_metrics()['finite_signal'])
        self.assertEqual(controller.nonfinite_signal_count, 5)
        self.assertTrue(torch.isfinite(controller.compute_r(sample_t(), stage=3)).all())

    def test_g_state_round_trip_and_resume_are_exact(self):
        source = self.make_controller(warmup_updates=2)
        for iteration, loss in enumerate([10.0, 12.0, 8.0], start=1):
            source.set_training_iteration(iteration)
            source.compute_r(sample_t(), stage=3)
            source.update_training_signal(loss)
        source.set_training_iteration(4)
        expected = source.compute_r(sample_t(), stage=3)

        clone = self.make_controller(warmup_updates=2)
        clone.load_state_dict(source.state_dict())
        self.assertEqual(clone.state_dict(), source.state_dict())
        self.assertTrue(torch.equal(clone.compute_r(sample_t(), stage=3), expected))
        clone.update_training_signal(7.0)
        source.update_training_signal(7.0)
        self.assertEqual(clone.fast_loss_ema, source.fast_loss_ema)
        self.assertEqual(clone.slow_loss_ema, source.slow_loss_ema)
        self.assertEqual(clone.signal_updates, source.signal_updates)

    def test_g_old_checkpoint_without_v2_state_initializes_safely(self):
        controller = self.make_controller()
        controller.load_state_dict({})
        self.assertIsNone(controller.fast_loss_ema)
        self.assertEqual(controller.signal_updates, 0)
        self.assertTrue(torch.equal(
            controller.compute_r(sample_t(), stage=2),
            get_schedule('sigmoid').compute_r(sample_t(), stage=2),
        ))

    def test_g_loss_pickle_and_schedule_state_preserve_controller(self):
        source = make_loss(
            adaptive_fast_beta=0.5,
            adaptive_slow_beta=0.9,
            adaptive_warmup_updates=1,
        )
        source.update_schedule(3)
        source.set_training_iteration(1)
        source.update_training_signal(10.0)
        source.set_training_iteration(2)
        source.update_training_signal(20.0)
        expected = source.schedule.compute_r(sample_t(), source.stage)
        clone = pickle.loads(pickle.dumps(source))
        self.assertTrue(torch.equal(clone.schedule.compute_r(sample_t(), clone.stage), expected))
        self.assertEqual(clone.schedule.state_dict(), source.schedule.state_dict())

    def test_h_invalid_parameters_are_rejected(self):
        invalid = [
            dict(beta_fast=0.0),
            dict(beta_fast=0.9, beta_slow=0.8),
            dict(beta_slow=1.0),
            dict(max_adjust=-0.1),
            dict(min_gap=0.0),
            dict(warmup_updates=-1),
            dict(warmup_updates=1.5),
            dict(eps=0.0),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_controller(**kwargs)

    def test_i_cpu_and_optional_cuda_have_no_device_mismatch_or_controller_graph(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                controller = self.make_controller()
                controller.update_training_signal(torch.tensor(10.0, device=device))
                controller.update_training_signal(torch.tensor(20.0, device=device))
                r = controller.compute_r(sample_t(device=device), stage=3)
                self.assertEqual(r.device.type, device)
                self.assertIsInstance(controller.fast_loss_ema, float)
                self.assertIsInstance(controller.slow_loss_ema, float)
                self.assertIsInstance(controller.correction(), float)
                self.assertFalse(math.isnan(controller.runtime_metrics()['raw_error']))


if __name__ == '__main__':
    unittest.main()
