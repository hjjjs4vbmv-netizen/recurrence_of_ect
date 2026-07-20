import unittest

from training.ct_training_loop import AdaptiveSignalWindow, adaptive_update_interval_nimg
from training.schedules import get_schedule


class AdaptiveSignalUpdatesTest(unittest.TestCase):
    def test_default_interval_is_half_kimg(self):
        self.assertEqual(adaptive_update_interval_nimg(0.5), 500)

    def test_updates_at_absolute_boundaries_not_maintenance_ticks(self):
        window = AdaptiveSignalWindow(update_kimg=0.5, start_nimg=0)
        window.add(loss_sum=12.0, loss_count=4)
        self.assertIsNone(window.pop_if_due(cur_nimg=499))
        window.add(loss_sum=8.0, loss_count=2)
        self.assertEqual(window.pop_if_due(cur_nimg=512), (20.0, 6))
        self.assertEqual(window.next_update_nimg, 1000)

        # A 50 kimg maintenance boundary has no special meaning here.
        window.add(loss_sum=9.0, loss_count=3)
        self.assertIsNone(window.pop_if_due(cur_nimg=999))
        self.assertEqual(window.pop_if_due(cur_nimg=1000), (9.0, 3))

    def test_resume_uses_next_absolute_boundary(self):
        window = AdaptiveSignalWindow(update_kimg=0.5, start_nimg=50_000)
        self.assertEqual(window.next_update_nimg, 50_500)

    def test_interval_must_be_whole_positive_images(self):
        for value in [0, -0.5, 0.0005, float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                adaptive_update_interval_nimg(value)

    def test_activation_budget_reaches_nonzero_correction_with_iterations_left(self):
        batch_size = 128
        final_iteration = 4096 // batch_size
        window = AdaptiveSignalWindow(update_kimg=0.5)
        schedule = get_schedule('adaptive_v1', loss_ema_beta=0.0, warmup_updates=2)
        first_nonzero_correction_iteration = None

        for attempted_iteration in range(1, final_iteration + 1):
            window.add(loss_sum=1.0, loss_count=1)
            if window.pop_if_due(attempted_iteration * batch_size) is not None:
                # Finite improving signals make the first post-warmup
                # correction observably nonzero.
                schedule.update_training_signal(10.0 / (schedule.signal_updates + 1))
                if schedule.correction() != 0 and first_nonzero_correction_iteration is None:
                    # Signal processing follows optimizer.step(), so the
                    # correction affects the next attempted iteration.
                    first_nonzero_correction_iteration = attempted_iteration + 1

        self.assertGreaterEqual(schedule.signal_updates, 3)
        self.assertIsNotNone(first_nonzero_correction_iteration)
        self.assertLess(first_nonzero_correction_iteration, final_iteration)
        self.assertGreaterEqual(
            final_iteration - first_nonzero_correction_iteration,
            4,
        )


if __name__ == '__main__':
    unittest.main()
