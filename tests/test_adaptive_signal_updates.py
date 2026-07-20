import csv
import tempfile
import unittest
from pathlib import Path

import torch

from training.ct_training_loop import (
    AdaptiveSignalWindow,
    _LEGACY_TRAIN_SUMMARY_FIELDS,
    _TRAIN_SUMMARY_FIELDS,
    adaptive_update_interval_nimg,
    globally_average_runtime_pairs,
    load_and_migrate_train_summary,
)
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

    def test_runtime_pair_metrics_average_without_schedule_internals(self):
        metrics = globally_average_runtime_pairs(
            [
                {'r_over_t_mean': 0.6, 'gap_mean': 0.4},
                {'r_over_t_mean': 0.8, 'gap_mean': 0.2},
            ],
            device=torch.device('cpu'),
        )
        self.assertAlmostEqual(metrics['r_over_t_mean'], 0.7)
        self.assertAlmostEqual(metrics['gap_mean'], 0.3)

    def test_resume_migrates_only_exact_legacy_summary_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / 'train_summary.csv'
            legacy_row = {
                'attempted_iteration': '4',
                'successful_optimizer_steps': '4',
                'processed_nimg': '512',
                'processed_kimg': '0.512',
                'loss': '1.25',
                'grad_scale': '65536',
                'step_skipped': '0',
                'schedule': 'adaptive_v1',
                'stage': '0',
                'elapsed_sec': '2.0',
                'peak_vram_gb': '1.5',
            }
            with summary_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=_LEGACY_TRAIN_SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(legacy_row)

            rows, backup_path = load_and_migrate_train_summary(summary_path)
            self.assertEqual(backup_path, f'{summary_path}.pre-telemetry.bak')
            self.assertTrue(Path(backup_path).is_file())
            with Path(backup_path).open(newline='') as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames), _LEGACY_TRAIN_SUMMARY_FIELDS)
            with summary_path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                migrated = next(reader)
                self.assertEqual(tuple(reader.fieldnames), _TRAIN_SUMMARY_FIELDS)
            for field in (
                'loss_ema', 'loss_reference', 'correction', 'signal_updates',
                'adaptive_active', 'r_over_t_mean', 'gap_mean',
            ):
                self.assertEqual(migrated[field], '')
                self.assertEqual(rows[0][field], '')

            current_rows, second_backup = load_and_migrate_train_summary(summary_path)
            self.assertIsNone(second_backup)
            self.assertEqual(current_rows, rows)

    def test_resume_rejects_unknown_summary_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / 'train_summary.csv'
            with summary_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=('attempted_iteration', 'loss'))
                writer.writeheader()
                writer.writerow({'attempted_iteration': '1', 'loss': '1.0'})
            with self.assertRaisesRegex(RuntimeError, 'unsupported schema'):
                load_and_migrate_train_summary(summary_path)
            self.assertFalse(Path(f'{summary_path}.pre-telemetry.bak').exists())

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
