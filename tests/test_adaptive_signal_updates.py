import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from training.ct_training_loop import (
    AdaptiveSignalWindow,
    _LEGACY_TRAIN_SUMMARY_FIELDS,
    _PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS,
    _TRAIN_SUMMARY_FIELDS,
    adaptive_update_interval_nimg,
    gather_adaptive_signal_window_state,
    globally_average_runtime_pairs,
    local_adaptive_signal_window_state,
    load_and_migrate_train_summary,
    pid_learning_rate_config,
    pid_learning_rate_multiplier,
    schedule_learning_rate_multiplier,
    set_optimizer_learning_rate,
    validate_pid_learning_rate_resume_config,
)
from training.schedules import get_schedule


class AdaptiveSignalUpdatesTest(unittest.TestCase):
    def test_default_interval_is_half_kimg(self):
        self.assertEqual(adaptive_update_interval_nimg(0.5), 500)

    def test_pid_default_cadence_is_exactly_500_updates_for_25_6_mimg(self):
        update_nimg = adaptive_update_interval_nimg(51.2)
        self.assertEqual(update_nimg, 51_200)
        self.assertEqual(25_600_000 // update_nimg, 500)

    def test_pid_learning_rate_boost_is_bounded_and_warmed_up(self):
        self.assertEqual(
            pid_learning_rate_multiplier(0.1, cur_nimg=0),
            1.0,
        )
        halfway = pid_learning_rate_multiplier(
            0.0, cur_nimg=128_000, boost=1.25,
            max_boost=1.5, warmup_kimg=256.0,
        )
        self.assertAlmostEqual(halfway, 1.125)
        active = pid_learning_rate_multiplier(
            math.log(1.1), cur_nimg=256_000, boost=1.25,
            max_boost=1.5, warmup_kimg=256.0,
        )
        self.assertAlmostEqual(active, 1.375)
        saturated = pid_learning_rate_multiplier(
            0.5, cur_nimg=256_000, boost=1.25,
            max_boost=1.5, warmup_kimg=0,
        )
        self.assertEqual(saturated, 1.5)
        self.assertEqual(
            pid_learning_rate_multiplier(
                -1.0, cur_nimg=256_000, boost=1.25,
                max_boost=1.5, warmup_kimg=0,
            ),
            1.0,
        )

    def test_non_pid_schedules_keep_base_learning_rate(self):
        for schedule in ('const', 'sigmoid', 'adaptive_v1'):
            with self.subTest(schedule=schedule):
                self.assertEqual(
                    schedule_learning_rate_multiplier(
                        schedule, control_output=0.1, cur_nimg=1_000_000,
                        pid_lr_boost=1.25, pid_lr_max_boost=1.5,
                        pid_lr_warmup_kimg=0,
                    ),
                    1.0,
                )

    def test_pid_learning_rate_rejects_invalid_state(self):
        invalid = [
            {'control_output': float('nan')},
            {'control_output': 81.0},
            {'control_output': 0.0, 'cur_nimg': -1},
            {'control_output': 0.0, 'boost': 0.9},
            {'control_output': 0.0, 'max_boost': 0.9},
            {'control_output': 0.0, 'warmup_kimg': -1.0},
        ]
        for kwargs in invalid:
            params = dict(control_output=0.0, cur_nimg=0)
            params.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pid_learning_rate_multiplier(**params)

    def test_pid_learning_rate_resume_config_must_match_exactly(self):
        current = pid_learning_rate_config(
            1e-4, boost=1.25, max_boost=1.5, warmup_kimg=256.0
        )
        validate_pid_learning_rate_resume_config(dict(current), current)
        for saved in [
            None,
            {key: value for key, value in current.items() if key != 'boost'},
            {**current, 'version': 2},
            {**current, 'base_learning_rate': 2e-4},
            {**current, 'boost': 1.3},
            {**current, 'max_boost': 1.6},
            {**current, 'warmup_kimg': 128.0},
        ]:
            with self.subTest(saved=saved), self.assertRaises(RuntimeError):
                validate_pid_learning_rate_resume_config(saved, current)

    def test_effective_learning_rate_reaches_every_optimizer_group(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD(
            [{'params': [first]}, {'params': [second], 'lr': 2e-4}],
            lr=1e-4,
        )
        actual = set_optimizer_learning_rate(optimizer, 1e-4, 1.25)
        self.assertEqual(actual, 1.25e-4)
        self.assertTrue(all(group['lr'] == actual for group in optimizer.param_groups))
        with self.assertRaises(ValueError):
            set_optimizer_learning_rate(optimizer, 1e-4, float('inf'))

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

    def test_resume_preserves_partial_window_loss_aggregation(self):
        uninterrupted = AdaptiveSignalWindow(update_kimg=0.5)
        uninterrupted.add(loss_sum=12.0, loss_count=4)
        self.assertIsNone(uninterrupted.pop_if_due(cur_nimg=384))

        resumed = AdaptiveSignalWindow(update_kimg=0.5, start_nimg=384)
        checkpoint_state = gather_adaptive_signal_window_state(
            uninterrupted, device=torch.device('cpu')
        )
        self.assertEqual(checkpoint_state['next_update_nimg'], 500)
        self.assertEqual(checkpoint_state['loss_sum'], 12.0)
        self.assertEqual(checkpoint_state['loss_count'], 4)
        resumed.load_state_dict(local_adaptive_signal_window_state(checkpoint_state))
        self.assertEqual(resumed.state_dict(), uninterrupted.state_dict())

        uninterrupted.add(loss_sum=8.0, loss_count=2)
        resumed.add(loss_sum=8.0, loss_count=2)
        self.assertEqual(uninterrupted.pop_if_due(cur_nimg=512), (20.0, 6))
        self.assertEqual(resumed.pop_if_due(cur_nimg=512), (20.0, 6))
        self.assertEqual(resumed.state_dict(), uninterrupted.state_dict())

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

    def test_resume_migrates_exact_legacy_summary_schema(self):
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
                'adaptive_active', 'r_over_t_mean', 'gap_mean', 'next_loop_cur_tick',
            ):
                self.assertEqual(migrated[field], '')
                self.assertEqual(rows[0][field], '')

            current_rows, second_backup = load_and_migrate_train_summary(summary_path)
            self.assertIsNone(second_backup)
            self.assertEqual(current_rows, rows)

    def test_resume_migrates_pre_next_loop_tick_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / 'train_summary.csv'
            row = {field: '' for field in _PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS}
            row.update(
                attempted_iteration='4', successful_optimizer_steps='4',
                processed_nimg='512', processed_kimg='0.512', loss='1.25',
                grad_scale='65536', step_skipped='0', schedule='adaptive_v1',
                stage='0', loss_ema='0.8', correction='0.02',
                elapsed_sec='2.0', peak_vram_gb='1.5',
            )
            with summary_path.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=_PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS
                )
                writer.writeheader()
                writer.writerow(row)

            rows, backup_path = load_and_migrate_train_summary(summary_path)
            self.assertEqual(backup_path, f'{summary_path}.pre-next-loop-tick.bak')
            self.assertEqual(rows[0]['next_loop_cur_tick'], '')
            self.assertEqual(rows[0]['loss_ema'], '0.8')
            self.assertEqual(rows[0]['correction'], '0.02')
            with summary_path.open(newline='') as handle:
                self.assertEqual(
                    tuple(csv.DictReader(handle).fieldnames), _TRAIN_SUMMARY_FIELDS
                )

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
