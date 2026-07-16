import unittest

from click.testing import CliRunner

import ct_train
from training.ct_training_loop import _schedule_stage


def parse_train_args(*extra_args):
    args = ['--outdir', 'out', '--data', 'dataset', *extra_args]
    with ct_train.main.make_context('ct_train.py', args) as ctx:
        return dict(ctx.params)


class TrainingCliCompatibilityTest(unittest.TestCase):
    def test_no_new_option_keeps_legacy_sigmoid_default(self):
        params = parse_train_args()
        self.assertEqual(params['mapping'], 'sigmoid')
        self.assertNotIn('schedule', params)

    def test_legacy_mapping_option_is_preserved(self):
        self.assertEqual(parse_train_args('--mapping=const')['mapping'], 'const')
        self.assertEqual(parse_train_args('--mapping=sigmoid')['mapping'], 'sigmoid')

    def test_schedule_and_mapping_are_equivalent_names(self):
        for schedule in ['const', 'sigmoid', 'adaptive_v1']:
            with self.subTest(schedule=schedule):
                legacy = parse_train_args('--mapping', schedule)
                current = parse_train_args('--schedule', schedule)
                self.assertEqual(legacy, current)

    def test_help_exposes_both_option_names(self):
        result = CliRunner().invoke(ct_train.main, ['--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('--schedule', result.output)
        self.assertIn('--mapping', result.output)


class TrainingStageCompatibilityTest(unittest.TestCase):
    def test_legacy_schedules_keep_integer_boundaries(self):
        for schedule in [None, 'const', 'sigmoid']:
            with self.subTest(schedule=schedule):
                self.assertEqual(_schedule_stage(schedule, 499, 500), 0)
                self.assertEqual(_schedule_stage(schedule, 500, 500), 1)

    def test_adaptive_schedule_uses_fractional_progress(self):
        self.assertEqual(_schedule_stage('adaptive_v1', 125, 500), 0.25)
        self.assertEqual(_schedule_stage('adaptive_v1', 500, 500), 1.0)


if __name__ == '__main__':
    unittest.main()
