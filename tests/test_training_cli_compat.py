import contextlib
import io
import unittest

import dnnlib
import torch
from click.testing import CliRunner

import ct_train
from training.loss import ECMLoss


def parse_train_args(*extra_args):
    args = ['--outdir', 'out', '--data', 'dataset', *extra_args]
    with ct_train.main.make_context('ct_train.py', args) as ctx:
        return dict(ctx.params)


class TrainingCliCompatibilityTest(unittest.TestCase):
    def test_no_new_option_keeps_legacy_sigmoid_default(self):
        params = parse_train_args()
        self.assertEqual(params['mapping'], 'sigmoid')
        self.assertNotIn('schedule', params)
        self.assertEqual(params['adaptive_loss_ema_beta'], 0.9)
        self.assertEqual(params['adaptive_update_kimg'], 0.5)
        self.assertEqual(params['adaptive_warmup_updates'], 2)
        self.assertEqual(params['adaptive_max_adjust'], 0.05)
        self.assertEqual(params['adaptive_min_gap'], 1e-3)

    def test_legacy_mapping_option_is_preserved(self):
        self.assertEqual(parse_train_args('--mapping=const')['mapping'], 'const')
        self.assertEqual(parse_train_args('--mapping=sigmoid')['mapping'], 'sigmoid')

    def test_schedule_and_mapping_are_equivalent_names(self):
        for schedule in ['const', 'sigmoid', 'adaptive_v1']:
            with self.subTest(schedule=schedule):
                legacy = parse_train_args('--mapping', schedule)
                current = parse_train_args('--schedule', schedule)
                self.assertEqual(legacy, current)

    def test_hyphenated_adaptive_name_is_canonicalized(self):
        self.assertEqual(
            parse_train_args('--schedule', 'adaptive-v1')['mapping'],
            'adaptive_v1',
        )

    def test_help_exposes_both_option_names(self):
        result = CliRunner().invoke(ct_train.main, ['--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('--schedule', result.output)
        self.assertIn('--mapping', result.output)

    def test_adaptive_parameters_are_complete_in_loss_config(self):
        params = parse_train_args(
            '--schedule', 'adaptive_v1',
            '--adaptive-loss-ema-beta', '0.8',
            '--adaptive-update-kimg', '0.25',
            '--adaptive-warmup-updates', '3',
            '--adaptive-max-adjust', '0.04',
            '--adaptive-min-gap', '0.002',
        )
        loss_kwargs = ct_train.make_loss_kwargs(dnnlib.EasyDict(params))
        self.assertEqual(loss_kwargs.adj, 'adaptive_v1')
        self.assertEqual(loss_kwargs.adaptive_loss_ema_beta, 0.8)
        self.assertEqual(params['adaptive_update_kimg'], 0.25)
        self.assertEqual(loss_kwargs.adaptive_warmup_updates, 3)
        self.assertEqual(loss_kwargs.adaptive_max_adjust, 0.04)
        self.assertEqual(loss_kwargs.adaptive_min_gap, 0.002)

    def test_explicit_sigmoid_disables_adaptive_v1(self):
        params = parse_train_args('--schedule', 'sigmoid')
        loss_kwargs = ct_train.make_loss_kwargs(dnnlib.EasyDict(params))
        self.assertEqual(loss_kwargs.adj, 'sigmoid')
        with contextlib.redirect_stdout(io.StringIO()):
            loss_fn = ECMLoss(**loss_kwargs)
        loss_fn.update_schedule(3)
        t = torch.tensor([0.01, 0.1, 1.0, 10.0])
        self.assertTrue(torch.equal(
            loss_fn.schedule.compute_r(t=t, stage=loss_fn.stage),
            loss_fn.t_to_r_sigmoid(t),
        ))


if __name__ == '__main__':
    unittest.main()
