import contextlib
import io
import pickle
import unittest

import torch

from training import schedules
from training.loss import ECMLoss
from training.schedules import compute_r, continuous_stage, get_schedule

# LogNormal(P_mean, P_std) noise levels, as sampled by ECMLoss.__call__.
P_MEAN = -1.1
P_STD = 2.0


def sample_t(n=512, dtype=torch.float32, seed=0, device='cpu'):
    generator = torch.Generator().manual_seed(seed)
    rnd_normal = torch.randn([n, 1, 1, 1], generator=generator, dtype=torch.float64)
    return (rnd_normal * P_STD + P_MEAN).exp().to(dtype=dtype, device=device)


def devices():
    """cpu always; cuda too when available (i.e. on the A100 server)."""
    return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def make_loss(adj, q=2.0, k=8.0, b=1.0):
    with contextlib.redirect_stdout(io.StringIO()):  # silence dist.print0
        return ECMLoss(q=q, k=k, b=b, adj=adj)


def official_t_to_r(adj, t, stage, q=2.0, k=8.0, b=1.0):
    """Reference output from the untouched official formulas in
    training/loss.py (t_to_r_const / t_to_r_sigmoid), NOT from the schedule
    dispatch, so the parity anchor stays independent of schedules.py."""
    loss_fn = make_loss(adj, q=q, k=k, b=b)
    loss_fn.update_schedule(stage)
    reference = {'const': loss_fn.t_to_r_const, 'sigmoid': loss_fn.t_to_r_sigmoid}[adj]
    return reference(t)


class OfficialFormulaParityTest(unittest.TestCase):
    """'const' and 'sigmoid' must reproduce training/loss.py bit-for-bit."""

    def test_bitwise_parity_with_ecm_loss(self):
        for device in devices():
            for adj in ['const', 'sigmoid']:
                for q, k, b in [(2.0, 8.0, 1.0), (256.0, 8.0, 1.0), (4.0, 4.0, 2.0)]:
                    for stage in [0, 1, 3, 7]:
                        for dtype in [torch.float32, torch.float64]:
                            with self.subTest(device=device, adj=adj, q=q, k=k, b=b, stage=stage, dtype=dtype):
                                t = sample_t(dtype=dtype, device=device)
                                expected = official_t_to_r(adj, t, stage, q=q, k=k, b=b)
                                via_object = get_schedule(adj, q=q, k=k, b=b).compute_r(t=t, stage=stage)
                                via_function = compute_r(t=t, stage=stage, schedule=adj, q=q, k=k, b=b)
                                self.assertTrue(torch.equal(via_object, expected))
                                self.assertTrue(torch.equal(via_function, expected))

    def test_const_stage0_halves_t_exactly(self):
        # q=2, stage=0: decay = 1/2, so r = t/2 with no clamping.
        t = sample_t()
        r = compute_r(t=t, stage=0, schedule='const', q=2.0)
        self.assertTrue(torch.equal(r, t * 0.5))

    def test_sigmoid_clamps_small_t_to_zero_at_stage0(self):
        # q=2, stage=0: decay = 1/2 and n(t) -> 1 + k/2 = 5 as t -> 0, so
        # ratio < 0 and r must clamp to 0 (diffusion-pretraining regime).
        t = torch.full([8, 1, 1, 1], 1e-4)
        r = compute_r(t=t, stage=0, schedule='sigmoid', q=2.0, k=8.0, b=1.0)
        self.assertTrue(torch.equal(r, torch.zeros_like(t)))

    def test_r_is_nonnegative_and_strictly_below_t(self):
        t = sample_t()
        for name in schedules.available_schedules():
            for stage in [0, 2, 5]:
                with self.subTest(schedule=name, stage=stage):
                    r = compute_r(t=t, stage=stage, schedule=name, q=2.0)
                    self.assertTrue((r >= 0).all())
                    self.assertTrue((r < t).all())

    def test_shape_and_dtype_preserved(self):
        for dtype in [torch.float32, torch.float64]:
            t = sample_t(dtype=dtype)
            r = compute_r(t=t, stage=2, schedule='sigmoid')
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r.dtype, t.dtype)


class AdaptiveV1Test(unittest.TestCase):
    def test_matches_official_sigmoid_at_integer_stages(self):
        for device in devices():
            for q in [2.0, 256.0]:
                for stage in [0, 1, 3, 7]:
                    with self.subTest(device=device, q=q, stage=stage):
                        t = sample_t(device=device)
                        expected = official_t_to_r('sigmoid', t, stage, q=q)
                        actual = compute_r(t=t, stage=stage, schedule='adaptive_v1', q=q)
                        self.assertTrue(torch.equal(actual, expected))

    def test_fractional_stage_stays_between_bracketing_stages(self):
        t = sample_t()
        adaptive = get_schedule('adaptive_v1', q=2.0)
        baseline = get_schedule('sigmoid', q=2.0)
        for lo in [0, 1, 4]:
            with self.subTest(lo=lo):
                r_lo = baseline.compute_r(t=t, stage=lo)
                r_hi = baseline.compute_r(t=t, stage=lo + 1)
                r_mid = adaptive.compute_r(t=t, stage=lo + 0.5)
                self.assertTrue((r_mid >= r_lo).all())
                self.assertTrue((r_mid <= r_hi).all())

    def test_r_tightens_monotonically_with_progress(self):
        t = sample_t()
        adaptive = get_schedule('adaptive_v1', q=2.0)
        rs = [adaptive.compute_r(t=t, stage=s) for s in [0, 0.25, 0.5, 1.0, 1.75, 3.0, 6.5]]
        for r_prev, r_next in zip(rs, rs[1:]):
            self.assertTrue((r_next >= r_prev).all())

    def test_negative_stage_rejected(self):
        with self.assertRaises(ValueError):
            compute_r(t=sample_t(), stage=-0.5, schedule='adaptive_v1')

    def test_continuous_stage_helper(self):
        self.assertEqual(continuous_stage(cur_tick=125, double_ticks=250), 0.5)
        self.assertEqual(continuous_stage(cur_tick=500, double_ticks=250), 2.0)
        with self.assertRaises(ValueError):
            continuous_stage(cur_tick=1, double_ticks=0)


class InterfaceTest(unittest.TestCase):
    def test_documented_call_forms_agree(self):
        t = sample_t()
        schedule = get_schedule('sigmoid', q=256, k=8, b=1)
        r_object = schedule.compute_r(t=t, stage=1)
        r_function = compute_r(t=t, stage=1, schedule='sigmoid', q=256, k=8, b=1)
        self.assertTrue(torch.equal(r_object, r_function))

    def test_stateful_ecm_loss_style_interface(self):
        t = sample_t()
        schedule = get_schedule('sigmoid')
        schedule.update_schedule(3)
        self.assertTrue(torch.equal(schedule.t_to_r(t), schedule.compute_r(t=t, stage=3)))

    def test_scalar_input_is_converted_to_tensor(self):
        r = compute_r(t=2.0, stage=0, schedule='const', q=2.0)
        self.assertIsInstance(r, torch.Tensor)
        self.assertAlmostEqual(float(r), 1.0)

    def test_available_schedules(self):
        self.assertEqual(schedules.available_schedules(), ['adaptive_v1', 'const', 'sigmoid'])

    def test_unknown_schedule_rejected(self):
        with self.assertRaises(ValueError):
            get_schedule('cosine')
        with self.assertRaises(ValueError):
            compute_r(t=sample_t(), stage=0, schedule='cosine')

    def test_invalid_q_rejected(self):
        with self.assertRaises(ValueError):
            get_schedule('const', q=1.0)


class ECMLossIntegrationTest(unittest.TestCase):
    """ECMLoss now dispatches t->r through training/schedules.py; the wired
    entry must be indistinguishable from the official reference formulas."""

    def test_loss_holds_matching_schedule_instance(self):
        for adj, cls in [('const', schedules.ConstSchedule),
                         ('sigmoid', schedules.SigmoidSchedule),
                         ('adaptive_v1', schedules.AdaptiveV1Schedule)]:
            with self.subTest(adj=adj):
                self.assertIsInstance(make_loss(adj).schedule, cls)

    def test_wired_entry_matches_official_reference_bitwise(self):
        for adj in ['const', 'sigmoid']:
            for stage in [0, 2, 5]:
                with self.subTest(adj=adj, stage=stage):
                    loss_fn = make_loss(adj, q=2.0, k=8.0, b=1.0)
                    loss_fn.update_schedule(stage)
                    t = sample_t()
                    reference = getattr(loss_fn, f't_to_r_{adj}')(t)
                    wired = loss_fn.schedule.compute_r(t=t, stage=loss_fn.stage)
                    self.assertTrue(torch.equal(wired, reference))

    def test_loss_hyperparams_reach_the_schedule(self):
        loss_fn = make_loss('sigmoid', q=256.0, k=4.0, b=2.0)
        self.assertEqual((loss_fn.schedule.q, loss_fn.schedule.k, loss_fn.schedule.b), (256.0, 4.0, 2.0))

    def test_unknown_adj_still_raises_value_error(self):
        with self.assertRaises(ValueError):
            make_loss('cosine')

    def test_update_schedule_keeps_ratio_for_loop_logging(self):
        # ct_training_loop.py:257 logs loss_fn.ratio; the contract must hold.
        loss_fn = make_loss('sigmoid', q=2.0)
        loss_fn.update_schedule(3)
        self.assertEqual(loss_fn.ratio, 1 - 1 / 2.0 ** 4)

    def test_loss_fn_pickles_with_schedule(self):
        # Training snapshots pickle loss_fn (ct_training_loop.py:340); the
        # schedule attribute must round-trip through persistence.
        loss_fn = make_loss('adaptive_v1', q=2.0)
        loss_fn.update_schedule(2)
        clone = pickle.loads(pickle.dumps(loss_fn))
        t = sample_t()
        self.assertTrue(torch.equal(clone.schedule.compute_r(t=t, stage=clone.stage),
                                    loss_fn.schedule.compute_r(t=t, stage=loss_fn.stage)))


class TinyNet(torch.nn.Module):
    def forward(self, x, t, labels=None, augment_labels=None):
        return x / (1 + t)


@unittest.skipUnless(torch.cuda.is_available(), 'ECMLoss.__call__ saves/restores CUDA RNG state (official code, unmodified)')
class ECMLossCallCudaTest(unittest.TestCase):
    """End-to-end __call__ checks; run on the A100 (skipped on cpu-only)."""

    def full_loss(self, adj, stage, seed=0):
        loss_fn = make_loss(adj, q=2.0, k=8.0, b=1.0)
        loss_fn.update_schedule(stage)
        net = TinyNet().cuda()
        torch.manual_seed(seed)
        images = torch.randn([8, 3, 8, 8], device='cuda')
        torch.manual_seed(seed)  # re-seed: identical t/eps draws across calls
        return loss_fn(net=net, images=images)

    def test_call_sigmoid_equals_adaptive_v1_at_integer_stage(self):
        self.assertTrue(torch.equal(self.full_loss('sigmoid', stage=1),
                                    self.full_loss('adaptive_v1', stage=1)))

    def test_call_adaptive_v1_fractional_stage_changes_loss(self):
        self.assertFalse(torch.equal(self.full_loss('adaptive_v1', stage=1),
                                     self.full_loss('adaptive_v1', stage=1.5)))


if __name__ == '__main__':
    unittest.main()
