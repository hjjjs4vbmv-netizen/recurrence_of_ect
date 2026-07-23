import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

from training.schedules import get_schedule

#----------------------------------------------------------------------------
# Loss function proposed in the blog "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0,
                 adj='sigmoid', adaptive_loss_ema_beta=0.9, adaptive_max_adjust=0.05,
                 adaptive_min_gap=1e-3, adaptive_warmup_updates=None,
                 adaptive_fast_beta=0.80, adaptive_slow_beta=0.98,
                 adaptive_eps=1e-8):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        # t -> r entry point, dispatched through training/schedules.py.
        # 'const' / 'sigmoid' are the official fixed formulas (bit-identical
        # to the reference methods below); 'adaptive_v1' is the Role C
        # experiments.
        schedule_kwargs = dict(q=q, k=k, b=b)
        if adj == 'adaptive_v1':
            warmup_updates = 2 if adaptive_warmup_updates is None else adaptive_warmup_updates
            schedule_kwargs.update(
                loss_ema_beta=adaptive_loss_ema_beta,
                max_adjust=adaptive_max_adjust,
                min_gap=adaptive_min_gap,
                warmup_updates=warmup_updates,
            )
        elif adj == 'adaptive_v2_dualema':
            warmup_updates = 8 if adaptive_warmup_updates is None else adaptive_warmup_updates
            schedule_kwargs.update(
                beta_fast=adaptive_fast_beta,
                beta_slow=adaptive_slow_beta,
                max_adjust=adaptive_max_adjust,
                min_gap=adaptive_min_gap,
                warmup_updates=warmup_updates,
                eps=adaptive_eps,
            )
        self.schedule = get_schedule(adj, **schedule_kwargs)

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        self._runtime_r_over_t_mean = float('nan')
        self._runtime_gap_mean = float('nan')
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.schedule.update_schedule(stage)
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def update_training_signal(self, loss):
        return self.schedule.update_training_signal(loss)

    def set_training_iteration(self, iteration):
        setter = getattr(self.schedule, 'set_training_iteration', None)
        if setter is not None:
            setter(iteration)

    def schedule_state_dict(self):
        return {
            'schedule_name': self.schedule.name,
            'stage': self.stage,
            'ratio': self.ratio,
            'schedule': self.schedule.state_dict(),
        }

    def load_schedule_state_dict(self, state):
        saved_name = state.get('schedule_name')
        if saved_name is not None and saved_name != self.schedule.name:
            return False
        self.stage = state.get('stage', self.stage)
        self.ratio = state.get('ratio', self.ratio)
        self.schedule.load_state_dict(state.get('schedule', {}))
        return True

    def schedule_metadata(self):
        metadata = self.schedule.metadata()
        metadata.update(stage=self.stage, ratio=self.ratio)
        return metadata

    def schedule_runtime_metrics(self):
        """Return stable, scalar telemetry without exposing schedule internals."""
        metrics = self.schedule.runtime_metrics()
        r_over_t_mean = float(self._runtime_r_over_t_mean)
        gap_mean = float(self._runtime_gap_mean)
        baseline_rho = metrics.get('baseline_rho')
        adaptive_rho = metrics.get('adaptive_rho')
        baseline_gap = metrics.get('baseline_gap')
        adaptive_gap = metrics.get('adaptive_gap')
        return {
            'loss_ema': metrics['loss_ema'],
            'loss_reference': metrics['loss_reference'],
            'correction': float(metrics['correction']),
            'signal_updates': int(metrics['signal_updates']),
            'adaptive_active': bool(metrics['adaptive_active']),
            'r_over_t_mean': r_over_t_mean,
            'gap_mean': gap_mean,
            'fast_loss_ema': metrics.get('fast_loss_ema'),
            'slow_loss_ema': metrics.get('slow_loss_ema'),
            'raw_error': float(metrics.get('raw_error', 0.0)),
            'adaptive_updates': int(metrics.get('adaptive_updates', 0)),
            'warmup_active': bool(metrics.get('warmup_active', False)),
            'finite_signal': bool(metrics.get('finite_signal', True)),
            'baseline_rho': r_over_t_mean if baseline_rho is None else baseline_rho,
            'adaptive_rho': r_over_t_mean if adaptive_rho is None else adaptive_rho,
            'baseline_gap': gap_mean if baseline_gap is None else baseline_gap,
            'adaptive_gap': gap_mean if adaptive_gap is None else adaptive_gap,
            'lower_bound_hit': bool(metrics.get('lower_bound_hit', False)),
            'upper_bound_hit': bool(metrics.get('upper_bound_hit', False)),
            'first_nonzero_correction_iteration': metrics.get(
                'first_nonzero_correction_iteration'
            ),
            'first_adapted_pair_iteration': metrics.get('first_adapted_pair_iteration'),
            'nonfinite_signal_count': int(metrics.get('nonfinite_signal_count', 0)),
        }

    def _record_schedule_runtime_pair(self, t, r):
        with torch.no_grad():
            valid = torch.isfinite(t) & torch.isfinite(r) & (t > 0)
            denominator = torch.where(valid, t, torch.ones_like(t)).to(torch.float64)
            ratio = torch.where(
                valid, r.to(torch.float64) / denominator, torch.zeros_like(denominator)
            )
            totals = torch.stack([
                ratio.sum(),
                (torch.where(valid, (t - r).to(torch.float64) / denominator,
                             torch.zeros_like(denominator))).sum(),
                valid.sum().to(torch.float64),
            ]).cpu().tolist()
            if totals[2] == 0:
                self._runtime_r_over_t_mean = float('nan')
                self._runtime_gap_mean = float('nan')
                return
            self._runtime_r_over_t_mean = totals[0] / totals[2]
            self._runtime_gap_mean = totals[1] / totals[2]

    # Official fixed t->r formulas, kept verbatim as the parity reference for
    # tests/test_schedules.py; the training path dispatches through
    # self.schedule (see __call__).
    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.schedule.compute_r(t=t, stage=self.stage)
        self._record_schedule_runtime_pair(t=t, r=r)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t, t, labels, augment_labels=augment_labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r, r, labels, augment_labels=augment_labels)
            
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()
