"""t -> r mapping schedules for Easy Consistency Tuning (ECT).

During consistency tuning, every training pair (x_t, x_r) is built from a
noise level t ~ p(t) and a smaller noise level r = r(t, stage) produced by a
mapping schedule ("Consistency Models Made Easy", arXiv 2406.14548, Sec. 3.3
and Appendix A). This module centralizes the t -> r schedules behind a single
interface; ECMLoss in training/loss.py dispatches its t -> r entry through it
as r = self.schedule.compute_r(t=t, stage=self.stage), while the official
reference formulas stay verbatim in training/loss.py as the parity anchor.

Supported schedules:
    'const'        Official ECT constant mapping, Eq. (17).
    'sigmoid'      Official ECT sigmoid mapping, Eq. (18); training default.
    'adaptive_v1'  Official sigmoid ratio plus a bounded correction driven by
                   the EMA of the globally aggregated training loss.

The 'const' and 'sigmoid' formulas are verbatim ports of
ECMLoss.t_to_r_const / ECMLoss.t_to_r_sigmoid in training/loss.py and MUST NOT
be modified: they are the official fixed baseline this project reproduces.
tests/test_schedules.py enforces bitwise parity against training/loss.py.

Usage (both forms are supported):
    from training.schedules import compute_r, get_schedule

    schedule = get_schedule('sigmoid', q=256, k=8, b=1)
    r = schedule.compute_r(t=t, stage=stage)

    r = compute_r(t=t, stage=stage, schedule='sigmoid', q=256, k=8, b=1)

`t` may be a torch tensor of any shape (the training loop uses [N, 1, 1, 1]),
or a python/numpy scalar or array, which is converted via torch.as_tensor();
the result is a tensor of the same shape with r clamped to r >= 0. `stage` is
the official integer curriculum stage maintained by the training loop
(stage = cur_tick // double_ticks). adaptive_v1 changes only r/t using the
loss EMA; it does not replace the official stage curriculum.
"""

import math
from statistics import NormalDist

import torch

#----------------------------------------------------------------------------
# Registry.

_SCHEDULES = {}

def register_schedule(name):
    def decorator(cls):
        cls.name = name
        _SCHEDULES[name] = cls
        return cls
    return decorator

def available_schedules():
    return sorted(_SCHEDULES)

def get_schedule(schedule, **schedule_kwargs):
    # training/loss.py imports this by name, and torch_utils.persistence
    # embeds that module's source into training snapshots — keep the public
    # names in this module stable or old snapshots stop unpickling.
    if schedule not in _SCHEDULES:
        raise ValueError(f"Unknown schedule type {schedule!r}! Available: {', '.join(available_schedules())}")
    return _SCHEDULES[schedule](**schedule_kwargs)

#----------------------------------------------------------------------------
# Interface. Hyperparameter defaults follow ct_train.py (-q 2.0 -k 8.0 -b 1.0).

class Schedule:
    name = None

    def __init__(self, q=2.0, k=8.0, b=1.0):
        if q <= 1:
            raise ValueError(f'q must be > 1 (Delta_t decay factor), got {q}')
        self.q = q
        self.k = k
        self.b = b
        self.stage = 0

    def compute_r(self, t, stage):
        raise NotImplementedError

    # Stateful interface mirroring ECMLoss, so a Schedule instance can drive
    # the existing training loop (update_schedule() at stage boundaries,
    # t_to_r() inside the loss) without further changes.
    def update_schedule(self, stage):
        self.stage = stage

    def t_to_r(self, t):
        return self.compute_r(t=t, stage=self.stage)

    def update_training_signal(self, loss):
        del loss
        return False

    def observe_training_batch(self, loss, t):
        """Observe per-sample training data for schedules that need it."""
        del loss, t

    def runtime_metrics(self):
        """Stable controller telemetry contract for training/evaluation code."""
        return {
            'loss_ema': None,
            'loss_reference': None,
            'correction': 0.0,
            'signal_updates': 0,
            'adaptive_active': False,
        }

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError(f'{type(self).__name__} does not have adaptive state')

    def metadata(self):
        return {
            'name': self.name,
            'enabled': False,
            'q': self.q,
            'k': self.k,
            'b': self.b,
        }

    def __repr__(self):
        return f'{type(self).__name__}(q={self.q}, k={self.k}, b={self.b})'

def _as_tensor(t):
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)

#----------------------------------------------------------------------------
# Official fixed schedules. Verbatim ports of training/loss.py — do not edit
# the formulas; tests/test_schedules.py checks them bit-for-bit against
# ECMLoss.

@register_schedule('const')
class ConstSchedule(Schedule):
    """Official constant mapping, Eq. (17): r/t = 1 - 1/q^(stage+1).

    Port of ECMLoss.t_to_r_const in training/loss.py.
    """

    def compute_r(self, t, stage):
        t = _as_tensor(t)
        decay = 1 / self.q ** (stage + 1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

@register_schedule('sigmoid')
class SigmoidSchedule(Schedule):
    """Official sigmoid mapping, Eq. (18): r/t = 1 - n(t)/q^(stage+1), where
    n(t) = 1 + k * sigmoid(-b * t). Training default (--mapping=sigmoid).

    Port of ECMLoss.t_to_r_sigmoid in training/loss.py.
    """

    def compute_r(self, t, stage):
        t = _as_tensor(t)
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (stage + 1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

#----------------------------------------------------------------------------
# Experimental schedules (Role C). Changes relative to the official fixed
# schedules live below this line only.

@register_schedule('adaptive_v1')
class AdaptiveV1Schedule(SigmoidSchedule):
    """Loss-EMA adaptive correction on top of the official sigmoid ratio.

    Let rho_0 be the official sigmoid r/t ratio, L_ref the loss EMA at the
    end of warm-up, and L_ema the current loss EMA. The correction is

        delta = max_adjust * tanh(log(L_ref) - log(L_ema))

    and rho = clamp(rho_0 + delta, 0, 1 - min_gap). Improving loss therefore
    tightens the pair (smaller t-r), while worsening loss widens it. The
    correction is deterministic and bounded by max_adjust.
    """

    def __init__(self, q=2.0, k=8.0, b=1.0, loss_ema_beta=0.9,
                 max_adjust=0.05, min_gap=1e-3, warmup_updates=2):
        super().__init__(q=q, k=k, b=b)
        for name, value in [('q', q), ('k', k), ('b', b)]:
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite, got {value}')
        if not math.isfinite(loss_ema_beta) or not 0 <= loss_ema_beta < 1:
            raise ValueError(f'loss_ema_beta must be in [0, 1), got {loss_ema_beta}')
        if not math.isfinite(max_adjust) or not 0 <= max_adjust <= 1:
            raise ValueError(f'max_adjust must be in [0, 1], got {max_adjust}')
        if not math.isfinite(min_gap) or not 0 < min_gap < 1:
            raise ValueError(f'min_gap must be in (0, 1), got {min_gap}')
        try:
            normalized_warmup_updates = int(warmup_updates)
        except (TypeError, ValueError, OverflowError):
            normalized_warmup_updates = -1
        if (isinstance(warmup_updates, bool) or normalized_warmup_updates != warmup_updates
                or normalized_warmup_updates < 0):
            raise ValueError(f'warmup_updates must be a non-negative integer, got {warmup_updates}')
        self.loss_ema_beta = float(loss_ema_beta)
        self.max_adjust = float(max_adjust)
        self.min_gap = float(min_gap)
        self.warmup_updates = normalized_warmup_updates
        self.loss_ema = None
        self.loss_reference = None
        self.signal_updates = 0

    def update_training_signal(self, loss):
        loss = float(loss)
        if not math.isfinite(loss) or loss < 0:
            return False
        loss = max(loss, torch.finfo(torch.float64).tiny)
        if self.loss_ema is None:
            updated_ema = loss
        else:
            beta = self.loss_ema_beta
            updated_ema = beta * self.loss_ema + (1 - beta) * loss
        if not math.isfinite(updated_ema) or updated_ema <= 0:
            return False
        self.loss_ema = updated_ema
        self.signal_updates += 1

        # Establish the baseline only after the requested number of valid
        # signals have contributed to the EMA. With no warm-up, the first
        # signal is necessarily the baseline (and therefore has zero
        # correction); otherwise the following signal is the first one that
        # can produce a correction relative to this reference.
        if self.loss_reference is None and (
            self.warmup_updates == 0 or self.signal_updates == self.warmup_updates
        ):
            self.loss_reference = updated_ema
        return True

    def correction_is_active(self):
        return (
            self.max_adjust != 0
            and self.loss_ema is not None
            and self.loss_reference is not None
            and self.signal_updates > self.warmup_updates
        )

    def correction(self):
        if not self.correction_is_active():
            return 0.0
        log_improvement = math.log(self.loss_reference) - math.log(self.loss_ema)
        return self.max_adjust * math.tanh(log_improvement)

    def compute_r(self, t, stage):
        stage = float(stage)
        if not math.isfinite(stage) or stage < 0:
            raise ValueError(f'stage must be finite and >= 0, got {stage}')
        t = _as_tensor(t)

        # Before a correction is active, adaptive_v1 is exactly the official
        # sigmoid schedule. In particular, min_gap must not alter the no-signal
        # or warmup path.
        if not self.correction_is_active():
            return super().compute_r(t=t, stage=stage)

        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        finite_max = torch.finfo(t.dtype).max
        t = torch.nan_to_num(t, nan=0.0, posinf=finite_max, neginf=0.0).clamp_min(0)

        try:
            base_r = super().compute_r(t=t, stage=stage)
        except OverflowError:
            # q**(stage+1) -> inf, so the mathematical sigmoid ratio -> 1.
            base_r = t
        delta = self.correction()
        base_ratio = torch.where(t > 0, base_r / t, torch.zeros_like(t))
        ratio = torch.clamp(base_ratio + delta, min=0, max=1 - self.min_gap)
        r = torch.nan_to_num(t * ratio, nan=0.0, posinf=finite_max, neginf=0.0)
        return torch.minimum(r.clamp_min(0), t)

    def state_dict(self):
        return {
            'loss_ema': self.loss_ema,
            'loss_reference': self.loss_reference,
            'signal_updates': self.signal_updates,
        }

    def load_state_dict(self, state):
        loss_ema = state.get('loss_ema')
        loss_reference = state.get('loss_reference')
        signal_updates = int(state.get('signal_updates', 0))
        for name, value in [('loss_ema', loss_ema), ('loss_reference', loss_reference)]:
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')
        if signal_updates < 0:
            raise ValueError(f'signal_updates must be >= 0, got {signal_updates}')
        self.loss_ema = None if loss_ema is None else float(loss_ema)
        self.loss_reference = None if loss_reference is None else float(loss_reference)
        self.signal_updates = signal_updates

    def metadata(self):
        return {
            'name': self.name,
            'enabled': True,
            'signal': 'loss_ema',
            'q': self.q,
            'k': self.k,
            'b': self.b,
            'loss_ema_beta': self.loss_ema_beta,
            'warmup_updates': self.warmup_updates,
            'max_adjust': self.max_adjust,
            'min_gap': self.min_gap,
            **self.runtime_metrics(),
        }

    def runtime_metrics(self):
        return {
            'loss_ema': self.loss_ema,
            'loss_reference': self.loss_reference,
            'correction': self.correction(),
            'signal_updates': self.signal_updates,
            'adaptive_active': self.correction_is_active(),
        }

@register_schedule('adaptive_variance_v1')
class AdaptiveVarianceV1Schedule(SigmoidSchedule):
    """Per-noise-bin normalized-loss-variance controller (Idea 5).

    Samples are divided into equal-probability bins under the configured
    log-normal t distribution.  Each bin tracks

        V = Var(loss) / max(E[loss]^2, eps)

    and scales the official sigmoid gap multiplicatively as

        gap = gap_0 / (1 + variance_strength * EMA(V)).

    High-variance regions therefore receive an easier (smaller-gap) pair.
    Before the warm-up completes this schedule is bit-identical to sigmoid.
    """

    def __init__(
        self, q=2.0, k=8.0, b=1.0, variance_ema_beta=0.9,
        variance_strength=1.0, min_gap_scale=0.5, num_bins=4,
        warmup_updates=2, p_mean=-1.1, p_std=2.0,
    ):
        super().__init__(q=q, k=k, b=b)
        if not math.isfinite(variance_ema_beta) or not 0 <= variance_ema_beta < 1:
            raise ValueError(
                f'variance_ema_beta must be in [0, 1), got {variance_ema_beta}'
            )
        if not math.isfinite(variance_strength) or variance_strength < 0:
            raise ValueError(
                f'variance_strength must be finite and >= 0, got {variance_strength}'
            )
        if not math.isfinite(min_gap_scale) or not 0 < min_gap_scale <= 1:
            raise ValueError(
                f'min_gap_scale must be in (0, 1], got {min_gap_scale}'
            )
        if isinstance(num_bins, bool) or int(num_bins) != num_bins or int(num_bins) < 2:
            raise ValueError(f'num_bins must be an integer >= 2, got {num_bins}')
        if (
            isinstance(warmup_updates, bool)
            or int(warmup_updates) != warmup_updates
            or int(warmup_updates) < 0
        ):
            raise ValueError(
                f'warmup_updates must be a non-negative integer, got {warmup_updates}'
            )
        if not math.isfinite(p_mean):
            raise ValueError(f'p_mean must be finite, got {p_mean}')
        if not math.isfinite(p_std) or p_std <= 0:
            raise ValueError(f'p_std must be finite and > 0, got {p_std}')

        self.variance_ema_beta = float(variance_ema_beta)
        self.variance_strength = float(variance_strength)
        self.min_gap_scale = float(min_gap_scale)
        self.num_bins = int(num_bins)
        self.warmup_updates = int(warmup_updates)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        normal = NormalDist()
        self.log_t_bin_edges = [
            self.p_mean + self.p_std * normal.inv_cdf(index / self.num_bins)
            for index in range(1, self.num_bins)
        ]
        self.variance_ema = [None] * self.num_bins
        self.signal_updates = 0
        self._bin_loss_sum = [0.0] * self.num_bins
        self._bin_loss_sq_sum = [0.0] * self.num_bins
        self._bin_count = [0.0] * self.num_bins

    def _bin_indices(self, t):
        t = _as_tensor(t)
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        tiny = torch.finfo(t.dtype).tiny
        log_t = t.reshape(-1).clamp_min(tiny).log()
        edges = torch.as_tensor(
            self.log_t_bin_edges, dtype=log_t.dtype, device=log_t.device
        )
        return torch.bucketize(log_t, edges)

    def observe_training_batch(self, loss, t):
        loss = _as_tensor(loss).detach().reshape(-1).to(torch.float64)
        bins = self._bin_indices(t).reshape(-1)
        if loss.numel() != bins.numel():
            raise ValueError(
                f'loss/t sample count mismatch: {loss.numel()} != {bins.numel()}'
            )
        valid = torch.isfinite(loss) & (loss >= 0)
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        bins = bins.to(device=loss.device)
        packed = torch.zeros(
            [3, self.num_bins], dtype=torch.float64, device=loss.device
        )
        packed[0].scatter_add_(0, bins, loss)
        packed[1].scatter_add_(0, bins, loss.square())
        packed[2].scatter_add_(0, bins, valid.to(torch.float64))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed)
        values = packed.cpu()
        for index in range(self.num_bins):
            self._bin_loss_sum[index] += float(values[0, index])
            self._bin_loss_sq_sum[index] += float(values[1, index])
            self._bin_count[index] += float(values[2, index])

    def update_training_signal(self, loss):
        # The scalar is the existing update-window trigger. Per-bin moments
        # were already accumulated by observe_training_batch().
        del loss
        updated_any = False
        tiny = torch.finfo(torch.float64).tiny
        for index in range(self.num_bins):
            count = self._bin_count[index]
            if count <= 0:
                continue
            mean = self._bin_loss_sum[index] / count
            second_moment = self._bin_loss_sq_sum[index] / count
            variance = max(second_moment - mean * mean, 0.0)
            normalized_variance = variance / max(mean * mean, tiny)
            if not math.isfinite(normalized_variance):
                continue
            previous = self.variance_ema[index]
            if previous is None:
                updated = normalized_variance
            else:
                beta = self.variance_ema_beta
                updated = beta * previous + (1 - beta) * normalized_variance
            self.variance_ema[index] = updated
            updated_any = True

        self._bin_loss_sum = [0.0] * self.num_bins
        self._bin_loss_sq_sum = [0.0] * self.num_bins
        self._bin_count = [0.0] * self.num_bins
        if updated_any:
            self.signal_updates += 1
        return updated_any

    def correction_is_active(self):
        return (
            self.variance_strength > 0
            and self.signal_updates > self.warmup_updates
            and any(value is not None for value in self.variance_ema)
        )

    def gap_scales(self):
        if not self.correction_is_active():
            return [1.0] * self.num_bins
        scales = []
        for value in self.variance_ema:
            if value is None:
                scale = 1.0
            else:
                scale = 1 / (1 + self.variance_strength * value)
            scales.append(min(1.0, max(self.min_gap_scale, scale)))
        return scales

    def correction(self):
        scales = self.gap_scales()
        return sum(scale - 1 for scale in scales) / len(scales)

    def compute_r(self, t, stage):
        stage = float(stage)
        if not math.isfinite(stage) or stage < 0:
            raise ValueError(f'stage must be finite and >= 0, got {stage}')
        t = _as_tensor(t)
        if not self.correction_is_active():
            return super().compute_r(t=t, stage=stage)
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        finite_max = torch.finfo(t.dtype).max
        clean_t = torch.nan_to_num(
            t, nan=0.0, posinf=finite_max, neginf=0.0
        ).clamp_min(0)
        try:
            base_r = super().compute_r(t=clean_t, stage=stage)
        except OverflowError:
            base_r = clean_t
        base_ratio = torch.where(
            clean_t > 0, base_r / clean_t, torch.zeros_like(clean_t)
        )
        base_gap = 1 - base_ratio
        bin_indices = self._bin_indices(clean_t)
        scales = torch.as_tensor(
            self.gap_scales(), dtype=clean_t.dtype, device=clean_t.device
        )
        gap = (base_gap.reshape(-1) * scales[bin_indices]).reshape(base_gap.shape)
        gap = gap.clamp(min=torch.finfo(clean_t.dtype).eps, max=1)
        r = clean_t * (1 - gap)
        return torch.minimum(
            torch.nan_to_num(r, nan=0.0, posinf=finite_max, neginf=0.0).clamp_min(0),
            clean_t,
        )

    def state_dict(self):
        return {
            'variance_ema': self.variance_ema,
            'signal_updates': self.signal_updates,
            'bin_loss_sum': self._bin_loss_sum,
            'bin_loss_sq_sum': self._bin_loss_sq_sum,
            'bin_count': self._bin_count,
        }

    def load_state_dict(self, state):
        variance_ema = list(state.get('variance_ema', [None] * self.num_bins))
        if len(variance_ema) != self.num_bins:
            raise ValueError('variance_ema bin count does not match configuration')
        for value in variance_ema:
            if value is not None and (not math.isfinite(float(value)) or value < 0):
                raise ValueError(f'variance EMA must be finite and >= 0, got {value}')
        self.variance_ema = [
            None if value is None else float(value) for value in variance_ema
        ]
        self.signal_updates = int(state.get('signal_updates', 0))
        if self.signal_updates < 0:
            raise ValueError('signal_updates must be >= 0')
        for attribute, key in [
            ('_bin_loss_sum', 'bin_loss_sum'),
            ('_bin_loss_sq_sum', 'bin_loss_sq_sum'),
            ('_bin_count', 'bin_count'),
        ]:
            values = list(state.get(key, [0.0] * self.num_bins))
            if len(values) != self.num_bins:
                raise ValueError(f'{key} bin count does not match configuration')
            setattr(self, attribute, [float(value) for value in values])

    def runtime_metrics(self):
        finite_values = [
            value for value in self.variance_ema if value is not None
        ]
        return {
            'loss_ema': (
                sum(finite_values) / len(finite_values)
                if finite_values else None
            ),
            'loss_reference': None,
            'correction': self.correction(),
            'signal_updates': self.signal_updates,
            'adaptive_active': self.correction_is_active(),
        }

    def metadata(self):
        return {
            'name': self.name,
            'enabled': True,
            'signal': 'per_log_t_bin_normalized_loss_variance',
            'q': self.q,
            'k': self.k,
            'b': self.b,
            'variance_ema_beta': self.variance_ema_beta,
            'variance_strength': self.variance_strength,
            'min_gap_scale': self.min_gap_scale,
            'num_bins': self.num_bins,
            'warmup_updates': self.warmup_updates,
            'p_mean': self.p_mean,
            'p_std': self.p_std,
            'log_t_bin_edges': self.log_t_bin_edges,
            'variance_ema_by_bin': self.variance_ema,
            'gap_scale_by_bin': self.gap_scales(),
            **self.runtime_metrics(),
        }


def continuous_stage(cur_tick, double_ticks):
    """Legacy fractional-stage helper retained for import compatibility.

    adaptive_v1 now uses the official integer stage and adapts only from the
    loss EMA; new training code should not use this helper.
    """
    if double_ticks <= 0:
        raise ValueError(f'double_ticks must be > 0, got {double_ticks}')
    return cur_tick / double_ticks

#----------------------------------------------------------------------------
# Functional one-shot interface.

def compute_r(t, stage, schedule='sigmoid', **schedule_kwargs):
    """r = compute_r(t=t, stage=stage, schedule='sigmoid', q=256, k=8, b=1)"""
    return get_schedule(schedule, **schedule_kwargs).compute_r(t=t, stage=stage)

#----------------------------------------------------------------------------
# Quick visual check: python -m training.schedules

if __name__ == '__main__':
    t = torch.tensor([0.002, 0.02, 0.2, 2.0, 20.0, 80.0], dtype=torch.float64)
    print('r/t with q=2, k=8, b=1 at t =', t.tolist())
    for name in available_schedules():
        schedule = get_schedule(name)
        if name == 'adaptive_v1':
            schedule.update_training_signal(10.0)
            schedule.update_training_signal(7.0)
        print(f'--- {name} ---')
        for stage in [0, 1, 3, 7]:
            ratio = schedule.compute_r(t=t, stage=stage) / t
            print(f'  stage {stage:>4}: ' + '  '.join(f'{v:.4f}' for v in ratio.tolist()))
