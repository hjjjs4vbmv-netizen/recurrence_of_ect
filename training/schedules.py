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
    'adaptive_v2_dualema'
                   Official sigmoid ratio plus a bounded negative-feedback
                   correction driven by fast/slow loss EMAs.

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
adaptive_v2_dualema likewise changes only r/t and preserves the official
sigmoid baseline and stage curriculum.
"""

import math

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

    def runtime_metrics(self):
        """Stable controller telemetry contract for training/evaluation code."""
        return {
            'loss_ema': None,
            'loss_reference': None,
            'correction': 0.0,
            'signal_updates': 0,
            'adaptive_active': False,
            'fast_loss_ema': None,
            'slow_loss_ema': None,
            'raw_error': 0.0,
            'adaptive_updates': 0,
            'warmup_active': False,
            'finite_signal': True,
            'baseline_rho': None,
            'adaptive_rho': None,
            'baseline_gap': None,
            'adaptive_gap': None,
            'lower_bound_hit': False,
            'upper_bound_hit': False,
            'first_nonzero_correction_iteration': None,
            'first_adapted_pair_iteration': None,
            'nonfinite_signal_count': 0,
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


@register_schedule('adaptive_v2_dualema')
class AdaptiveV2DualEMASchedule(SigmoidSchedule):
    """Dual-EMA trend controller with additive correction to sigmoid r/t.

    The fast EMA ``F`` represents recent loss and the slow EMA ``S`` the
    longer trend.  After warm-up the controller applies

        error = log((F + eps) / (S + eps))
        correction = max_adjust * tanh(error)

    to the official sigmoid ratio.  Thus F > S makes the pair easier by
    increasing r/t, while F < S makes it harder.  Loss updates are performed
    by the training loop only after an optimizer attempt, so an update can
    affect only a later schedule call.
    """

    controller_type = 'dualema_additive_rho'
    controller_version = 1

    def __init__(self, q=2.0, k=8.0, b=1.0, beta_fast=0.80,
                 beta_slow=0.98, max_adjust=0.05, min_gap=1e-3,
                 warmup_updates=8, eps=1e-8):
        super().__init__(q=q, k=k, b=b)
        for name, value in [('q', q), ('k', k), ('b', b),
                            ('beta_fast', beta_fast), ('beta_slow', beta_slow),
                            ('max_adjust', max_adjust), ('min_gap', min_gap),
                            ('eps', eps)]:
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite, got {value}')
        if not 0 < beta_fast < beta_slow < 1:
            raise ValueError(
                'EMA betas must satisfy 0 < beta_fast < beta_slow < 1, '
                f'got beta_fast={beta_fast}, beta_slow={beta_slow}'
            )
        if not 0 <= max_adjust <= 1:
            raise ValueError(f'max_adjust must be in [0, 1], got {max_adjust}')
        if not 0 < min_gap < 1:
            raise ValueError(f'min_gap must be in (0, 1), got {min_gap}')
        if eps <= 0:
            raise ValueError(f'eps must be > 0, got {eps}')
        try:
            normalized_warmup_updates = int(warmup_updates)
        except (TypeError, ValueError, OverflowError):
            normalized_warmup_updates = -1
        if (isinstance(warmup_updates, bool)
                or normalized_warmup_updates != warmup_updates
                or normalized_warmup_updates < 0):
            raise ValueError(
                f'warmup_updates must be a non-negative integer, got {warmup_updates}'
            )

        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.max_adjust = float(max_adjust)
        self.min_gap = float(min_gap)
        self.warmup_updates = normalized_warmup_updates
        self.eps = float(eps)

        self.fast_loss_ema = None
        self.slow_loss_ema = None
        self.signal_updates = 0
        self.adaptive_updates = 0
        self.last_raw_error = 0.0
        self.last_correction = 0.0
        self.last_baseline_rho = None
        self.last_adaptive_rho = None
        self.last_baseline_gap = None
        self.last_adaptive_gap = None
        self.first_nonzero_correction_iteration = None
        self.first_adapted_pair_iteration = None
        self.nonfinite_signal_count = 0
        self.last_finite_signal = True
        self.current_training_iteration = None
        self.lower_bound_hit = False
        self.upper_bound_hit = False

    def set_training_iteration(self, iteration):
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError(f'training iteration must be >= 0, got {iteration}')
        self.current_training_iteration = iteration

    def correction_is_active(self):
        return (
            self.max_adjust != 0
            and self.fast_loss_ema is not None
            and self.slow_loss_ema is not None
            and self.signal_updates >= self.warmup_updates
        )

    def correction(self):
        return self.last_correction if self.correction_is_active() else 0.0

    def update_training_signal(self, loss):
        try:
            loss = float(loss)
        except (TypeError, ValueError, OverflowError):
            loss = float('nan')
        if not math.isfinite(loss) or loss <= 0:
            self.last_finite_signal = False
            self.nonfinite_signal_count += 1
            return False

        if self.fast_loss_ema is None:
            fast = loss
            slow = loss
        else:
            fast = self.beta_fast * self.fast_loss_ema + (1 - self.beta_fast) * loss
            slow = self.beta_slow * self.slow_loss_ema + (1 - self.beta_slow) * loss
        if not math.isfinite(fast) or not math.isfinite(slow) or fast <= 0 or slow <= 0:
            self.last_finite_signal = False
            self.nonfinite_signal_count += 1
            return False

        was_warm = self.signal_updates < self.warmup_updates
        self.fast_loss_ema = fast
        self.slow_loss_ema = slow
        self.signal_updates += 1
        if not was_warm:
            self.adaptive_updates += 1
        self.last_raw_error = math.log((fast + self.eps) / (slow + self.eps))
        self.last_correction = (
            self.max_adjust * math.tanh(self.last_raw_error)
            if self.signal_updates >= self.warmup_updates else 0.0
        )
        self.last_finite_signal = True
        return True

    @staticmethod
    def _pair_statistics(t, base_r, adaptive_r, lower_hits, upper_hits):
        with torch.no_grad():
            valid = torch.isfinite(t) & torch.isfinite(base_r) & torch.isfinite(adaptive_r) & (t > 0)
            denominator = torch.where(valid, t, torch.ones_like(t)).to(torch.float64)
            baseline_ratio = torch.where(
                valid, base_r.to(torch.float64) / denominator, torch.zeros_like(denominator)
            )
            adaptive_ratio = torch.where(
                valid, adaptive_r.to(torch.float64) / denominator, torch.zeros_like(denominator)
            )
            totals = torch.stack([
                baseline_ratio.sum(),
                adaptive_ratio.sum(),
                valid.sum().to(torch.float64),
                (adaptive_r != base_r).sum().to(torch.float64),
                lower_hits.sum().to(torch.float64),
                upper_hits.sum().to(torch.float64),
            ]).cpu().tolist()
            if totals[2] == 0:
                return None, None, None, None
            baseline_rho = totals[0] / totals[2]
            adaptive_rho = totals[1] / totals[2]
            return (
                baseline_rho,
                adaptive_rho,
                1 - baseline_rho,
                1 - adaptive_rho,
                bool(totals[3]),
                bool(totals[4]),
                bool(totals[5]),
            )

    def compute_r(self, t, stage):
        stage = float(stage)
        if not math.isfinite(stage) or stage < 0:
            raise ValueError(f'stage must be finite and >= 0, got {stage}')
        t = _as_tensor(t)
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        finite_max = torch.finfo(t.dtype).max
        safe_t = torch.nan_to_num(t, nan=0.0, posinf=finite_max, neginf=0.0).clamp_min(0)
        try:
            base_r = super().compute_r(t=safe_t, stage=stage)
        except OverflowError:
            base_r = safe_t

        lower_hits = torch.zeros_like(safe_t, dtype=torch.bool)
        upper_hits = torch.zeros_like(safe_t, dtype=torch.bool)
        if not self.correction_is_active() or self.correction() == 0:
            adaptive_r = base_r
        else:
            base_ratio = torch.where(safe_t > 0, base_r / safe_t, torch.zeros_like(safe_t))
            proposed_ratio = base_ratio + self.correction()
            lower_hits = proposed_ratio < 0
            upper_hits = proposed_ratio > 1 - self.min_gap
            ratio = torch.clamp(proposed_ratio, min=0, max=1 - self.min_gap)
            adaptive_r = torch.nan_to_num(
                safe_t * ratio, nan=0.0, posinf=finite_max, neginf=0.0
            )
            adaptive_r = torch.minimum(adaptive_r.clamp_min(0), safe_t * (1 - self.min_gap))

        pair_statistics = self._pair_statistics(
            safe_t, base_r, adaptive_r, lower_hits, upper_hits
        )
        adapted_pair = False
        if len(pair_statistics) == 4:
            (self.last_baseline_rho, self.last_adaptive_rho,
             self.last_baseline_gap, self.last_adaptive_gap) = pair_statistics
            self.lower_bound_hit = False
            self.upper_bound_hit = False
        else:
            (self.last_baseline_rho, self.last_adaptive_rho,
             self.last_baseline_gap, self.last_adaptive_gap,
             adapted_pair, self.lower_bound_hit, self.upper_bound_hit) = pair_statistics
        iteration = self.current_training_iteration
        if self.correction() != 0 and self.first_nonzero_correction_iteration is None:
            self.first_nonzero_correction_iteration = iteration
        if self.first_adapted_pair_iteration is None and adapted_pair:
            self.first_adapted_pair_iteration = iteration
        return adaptive_r

    def _config(self):
        return {
            'q': self.q,
            'k': self.k,
            'b': self.b,
            'beta_fast': self.beta_fast,
            'beta_slow': self.beta_slow,
            'max_adjust': self.max_adjust,
            'min_gap': self.min_gap,
            'warmup_updates': self.warmup_updates,
            'eps': self.eps,
        }

    def state_dict(self):
        return {
            'controller_type': self.controller_type,
            'controller_version': self.controller_version,
            'fast_loss_ema': self.fast_loss_ema,
            'slow_loss_ema': self.slow_loss_ema,
            'signal_updates': self.signal_updates,
            'adaptive_updates': self.adaptive_updates,
            'warmup_updates': self.warmup_updates,
            'last_raw_error': self.last_raw_error,
            'last_correction': self.last_correction,
            'last_baseline_rho': self.last_baseline_rho,
            'last_adaptive_rho': self.last_adaptive_rho,
            'last_baseline_gap': self.last_baseline_gap,
            'last_adaptive_gap': self.last_adaptive_gap,
            'first_nonzero_correction_iteration': self.first_nonzero_correction_iteration,
            'first_adapted_pair_iteration': self.first_adapted_pair_iteration,
            'nonfinite_signal_count': self.nonfinite_signal_count,
            'last_finite_signal': self.last_finite_signal,
            'current_training_iteration': self.current_training_iteration,
            'lower_bound_hit': self.lower_bound_hit,
            'upper_bound_hit': self.upper_bound_hit,
            'config': self._config(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        controller_type = state.get('controller_type', self.controller_type)
        controller_version = int(state.get('controller_version', self.controller_version))
        if controller_type != self.controller_type or controller_version != self.controller_version:
            raise ValueError(
                f'unsupported controller state {controller_type!r} version {controller_version}'
            )
        saved_config = state.get('config')
        if saved_config is not None:
            for key, expected in self._config().items():
                if key in saved_config and saved_config[key] != expected:
                    raise ValueError(
                        f'adaptive v2 config mismatch for {key}: '
                        f'checkpoint={saved_config[key]}, current={expected}'
                    )

        fast = state.get('fast_loss_ema')
        slow = state.get('slow_loss_ema')
        for name, value in [('fast_loss_ema', fast), ('slow_loss_ema', slow)]:
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')
        for name in ('signal_updates', 'adaptive_updates', 'nonfinite_signal_count'):
            if int(state.get(name, 0)) < 0:
                raise ValueError(f'{name} must be >= 0')

        self.fast_loss_ema = None if fast is None else float(fast)
        self.slow_loss_ema = None if slow is None else float(slow)
        self.signal_updates = int(state.get('signal_updates', 0))
        self.adaptive_updates = int(state.get('adaptive_updates', 0))
        self.last_raw_error = float(state.get('last_raw_error', 0.0))
        self.last_correction = float(state.get('last_correction', 0.0))
        self.last_baseline_rho = state.get('last_baseline_rho')
        self.last_adaptive_rho = state.get('last_adaptive_rho')
        self.last_baseline_gap = state.get('last_baseline_gap')
        self.last_adaptive_gap = state.get('last_adaptive_gap')
        self.first_nonzero_correction_iteration = state.get(
            'first_nonzero_correction_iteration'
        )
        self.first_adapted_pair_iteration = state.get('first_adapted_pair_iteration')
        self.nonfinite_signal_count = int(state.get('nonfinite_signal_count', 0))
        self.last_finite_signal = bool(state.get('last_finite_signal', True))
        self.current_training_iteration = state.get('current_training_iteration')
        self.lower_bound_hit = bool(state.get('lower_bound_hit', False))
        self.upper_bound_hit = bool(state.get('upper_bound_hit', False))

    def runtime_metrics(self):
        return {
            'loss_ema': self.fast_loss_ema,
            'loss_reference': self.slow_loss_ema,
            'correction': self.correction(),
            'signal_updates': self.signal_updates,
            'adaptive_active': self.correction_is_active(),
            'fast_loss_ema': self.fast_loss_ema,
            'slow_loss_ema': self.slow_loss_ema,
            'raw_error': self.last_raw_error,
            'adaptive_updates': self.adaptive_updates,
            'warmup_active': self.signal_updates < self.warmup_updates,
            'finite_signal': self.last_finite_signal,
            'baseline_rho': self.last_baseline_rho,
            'adaptive_rho': self.last_adaptive_rho,
            'baseline_gap': self.last_baseline_gap,
            'adaptive_gap': self.last_adaptive_gap,
            'lower_bound_hit': self.lower_bound_hit,
            'upper_bound_hit': self.upper_bound_hit,
            'first_nonzero_correction_iteration': self.first_nonzero_correction_iteration,
            'first_adapted_pair_iteration': self.first_adapted_pair_iteration,
            'nonfinite_signal_count': self.nonfinite_signal_count,
        }

    def metadata(self):
        return {
            'name': self.name,
            'enabled': True,
            'signal': 'dual_loss_ema_trend',
            'controller_type': self.controller_type,
            'controller_version': self.controller_version,
            **self._config(),
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
