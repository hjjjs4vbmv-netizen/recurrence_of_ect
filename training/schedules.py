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

    Let rho_0 be the official sigmoid r/t ratio, L_0 the first finite loss
    EMA, and L_ema the current loss EMA. The correction is

        delta = max_adjust * tanh(log(L_0) - log(L_ema))

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
        if self.loss_reference is None:
            self.loss_reference = updated_ema
        self.signal_updates += 1
        return True

    def correction(self):
        if (self.max_adjust == 0 or self.loss_ema is None or self.loss_reference is None
                or self.signal_updates <= self.warmup_updates):
            return 0.0
        log_improvement = math.log(self.loss_reference) - math.log(self.loss_ema)
        return self.max_adjust * math.tanh(log_improvement)

    def compute_r(self, t, stage):
        stage = float(stage)
        if not math.isfinite(stage) or stage < 0:
            raise ValueError(f'stage must be finite and >= 0, got {stage}')
        t = _as_tensor(t)
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
        if delta == 0:
            upper = t if self.max_adjust == 0 else t * (1 - self.min_gap)
            return torch.minimum(
                torch.nan_to_num(base_r, nan=0.0, posinf=finite_max, neginf=0.0).clamp_min(0),
                upper,
            )

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
            'loss_ema': self.loss_ema,
            'loss_reference': self.loss_reference,
            'signal_updates': self.signal_updates,
            'correction': self.correction(),
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
