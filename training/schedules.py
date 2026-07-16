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
    'adaptive_v1'  Role C experiment: the sigmoid law evaluated at a
                   continuous (fractional) stage, so Delta_t = t - r anneals
                   smoothly with training progress instead of shrinking in
                   discrete factor-q jumps at stage boundaries.

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
the curriculum stage maintained by the training loop
(stage = cur_tick // double_ticks); only 'adaptive_v1' accepts fractional
stages.
"""

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
    """Progress-adaptive sigmoid mapping (Role C experiment, v1).

    Identical law to 'sigmoid', but `stage` may be fractional: passing
    stage = cur_tick / double_ticks (instead of the official integer
    cur_tick // double_ticks) makes the decay q^-(stage+1) anneal
    continuously with training progress, removing the sudden factor-q
    shrinks of Delta_t (and the resulting loss-scale jumps) at stage
    boundaries. At integer stages it coincides exactly with the official
    sigmoid schedule, pinning its behavior to the baseline at every stage
    start.

    Usable today via ECMLoss(adj='adaptive_v1'); however ct_train.py's
    --mapping choices and the integer stage passed by
    training/ct_training_loop.py (both protected files) are unchanged, so
    enabling it end-to-end in training needs a follow-up PR — see
    continuous_stage() and docs/SCHEDULES.md.
    """

    def compute_r(self, t, stage):
        stage = float(stage)
        if stage < 0:
            raise ValueError(f'stage must be >= 0, got {stage}')
        return super().compute_r(t=t, stage=stage)

def continuous_stage(cur_tick, double_ticks):
    """Fractional training-progress stage consumed by 'adaptive_v1'.

    The official loop advances the curriculum as
    stage = cur_tick // double_ticks; adaptive_v1 uses the un-floored ratio.
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
        stages = [0, 0.5, 1, 2.5, 3, 7] if name == 'adaptive_v1' else [0, 1, 3, 7]
        print(f'--- {name} ---')
        for stage in stages:
            ratio = schedule.compute_r(t=t, stage=stage) / t
            print(f'  stage {stage:>4}: ' + '  '.join(f'{v:.4f}' for v in ratio.tolist()))
