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
                 adaptive_min_gap=1e-3):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        # t -> r entry point, dispatched through training/schedules.py.
        # 'const' / 'sigmoid' are the official fixed formulas (bit-identical
        # to the reference methods below); 'adaptive_v1' is the Role C
        # experiment.
        schedule_kwargs = dict(q=q, k=k, b=b)
        if adj == 'adaptive_v1':
            schedule_kwargs.update(
                loss_ema_beta=adaptive_loss_ema_beta,
                max_adjust=adaptive_max_adjust,
                min_gap=adaptive_min_gap,
            )
        self.schedule = get_schedule(adj, **schedule_kwargs)

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def update_training_signal(self, loss):
        return self.schedule.update_training_signal(loss)

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
