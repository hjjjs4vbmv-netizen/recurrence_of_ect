import os
import csv
import time
import copy
import filecmp
import json
import math
import pickle
import psutil
import shutil
import functools
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import metric_main

# Per-attempted-iteration CSV for paired fixed/adaptive comparisons.
# Schedule telemetry comes exclusively from loss_fn.schedule_runtime_metrics().
_LEGACY_TRAIN_SUMMARY_FIELDS = (
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'loss',
    'grad_scale',
    'step_skipped',
    'schedule',
    'stage',
    'elapsed_sec',
    'peak_vram_gb',
)

# The telemetry schema predating next_loop_cur_tick. Keep this exact tuple so
# resumed runs can be migrated without guessing historical tick state.
_PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS = (
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'loss',
    'grad_scale',
    'step_skipped',
    'schedule',
    'stage',
    'loss_ema',
    'loss_reference',
    'correction',
    'signal_updates',
    'adaptive_active',
    'r_over_t_mean',
    'gap_mean',
    'elapsed_sec',
    'peak_vram_gb',
)

_TRAIN_SUMMARY_FIELDS = (
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'loss',
    'grad_scale',
    'step_skipped',
    'schedule',
    'stage',
    # The state that will be used by the next loop iteration. At a
    # maintenance boundary this is also the cur_tick persisted in a checkpoint.
    'next_loop_cur_tick',
    'loss_ema',
    'loss_reference',
    'correction',
    'signal_updates',
    'adaptive_active',
    'r_over_t_mean',
    'gap_mean',
    'elapsed_sec',
    'peak_vram_gb',
)

#----------------------------------------------------------------------------

def load_and_migrate_train_summary(summary_path):
    """Load a resume CSV, upgrading only known historical schemas.

    Values absent from the original schema cannot be reconstructed, so their
    migrated cells deliberately stay empty. The original file is retained
    beside the upgraded CSV for auditability.
    """
    with open(summary_path, 'rt', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)

    if not rows:
        raise RuntimeError(f'resume requested but {summary_path} has no data rows')
    if fieldnames == _TRAIN_SUMMARY_FIELDS:
        return rows, None
    if fieldnames == _LEGACY_TRAIN_SUMMARY_FIELDS:
        backup_path = f'{summary_path}.pre-telemetry.bak'
    elif fieldnames == _PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS:
        backup_path = f'{summary_path}.pre-next-loop-tick.bak'
    else:
        raise RuntimeError(
            f'resume requested but {summary_path} has an unsupported schema; '
            'expected the current schema or an exact supported legacy schema'
        )

    if os.path.exists(backup_path):
        if not filecmp.cmp(summary_path, backup_path, shallow=False):
            raise RuntimeError(
                f'refuse to overwrite non-matching train-summary backup: {backup_path}'
            )
    else:
        shutil.copy2(summary_path, backup_path)

    migrated_rows = [
        {field: row.get(field, '') for field in _TRAIN_SUMMARY_FIELDS}
        for row in rows
    ]
    temporary_path = f'{summary_path}.telemetry-migration.tmp-{os.getpid()}'
    try:
        with open(temporary_path, 'wt', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=_TRAIN_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(migrated_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, summary_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return migrated_rows, backup_path

#----------------------------------------------------------------------------

def adaptive_update_interval_nimg(update_kimg):
    """Convert an adaptive update period to an exact image-count interval."""
    update_kimg = float(update_kimg)
    update_nimg = update_kimg * 1000
    if not math.isfinite(update_kimg) or update_kimg <= 0 or not update_nimg.is_integer():
        raise ValueError(
            f'adaptive_update_kimg must be positive and represent whole images, got {update_kimg}'
        )
    return int(update_nimg)


class AdaptiveSignalWindow:
    """Accumulate local loss until the next absolute adaptive-update boundary.

    Windows are deliberately independent of maintenance ticks so changing
    --tick does not change the controller's update frequency.
    """

    def __init__(self, update_kimg, start_nimg=0):
        self.update_nimg = adaptive_update_interval_nimg(update_kimg)
        start_nimg = int(start_nimg)
        if start_nimg < 0:
            raise ValueError(f'start_nimg must be non-negative, got {start_nimg}')
        self.next_update_nimg = (start_nimg // self.update_nimg + 1) * self.update_nimg
        self.loss_sum = 0.0
        self.loss_count = 0

    def add(self, loss_sum, loss_count):
        self.loss_sum += float(loss_sum)
        self.loss_count += int(loss_count)

    def pop_if_due(self, cur_nimg):
        cur_nimg = int(cur_nimg)
        if cur_nimg < self.next_update_nimg:
            return None
        loss_sum, loss_count = self.loss_sum, self.loss_count
        self.loss_sum = 0.0
        self.loss_count = 0
        while self.next_update_nimg <= cur_nimg:
            self.next_update_nimg += self.update_nimg
        return loss_sum, loss_count

    def state_dict(self):
        """Return all state needed to resume a partially accumulated window."""
        return {
            'update_nimg': self.update_nimg,
            'next_update_nimg': self.next_update_nimg,
            'loss_sum': self.loss_sum,
            'loss_count': self.loss_count,
        }

    def load_state_dict(self, state):
        """Restore a window checkpointed after an arbitrary training step."""
        if not isinstance(state, dict):
            raise ValueError('adaptive signal window state must be a dict')
        required = ('update_nimg', 'next_update_nimg', 'loss_sum', 'loss_count')
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(
                f'adaptive signal window state missing required fields: {", ".join(missing)}'
            )
        update_nimg = int(state['update_nimg'])
        next_update_nimg = int(state['next_update_nimg'])
        loss_sum = float(state['loss_sum'])
        loss_count = int(state['loss_count'])
        if update_nimg != self.update_nimg:
            raise ValueError(
                f'adaptive signal window interval mismatch: checkpoint={update_nimg}, '
                f'current={self.update_nimg}'
            )
        if next_update_nimg <= 0 or next_update_nimg % self.update_nimg != 0:
            raise ValueError(
                f'invalid adaptive signal window next_update_nimg: {next_update_nimg}'
            )
        if loss_count < 0:
            raise ValueError(f'adaptive signal window loss_count must be non-negative, got {loss_count}')
        self.next_update_nimg = next_update_nimg
        self.loss_sum = loss_sum
        self.loss_count = loss_count


def gather_adaptive_signal_window_state(window, device):
    """Collect each rank's local adaptive-window state for a rank-0 checkpoint."""
    local_state = window.state_dict()
    world_size = dist.get_world_size()
    if world_size == 1:
        rank_states = [local_state]
    else:
        local_values = torch.tensor(
            [window.next_update_nimg, window.loss_sum, window.loss_count],
            dtype=torch.float64,
            device=device,
        )
        gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_values, local_values)
        rank_states = [
            {
                'update_nimg': window.update_nimg,
                'next_update_nimg': int(values[0]),
                'loss_sum': float(values[1]),
                'loss_count': int(values[2]),
            }
            for values in gathered_values
        ]

    # Keep the rank-0 fields at the top level for transparent single-rank
    # inspection, and retain every local accumulator for exact DDP resumes.
    return {**rank_states[0], 'rank_states': rank_states}


def local_adaptive_signal_window_state(state):
    """Select this rank's window state from a training-state checkpoint."""
    if not isinstance(state, dict):
        return state
    rank_states = state.get('rank_states')
    if rank_states is None:
        return state
    if not isinstance(rank_states, list) or len(rank_states) != dist.get_world_size():
        raise ValueError(
            'adaptive signal window checkpoint rank count does not match the current world size'
        )
    return rank_states[dist.get_rank()]


def globally_average_adaptive_loss(loss_sum, loss_count, device):
    """Return the sample-weighted loss mean, identical on every rank."""
    totals = torch.tensor([loss_sum, loss_count], dtype=torch.float64, device=device)
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(totals)
    total_count = float(totals[1])
    return float(totals[0] / total_count) if total_count > 0 else float('nan')


def globally_average_runtime_pairs(metric_batches, device):
    """Average public r/t telemetry across accumulation rounds and ranks."""
    r_values = [float(metrics['r_over_t_mean']) for metrics in metric_batches]
    gap_values = [float(metrics['gap_mean']) for metrics in metric_batches]
    r_values = [value for value in r_values if math.isfinite(value)]
    gap_values = [value for value in gap_values if math.isfinite(value)]
    totals = torch.tensor(
        [sum(r_values), len(r_values), sum(gap_values), len(gap_values)],
        dtype=torch.float64,
        device=device,
    )
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(totals)
    r_count = float(totals[1])
    gap_count = float(totals[3])
    return {
        'r_over_t_mean': float(totals[0] / r_count) if r_count > 0 else float('nan'),
        'gap_mean': float(totals[2] / gap_count) if gap_count > 0 else float('nan'),
    }


#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 16)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 16)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
    
#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

@torch.no_grad()
def generator_fn(
    net, latents, class_labels=None, 
    t_max=80, mid_t=None
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float64, device=latents.device)

    # t_0 = T, t_N = 0
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    return x

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    max_steps           = None,     # Optional exact attempted-iteration cap for diagnostics.
    ema_beta            = 0.9999,   # EMA decay rate. Overwritten by ema_halflife_kimg.
    ema_halflife_kimg   = None,     # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = None,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 0,        # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 500,      # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    ckpt_ticks          = 100,      # How often to save latest checkpoints, None = disable.
    sample_ticks        = 50,       # How often to sample images, None = disable.
    eval_ticks          = 500,      # How often to evaluate models, None = disable.
    double_ticks        = 500,      # How often to evaluate models, None = disable.
    adaptive_update_kimg = 0.5,     # Adaptive loss-EMA signal period, independent of ticks.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_tick         = 0,        # Start from the given training progress.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    enable_tf32         = False,    # Enable tf32 for A100/H100 GPUs?
    enable_amp          = False,    # Enable torch.cuda.amp.GradScaler
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Enable these to speed up on A100 GPUs
    dist.print0(f'Enable tf32: {enable_tf32}')
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enable_tf32

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    
    # Automatic Mixed Precision
    dist.print0(f'GradScaler enabled: {enable_amp} for mixed precision training')
    if enable_amp:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        dist.print0('Setting up GradScaler...')
        scaler = torch.cuda.amp.GradScaler()
        dist.print0('Loss scaling is overwritten when GradScaler is enabled')

    dist.print0('Setting up DDP...')
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    
    # Stats
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    attempted_iteration = 0
    successful_optimizer_steps = 0
    resumed_cur_nimg = None
    resumed_cur_tick = None
    resumed_tick_start_nimg = None
    resumed_adaptive_signal_window_state = None
    elapsed_base_sec = 0.0
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        if 'cur_nimg' not in data:
            raise RuntimeError(
                f'resume training-state missing cur_nimg: {resume_state_dump}; '
                f'refuse filename-derived progress fallback for paired runs'
            )
        attempted_iteration = int(data.get('attempted_iteration', 0))
        successful_optimizer_steps = int(data.get('successful_optimizer_steps', 0))
        resumed_cur_nimg = int(data['cur_nimg'])
        if 'cur_tick' in data:
            resumed_cur_tick = int(data['cur_tick'])
        if 'tick_start_nimg' in data:
            resumed_tick_start_nimg = int(data['tick_start_nimg'])
        elapsed_base_sec = float(data.get('elapsed_sec', 0.0))
        if hasattr(loss_fn, 'load_schedule_state_dict') and 'loss_fn_state' in data:
            loss_fn.load_schedule_state_dict(data['loss_fn_state'])
        if 'adaptive_signal_window_state' in data:
            resumed_adaptive_signal_window_state = data['adaptive_signal_window_state']
        if enable_amp:
            if 'gradscaler_state' in data:
                # NOTE(aiihn): Although not loading the state_dict of the GradScaler works well,
                # loading it can improve reproducibility.
                dist.print0(f'Loading GradScaler state from "{resume_state_dump}"...')
                scaler.load_state_dict(data['gradscaler_state'])
            else:
                dist.print0(f'GradScaler state is not found in "{resume_state_dump}", using the default state.')
        del data # conserve memory
    
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], ema.img_channels, ema.img_resolution, ema.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
        
        images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'model_init.png'), drange=[-1,1], grid_size=grid_size)
        del images

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    # Prefer exact progress from training-state; filename-derived resume_tick is only a fallback.
    if resumed_cur_nimg is not None:
        cur_nimg = resumed_cur_nimg
    else:
        cur_nimg = resume_tick * kimg_per_tick * 1000
    if resumed_cur_tick is not None:
        cur_tick = resumed_cur_tick
    else:
        cur_tick = resume_tick
    if resumed_tick_start_nimg is not None:
        tick_start_nimg = resumed_tick_start_nimg
    else:
        tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg / 1000, total_kimg)
    stats_jsonl = None
    train_summary_csv = None
    train_summary_writer = None
    schedule_name = getattr(getattr(loss_fn, 'schedule', None), 'name', None)
    if schedule_name is None:
        schedule_name = getattr(loss_fn, 'adj', None)
    if schedule_name is None:
        schedule_name = loss_kwargs.get('adj', 'unknown')
    adaptive_signal_window = (
        AdaptiveSignalWindow(adaptive_update_kimg, start_nimg=cur_nimg)
        if schedule_name in ('adaptive_v1', 'adaptive_variance_v1') else None
    )
    if adaptive_signal_window is not None and resume_state_dump:
        if resumed_adaptive_signal_window_state is None:
            raise RuntimeError(
                f'resume training-state missing adaptive_signal_window_state: {resume_state_dump}; '
                'cannot exactly resume adaptive loss aggregation'
            )
        adaptive_signal_window.load_state_dict(
            local_adaptive_signal_window_state(resumed_adaptive_signal_window_state)
        )
        if adaptive_signal_window.next_update_nimg <= cur_nimg:
            raise RuntimeError(
                'resumed adaptive signal window is due before or at the restored progress: '
                f'{adaptive_signal_window.next_update_nimg} <= {cur_nimg}'
            )

    if dist.get_rank() == 0:
        summary_path = os.path.join(run_dir, 'train_summary.csv')
        summary_exists = os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
        if resume_state_dump:
            if summary_exists:
                rows, migrated_backup = load_and_migrate_train_summary(summary_path)
                if migrated_backup is not None:
                    dist.print0(
                        f'Migrated legacy train_summary.csv to telemetry schema; '
                        f'original saved as "{migrated_backup}"'
                    )
                last = rows[-1]
                last_attempted = int(float(last['attempted_iteration']))
                last_nimg = int(float(last.get('processed_nimg', last.get('nimg', -1))))
                last_schedule = str(last.get('schedule', '')).strip()
                if last_schedule and last_schedule != str(schedule_name):
                    raise RuntimeError(
                        f'train_summary.csv schedule={last_schedule!r} does not match '
                        f'current schedule={schedule_name!r}; refuse mixed-schedule resume'
                    )
                if attempted_iteration and last_attempted != attempted_iteration:
                    raise RuntimeError(
                        f'train_summary.csv last attempted_iteration={last_attempted} '
                        f'does not match training-state attempted_iteration={attempted_iteration}'
                    )
                if last_nimg >= 0 and last_nimg != cur_nimg:
                    raise RuntimeError(
                        f'train_summary.csv last processed_nimg={last_nimg} '
                        f'does not match resumed cur_nimg={cur_nimg}'
                    )
                last_next_loop_tick = str(last.get('next_loop_cur_tick', '')).strip()
                if last_next_loop_tick:
                    try:
                        parsed_next_loop_tick = float(last_next_loop_tick)
                    except ValueError as exc:
                        raise RuntimeError(
                            'train_summary.csv last next_loop_cur_tick must be numeric: '
                            f'{last_next_loop_tick!r}'
                        ) from exc
                    if (
                        not math.isfinite(parsed_next_loop_tick)
                        or not parsed_next_loop_tick.is_integer()
                        or parsed_next_loop_tick < 0
                    ):
                        raise RuntimeError(
                            'train_summary.csv last next_loop_cur_tick must be a '
                            f'non-negative integer: {last_next_loop_tick!r}'
                        )
                    if int(parsed_next_loop_tick) != cur_tick:
                        raise RuntimeError(
                            f'train_summary.csv last next_loop_cur_tick={last_next_loop_tick} '
                            f'does not match resumed cur_tick={cur_tick}'
                        )
                if not attempted_iteration:
                    attempted_iteration = last_attempted
                    successful_optimizer_steps = int(float(
                        last.get('successful_optimizer_steps', last_attempted)
                    ))
            train_summary_csv = open(summary_path, 'at', newline='')
            train_summary_writer = csv.DictWriter(train_summary_csv, fieldnames=_TRAIN_SUMMARY_FIELDS)
            if not summary_exists:
                train_summary_writer.writeheader()
                train_summary_csv.flush()
        else:
            if summary_exists:
                raise RuntimeError(
                    f'fresh run refuses to append existing train_summary.csv: {summary_path}; '
                    f'pass --resume for a legal continuation or use an empty outdir'
                )
            train_summary_csv = open(summary_path, 'wt', newline='')
            train_summary_writer = csv.DictWriter(train_summary_csv, fieldnames=_TRAIN_SUMMARY_FIELDS)
            train_summary_writer.writeheader()
            train_summary_csv.flush()

    # Prepare for the mapping fn p(r|t).
    dist.print0(f'Reduce dt every {double_ticks} ticks.')
    
    def update_scheduler(loss_fn):
        loss_fn.update_schedule(stage)
        dist.print0(f'Update scheduler at {cur_tick} ticks, {cur_nimg / 1e3} kimg, ratio {loss_fn.ratio}')

    def build_training_state(adaptive_signal_window_state=None):
        # Checkpointing happens during maintenance, before the loop advances:
        #   cur_tick += 1
        #   tick_start_nimg = cur_nimg
        # Persist the *next-loop* values so resume matches uninterrupted training.
        data = dict(
            net=net,
            optimizer_state=optimizer.state_dict(),
            attempted_iteration=attempted_iteration,
            successful_optimizer_steps=successful_optimizer_steps,
            cur_nimg=cur_nimg,
            cur_tick=cur_tick + 1,
            tick_start_nimg=cur_nimg,
            # Match the final CSV row exactly; resume timing continues from
            # the last completed attempted iteration rather than from later
            # checkpoint I/O and maintenance work.
            elapsed_sec=elapsed_sec,
        )
        if hasattr(loss_fn, 'schedule_state_dict'):
            data['loss_fn_state'] = loss_fn.schedule_state_dict()
        if adaptive_signal_window is not None:
            if adaptive_signal_window_state is None:
                raise RuntimeError('adaptive signal window state was not collected for checkpointing')
            data['adaptive_signal_window_state'] = adaptive_signal_window_state
        if enable_amp:
            data['gradscaler_state'] = scaler.state_dict()
        return data
        
    stage = cur_tick // double_ticks
    update_scheduler(loss_fn)

    # Already at/past the requested budget (e.g. resume with same duration): do not
    # execute an extra optimizer step before noticing done.
    if cur_nimg >= total_kimg * 1000:
        dist.print0(f'Already reached training budget at {cur_nimg / 1e3:.3f} kimg; exiting.')
        if train_summary_csv is not None:
            train_summary_csv.close()
        dist.print0()
        dist.print0('Exiting...')
        return

    if max_steps is not None and attempted_iteration >= max_steps:
        dist.print0(f'Already reached diagnostic step limit at {attempted_iteration} steps; exiting.')
        if train_summary_csv is not None:
            train_summary_csv.close()
        dist.print0()
        dist.print0('Exiting...')
        return

    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        loss_batches = []
        schedule_metric_batches = []
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                loss_batches.append(loss.detach())
                schedule_metric_batches.append(loss_fn.schedule_runtime_metrics())
                training_stats.report('Loss/loss', loss)
                if enable_amp:
                    scaler.scale(loss.mean()).backward()
                else:
                    loss.mul(loss_scaling).mean().backward()

        # Unscale first so GradScaler can detect non-finite gradients before
        # they are sanitized below. scaler.step() will still skip the update
        # when unscale_() records an overflow.
        if enable_amp:
            scaler.unscale_(optimizer)

        # NOTE(aiihn & Gsunshine): This should be further tested for AMP.
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # LR scheduler (if needed in the future)
        # for g in optimizer.param_groups:
        #     g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)

        # Update weights. Record GradScaler scale / skip for train_summary.csv.
        # scale_before is the scale applied to this step; a drop after update()
        # means overflow was detected and optimizer.step was skipped.
        grad_scale = float(loss_scaling)
        step_skipped = 0
        if enable_amp:
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            grad_scale = scale_before
            step_skipped = int(scale_after < scale_before)
        else:
            optimizer.step()

        attempted_iteration += 1
        if not step_skipped:
            successful_optimizer_steps += 1

        loss_count = sum(x.numel() for x in loss_batches)
        loss_sum = sum(float(x.sum().cpu()) for x in loss_batches)
        loss_mean = loss_sum / loss_count
        runtime_pair_metrics = globally_average_runtime_pairs(schedule_metric_batches, device=device)
        elapsed_sec = elapsed_base_sec + (time.time() - start_time)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30
        training_stats.report0('Progress/grad_scale', grad_scale)
        training_stats.report0('Progress/step_skipped', step_skipped)
        training_stats.report0('Progress/attempted_iteration', attempted_iteration)
        training_stats.report0('Progress/successful_optimizer_steps', successful_optimizer_steps)
        training_stats.report0('Timing/elapsed_sec', elapsed_sec)
        training_stats.report0('Resources/update_peak_gpu_mem_gb', peak_vram_gb)

        # Update EMA.
        if ema_halflife_kimg is not None:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Advance iteration-local state. Adaptive updates intentionally happen
        # here, before the maintenance early-continue below.
        cur_nimg += batch_size
        if adaptive_signal_window is not None:
            adaptive_signal_window.add(loss_sum, loss_count)
            signal_window = adaptive_signal_window.pop_if_due(cur_nimg)
            if signal_window is not None:
                signal_loss = globally_average_adaptive_loss(*signal_window, device=device)
                loss_fn.update_training_signal(signal_loss)

        schedule_runtime_metrics = loss_fn.schedule_runtime_metrics()
        schedule_runtime_metrics.update(runtime_pair_metrics)
        if schedule_runtime_metrics['loss_ema'] is not None:
            training_stats.report0('Schedule/loss_ema', schedule_runtime_metrics['loss_ema'])
        if schedule_runtime_metrics['loss_reference'] is not None:
            training_stats.report0('Schedule/loss_reference', schedule_runtime_metrics['loss_reference'])
        training_stats.report0('Schedule/correction', schedule_runtime_metrics['correction'])
        training_stats.report0('Schedule/signal_updates', schedule_runtime_metrics['signal_updates'])
        training_stats.report0('Schedule/adaptive_active', int(schedule_runtime_metrics['adaptive_active']))
        training_stats.report0('Schedule/r_over_t_mean', schedule_runtime_metrics['r_over_t_mean'])
        training_stats.report0('Schedule/gap_mean', schedule_runtime_metrics['gap_mean'])

        # Record the exact state that the following loop iteration will see.
        # This cannot be derived reliably from image count: the first iteration
        # always performs maintenance, and completion forces it regardless of
        # --tick. A checkpoint saved below persists this same cur_tick value.
        done = (
            cur_nimg >= total_kimg * 1000
            or (max_steps is not None and attempted_iteration >= max_steps)
        )
        maintenance_due = (
            done
            or cur_tick == 0
            or cur_nimg >= tick_start_nimg + kimg_per_tick * 1000
        )
        next_loop_cur_tick = cur_tick + int(maintenance_due)

        if train_summary_writer is not None:
            train_summary_writer.writerow({
                'attempted_iteration': attempted_iteration,
                'successful_optimizer_steps': successful_optimizer_steps,
                'processed_nimg': cur_nimg,
                'processed_kimg': f'{cur_nimg / 1e3:.6f}',
                'loss': f'{loss_mean:.8f}',
                'grad_scale': f'{grad_scale:.8g}',
                'step_skipped': step_skipped,
                'schedule': schedule_name,
                'stage': stage,
                'next_loop_cur_tick': next_loop_cur_tick,
                'loss_ema': '' if schedule_runtime_metrics['loss_ema'] is None else f"{schedule_runtime_metrics['loss_ema']:.12g}",
                'loss_reference': '' if schedule_runtime_metrics['loss_reference'] is None else f"{schedule_runtime_metrics['loss_reference']:.12g}",
                'correction': f"{schedule_runtime_metrics['correction']:.12g}",
                'signal_updates': schedule_runtime_metrics['signal_updates'],
                'adaptive_active': int(schedule_runtime_metrics['adaptive_active']),
                'r_over_t_mean': f"{schedule_runtime_metrics['r_over_t_mean']:.12g}",
                'gap_mean': f"{schedule_runtime_metrics['gap_mean']:.12g}",
                'elapsed_sec': f'{elapsed_sec:.6f}',
                'peak_vram_gb': f'{peak_vram_gb:.6f}',
            })
            train_summary_csv.flush()

        # Perform maintenance tasks once per tick.
        if not maintenance_due:
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"loss {training_stats.default_collector['Loss/loss']:<9.5f}"]
        fields += [f"grad_scale {grad_scale:<9g}"]
        fields += [f"step_skipped {step_skipped:<7d}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick != 0:
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_tick:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state. Every rank participates in
        # collecting its local adaptive-loss accumulator; rank 0 writes the
        # resulting combined state.
        state_dump_due = (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
        )
        if state_dump_due:
            adaptive_signal_window_state = (
                gather_adaptive_signal_window_state(adaptive_signal_window, device)
                if adaptive_signal_window is not None else None
            )
            if dist.get_rank() == 0:
                torch.save(
                    build_training_state(adaptive_signal_window_state),
                    os.path.join(run_dir, f'training-state-{cur_tick:06d}.pt'),
                )

        # Save latest checkpoints
        latest_checkpoint_due = (
            (ckpt_ticks is not None)
            and (done or cur_tick % ckpt_ticks == 0)
            and cur_tick != 0
        )
        if latest_checkpoint_due:
            dist.print0(f'Save the latest checkpoint at {cur_tick:06d} img...')
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-latest.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

            adaptive_signal_window_state = (
                gather_adaptive_signal_window_state(adaptive_signal_window, device)
                if adaptive_signal_window is not None else None
            )
            if dist.get_rank() == 0:
                torch.save(
                    build_training_state(adaptive_signal_window_state),
                    os.path.join(run_dir, f'training-state-latest.pt'),
                )

        # Sample Img
        if (sample_ticks is not None) and (done or cur_tick % sample_ticks == 0) and dist.get_rank() == 0:
            dist.print0('Exporting sample images...')
            images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, f'{cur_tick:06d}.png'), drange=[-1,1], grid_size=grid_size)
            del images
    
        # Evaluation
        if metrics and (eval_ticks is not None) and (done or cur_tick % eval_ticks == 0) and cur_tick > 0:
            dist.print0('Evaluating models...')
            result_dict = metric_main.calc_metric(metric='fid50k_full', 
                    generator_fn=generator_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')                        
            
            few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
            result_dict = metric_main.calc_metric(metric='two_step_fid50k_full', 
                    generator_fn=few_step_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')                        

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg / 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
        
        # Update Scheduler
        new_stage = (cur_tick-1) // double_ticks
        if new_stage > stage:
            stage = new_stage
            update_scheduler(loss_fn)
    
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'final.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating few-step generation...')
    for _ in range(3):
        for metric in metrics:
            result_dict = metric_main.calc_metric(metric=metric, 
                generator_fn=few_step_fn, G=ema, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl='network-snapshot-latest.pkl')

    # Done.
    if train_summary_csv is not None:
        train_summary_csv.close()
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
