# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
from functools import cached_property
import re
import warnings
from inspect import signature
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np

import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *

from torch.nn import Identity, Flatten  # For direct accessibility via command line
from torchvision import transforms  # For direct accessibility via command line
from Blocks.Augmentations import RandomShiftsAug, IntensityAug  # For direct accessibility via command line
from Blocks.Architectures import *  # For direct accessibility via command line


# Sets all Pytorch and Numpy random seeds
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Initializes seeds, device, and CUDA acceleration
def init(args):
    # Set seeds
    set_seeds(args.seed)

    # Set device
    mps = getattr(torch.backends, 'mps', None)  # M1 MacBook speedup

    args.device = args.device or ('cuda' if torch.cuda.is_available()
                                  else 'mps' if mps and mps.is_available() else 'cpu')

    # CUDA speedup via automatic mixed precision
    MP.enable(args)

    # CUDA speedup when input sizes don't vary
    torch.backends.cudnn.benchmark = True

    print('Device:', args.device)


# Format path names
# e.g. "Checkpoints/Agents.DQNAgent" -> "Checkpoints/DQNAgent"
OmegaConf.register_new_resolver("format", lambda name: name.split('.')[-1])

# Allow recipes config to accept objects as args
OmegaConf.register_new_resolver("allow_objects", lambda config: config._set_flag("allow_objects", True))

# A boolean "not" operation for config
OmegaConf.register_new_resolver("not", lambda bool: not bool)


# Saves model + args + selected attributes
def save(path, model, args, *attributes):
    root, name = path.rsplit('/', 1)
    Path(root).mkdir(exist_ok=True, parents=True)
    torch.save(args, Path(root) / f'{name.replace(".pt", "")}.args')  # Save args

    torch.save({'state_dict': model.state_dict(), **{attr: getattr(model, attr)
                                                     for attr in attributes}}, path)  # Save params + attributes
    print(f'Model successfully saved to {path}')


# Loads model or part of model
def load(path, device='cuda', args=None, preserve=(), distributed=False, attr='', **kwargs):
    root, name = path.replace('.pt', '').rsplit('/', 1)

    while True:
        try:
            to_load = torch.load(path, map_location=device)  # Load
            args = args or DictConfig({})
            args.update(torch.load(f'{root}/{name}.args'))  # Load
            break
        except Exception as e:  # Pytorch's load and save are not atomic transactions, can conflict in distributed setup
            if not distributed:
                raise RuntimeError(e)
            warnings.warn(f'Load conflict, resolving...')  # For distributed training

    # Overriding original args where specified
    for key, value in kwargs.items():
        OmegaConf.update(args.recipes if attr else args, attr + f'._override_.{key}' if attr else key, value)

    model = instantiate(args).to(device)

    # Load model's params
    model.load_state_dict(to_load['state_dict'], strict=False)

    # Load saved attributes as well
    for key in to_load:
        if hasattr(model, key) and key not in ['state_dict', *preserve]:
            setattr(model, key, to_load[key])

    # Can also load part of a model. Useful for recipes,
    # e.g. python Run.py Eyes=load +eyes.path=<Path To Agent Checkpoint> +eyes.attr=encoder.Eyes +eyes.device=<device>
    for key in attr.split('.'):
        if key:
            model = getattr(model, key)
    print(f'Successfully loaded {attr if attr else "agent"} from {path}')
    return model


# Simple-sophisticated instantiation of a class or module by various semantics
def instantiate(args, i=0, **kwargs):
    if isinstance(args, (DictConfig, dict)):
        args = DictConfig(args)  # Non-destructive copy

    if hasattr(args, '_override_'):
        kwargs.update(args.pop('_override_'))  # For loading old models with new, overridden args

    while hasattr(args, '_default_'):  # Allow inheritance between shorthands
        args = args['_default_'] if isinstance(args['_default_'], str) else DictConfig(dict(args.pop('_default_'), **args))

    if hasattr(args, '_target_') and args._target_:
        try:
            return hydra.utils.instantiate(args, **kwargs)  # Regular hydra
        except ImportError as e:
            if '(' in args._target_ and ')' in args._target_:  # Direct code execution
                args = args._target_
            else:
                if 'Utils.' in args._target_:
                    raise ImportError
                args._target_ = 'Utils.' + args._target_  # Portal into Utils
                try:
                    return instantiate(args, i, **kwargs)
                except ImportError:
                    raise e  # Original error if all that doesn't work
        except TypeError as e:
            kwarg = re.search('got an unexpected keyword argument \'(.+?)\'', str(e))
            if kwarg and kwarg.group(1) not in args:
                kwargs = {key: kwargs[key] for key in kwargs if key != kwarg.group(1)}
                return instantiate(args, i, **kwargs)  # Signature matching, only for kwargs not args
            raise e  # Original error

    if isinstance(args, str):
        for key in kwargs:
            args = args.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
        args = eval(args)  # Direct code execution

    # Signature matching
    if isinstance(args, type):
        _args = signature(args).parameters
        args = args(**kwargs if 'kwargs' in _args else {key: kwargs[key] for key in kwargs.keys() & _args})

    return None if hasattr(args, '_target_') \
        else args[i] if isinstance(args, (list, nn.ModuleList)) \
        else args  # Additional useful ones


# Initializes model weights a la orthogonal
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# Initializes model optimizer. Default: AdamW
def optimizer_init(params, optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None):
    params = list(params)

    # Optimizer
    optim = len(params) > 0 and (instantiate(optim, params=params, lr=getattr(optim, 'lr', lr)) or lr
                                 and AdamW(params, lr=lr, weight_decay=weight_decay or 0))  # Default

    # Learning rate scheduler
    scheduler = optim and (instantiate(scheduler, optimizer=optim) or lr_decay_epochs
                           and CosineAnnealingLR(optim, lr_decay_epochs))  # Default

    return optim, scheduler


# Copies parameters from a source model to a target model, optionally EMA weighing
def update_ema_target(source, target, ema_decay=0):
    with torch.no_grad():
        for params, target_params in zip(source.state_dict().values(), target.state_dict().values()):
            target_params.copy_((1 - ema_decay) * params + ema_decay * target_params)


# Compute the output shape of a CNN layer
def cnn_layer_feature_shape(*spatial_shape, kernel_size=1, stride=1, padding=0, dilation=1):
    if padding == 'same':
        return spatial_shape
    axes = [size for size in spatial_shape if size]
    if type(kernel_size) is not tuple:
        kernel_size = [kernel_size] * len(axes)
    if type(stride) is not tuple:
        stride = [stride] * len(axes)
    if type(padding) is not tuple:
        padding = [padding] * len(axes)
    if type(dilation) is not tuple:
        dilation = [dilation] * len(axes)
    kernel_size = [min(size, kernel_size[i]) for i, size in enumerate(axes)]
    padding = [min(size, padding[i]) for i, size in enumerate(axes)]  # Assumes adaptive
    out_shape = [math.floor(((size + (2 * padding[i]) - (dilation[i] * (kernel_size[i] - 1)) - 1) / stride[i]) + 1)
                 for i, size in enumerate(axes)] + list(spatial_shape[len(axes):])
    return out_shape


# Compute the output shape of a whole CNN (or other architecture)
def cnn_feature_shape(chw, *blocks, verbose=False):
    channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
    for block in blocks:
        if isinstance(block, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.Conv1d, nn.AvgPool1d, nn.MaxPool1d)):
            channels = block.out_channels if hasattr(block, 'out_channels') else channels
            height, width = cnn_layer_feature_shape(height, width,
                                                    kernel_size=block.kernel_size,
                                                    stride=block.stride,
                                                    padding=block.padding)
        elif isinstance(block, nn.Linear):
            channels = block.out_features  # Assumes channels-last if linear
        elif isinstance(block, nn.Flatten) and (block.start_dim == -3 or block.start_dim == 1):
            channels, height, width = channels * (height or 1) * (width or 1), None, None  # Placeholder height/width
        elif isinstance(block, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
            size = to_tuple(block.output_size)  # Can be int
            pair = size[0] if isinstance(block, nn.AdaptiveAvgPool2d) else None
            height, width = (size[0], pair) if width is None else size + (pair,) * (2 - len(size))
        elif hasattr(block, 'repr_shape'):
            chw = block.repr_shape(*chw)
            channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
        elif hasattr(block, 'modules'):
            for layer in block.children():
                chw = cnn_feature_shape(chw, layer, verbose=verbose)
                channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
        if verbose:
            print(type(block), (channels, height, width))

    feature_shape = tuple(size for size in (channels, height, width) if size is not None)

    return feature_shape


# General-purpose shape pre-computation. Unlike above, uses manual forward pass through model(s).
def repr_shape(input_shape, *blocks):
    for block in blocks:
        input_shape = block(torch.ones(1, *input_shape)).shape[1:]
    return input_shape


# "Ensembles" (stacks) multiple modules' outputs
class Ensemble(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *x, **kwargs):
        return torch.stack([m(*x, **kwargs) for m in self.ensemble],
                           self.dim)

    def __len__(self):
        return len(self.ensemble)


# Replaces tensor's batch items with Normal-sampled random latent
class Rand(nn.Module):
    def __init__(self, size=1, output_shape=None, uniform=False):
        super().__init__()

        self.output_shape = to_tuple(output_shape or size)
        self.uniform = uniform

    def repr_shape(self, *_):
        return self.output_shape

    def forward(self, *x):
        x = torch.randn((x[0].shape[0], *self.output_shape), device=x[0].device)

        if self.uniform:
            x.uniform_()

        return x


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes, null_value=0, one_value=1):
    # assert x.shape[-1] == 1  # Can check this
    x = x.squeeze(-1).unsqueeze(-1)  # Or do this
    x = x.long()
    shape = x.shape[:-1]
    nulls = torch.full([*shape, num_classes], null_value, dtype=torch.float32, device=x.device)
    return nulls.scatter(len(shape), x, one_value).float()


# Differentiable one_hot via "re-parameterization"
def rone_hot(x, null_value=0):
    return x - (x - one_hot(torch.argmax(x, -1, keepdim=True), x.shape[-1]) * (1 - null_value) + null_value)


# Differentiable clamp via "re-parameterization"
def rclamp(x, min, max):
    return x - (x - torch.clamp(x, min, max))


# (Multi-dim) indexing
def gather(item, ind, dim=-1, ind_dim=-1):
    """
    Generalizes torch.gather indexing to multi-dim indexing.

    Indexes a specific dimension "dim" in "item"  and any number of subsequent dimensions depending on ind.

    Automatically batches/broadcasts batch and tail shapes depending on i and j:
        item: [item.size(0), ..., item.size(N), item.size(dim), item.size(N + 2), ..., item.size(M)],
        ind: [item.size(i), ..., item.size(N), ind.size(ind_dim), item.size(N + 2), ..., item.size(j)] where i ≤ N j ≤ M
        --> [item.size(0), ..., item.size(N), ind.size(ind_dim), item.size(N + 2), ..., item.size(M)]
    """

    ind_shape = ind.shape[ind_dim:]  # ["ind_dim", ..., j]
    tail_shape = item.shape[dim:][len(ind_shape):]  # [j + 1, ..., M]

    ind = ind.long().expand(*item.shape[:dim], *ind_shape)  # Assumes ind.shape[ind_dim] is desired num indices
    ind = ind.reshape(ind.shape + (1,) * len(tail_shape)).expand(*ind.shape, *tail_shape)  # [0, ..., "ind_dim", ... M]

    return torch.gather(item, dim, ind)


# (Multi-dim) cartesian product
def batched_cartesian_prod(items: (list, tuple), dim=-1, collapse_dims=True):
    """
    # Get all combinations of tensors starting at "dim", keeping all dims before "dim" independent (as batches)

    # Given N tensors with leading dims (B1 x B2 x ... x BL),
    # a specified "dim" (can vary in size across tensors, e.g. D1, D2, ..., DN), "dim" = L + 1
    # and tail dims (O1 x O2 x ... x OT), returns:
    # --> cartesian prod, batches independent:
    # --> B1 x B2 x ... x BL x D1 * D2 * ... * DN x O1 X O2 x ... x OT x N
    # Or if not collapse_dims:
    # --> B1 x B2 x ... x BL x D1 x D2 x ... x DN x O1 X O2 x ... x OT x N

    Consistent with torch.cartesian_prod except generalized to multi-dim and batches.
    """

    lead_dims = items[0].shape[:dim]  # B1, B2, ..., BL
    dims = [item.shape[dim] for item in items]  # D1, D2, ..., DN
    tail_dims = items[0].shape[dim + 1:] if dim + 1 else []  # O1, O2, ..., OT

    return torch.stack([item.reshape(-1, *(1,) * i, item.shape[dim], *(1,) * (len(items) - i - 1), *tail_dims).expand(
        -1, *dims[:i], item.shape[dim], *dims[i + 1:], *tail_dims)
        for i, item in enumerate(items)], -1).view(*lead_dims, *[-1] if collapse_dims else dims, *tail_dims, len(items))


# Sequential of instantiations e.g. python Run.py Eyes=Sequential +eyes._targets_="[CNN, Transformer]"
class Sequential(nn.Module):
    def __init__(self, _targets_, i=0, **kwargs):
        super().__init__()

        self.Sequence = nn.ModuleList()

        for _target_ in _targets_:
            self.Sequence.append(instantiate(OmegaConf.create({'_target_': _target_}) if isinstance(_target_, str)
                                             else _target_, i, **kwargs))

            if 'input_shape' in kwargs:
                kwargs['input_shape'] = cnn_feature_shape(kwargs['input_shape'], self.Sequence[-1])

    def repr_shape(self, *_):
        return cnn_feature_shape(_, self.Sequence)

    def forward(self, obs, *context):
        out = (obs, *context)
        # Multi-input/output Sequential
        for i, module in enumerate(self.Sequence):
            out = module(*out)
            if not isinstance(out, tuple) and i < len(self.Sequence) - 1:
                out = (out,)
        return out


# Swaps image dims between channel-last and channel-first format (Convenient helper)
class ChannelSwap(nn.Module):
    def repr_shape(self, *_):
        return _[-1], *_[1:-1], _[0]

    def forward(self, x, spatial2d=True):
        return x.transpose(-1, -3 if spatial2d and len(x.shape) > 3 else 1)  # Assumes 2D, otherwise Nd


# Convenient helper
ChSwap = ChannelSwap()


# Converts data to torch Tensors and moves them to the specified device as floats
def to_torch(xs, device=None):
    return tuple(None if x is None
                 else torch.as_tensor(x, dtype=torch.float32, device=device) for x in xs)


# Converts lists or scalars to tuple, preserving NoneType
def to_tuple(items: (int, float, bool, list, tuple)):
    return None if items is None else (items,) if isinstance(items, (int, float, bool)) else tuple(items)


# Multiples list items or returns item
def prod(items: (int, float, bool, list, tuple)):
    return items if isinstance(items, (int, float, bool)) or items is None else math.prod(items)


# Shifts to positive, normalizes to [0, 1]
class Norm(nn.Module):
    def __init__(self, start_dim=-1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        y = x.flatten(self.start_dim)
        y = y - y.min(-1, keepdim=True)[0]
        y = y / y.max(-1, keepdim=True)[0]
        return y.view(*x.shape)


# Context manager that temporarily switches on eval() mode for specified models; then resets them
class act_mode:
    def __init__(self, *models):
        super().__init__()

        self.models = models

    def __enter__(self):
        self.start_modes = []
        for model in self.models:
            if model is None:
                self.start_modes.append(None)
            else:
                self.start_modes.append(model.training)
                model.eval()  # Disables things like dropout, etc.

    def __exit__(self, *args):
        for model, mode in zip(self.models, self.start_modes):
            if model is not None:
                model.train(mode)


# Pytorch incorrect (in this case) warning suppression
warnings.filterwarnings("ignore", message='.* skipping the first value of the learning rate schedule')


# Scales gradients for automatic mixed precision training speedup, or updates gradients normally
class MixedPrecision:
    def __init__(self):
        self.mixed_precision_enabled = False  # Corresponds to mixed_precision=true
        self.ready = False
        self.models = set()

    @cached_property
    def scaler(self):
        return torch.cuda.amp.GradScaler()  # Gradient scaler to magnify imprecise Float16 gradients

    def enable(self, args):
        self.mixed_precision_enabled = args.mixed_precision and 'cuda' in args.device

    # Backward pass
    def backward(self, loss, retain_graph=False):
        if self.ready:
            loss = self.scaler.scale(loss)
        loss.backward(retain_graph=retain_graph)  # Backward

    # Optimize
    def step(self, model):
        if self.mixed_precision_enabled:
            if self.ready:
                # Model must AutoCast-initialize before first call to update
                assert id(model) in self.models, 'A new model or block is being optimized after the initial learning ' \
                                                 'update while "mixed_precision=true". ' \
                                                 'Not supported by lazy-AutoCast. Try "mixed_precision=false".'
                try:
                    return self.scaler.step(model.optim)  # Optimize
                except RuntimeError as e:
                    if 'step() has already been called since the last update().' in str(e):
                        e = RuntimeError(
                            f'The {type(model)} optimizer is being stepped twice while "mixed_precision=true" is '
                            'enabled. Currently, Pytorch automatic mixed precision only supports stepping an optimizer '
                            'once per update. Try running with "mixed_precision=false".')
                    raise e

            # Lazy-initialize AutoCast context

            forward = model.forward

            # Enable Pytorch AutoCast context
            model.forward = torch.autocast('cuda', dtype=torch.float16)(forward)

            for module in model.children():  # In case parts are shared across blocks e.g. Discrete Critic <- Actor
                forward = module.forward

                module.forward = torch.autocast('cuda', dtype=torch.float16)(forward)

            # EMA
            if hasattr(model, 'ema'):
                forward = model.ema.forward

                model.ema.forward = torch.autocast('cuda', dtype=torch.float16)(forward)

            self.models.add(id(model))

        model.optim.step()  # Optimize

    def update(self):
        if self.ready:
            self.scaler.update()  # Update gradient scaler
        self.ready = True


MP = MixedPrecision()  # AutoCast + GradScaler scales gradients for automatic mixed precision training speedup


# Backward pass on a loss; clear the grads of models; update EMAs; step optimizers and schedulers
def optimize(loss, *models, clear_grads=True, backward=True, retain_graph=False, step_optim=True, epoch=0, ema=True):
    # Clear grads
    if clear_grads and loss is not None:
        for model in models:
            if model.optim:
                model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        MP.backward(loss, retain_graph)  # Backward pass

    # Optimize
    if step_optim:
        for model in models:
            # Step scheduler
            if model.scheduler and epoch > model.scheduler.last_epoch:
                model.scheduler.step()
                model.scheduler.last_epoch = epoch

            # Update EMA target
            if ema and hasattr(model, 'ema'):
                update_ema_target(source=model, target=model.ema, ema_decay=model.ema_decay)

            if model.optim:
                MP.step(model)  # Step optimizer

                if loss is None and clear_grads:
                    model.optim.zero_grad(set_to_none=True)


# Increment/decrement a value in proportion to a step count and a string-formatted schedule
def schedule(schedule, step):
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        if match:
            start, stop, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * start + mix * stop
