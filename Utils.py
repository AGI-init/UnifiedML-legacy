# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
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
from Blocks.Augmentations import RandomShiftsAug, IntensityAug  # For direct accessibility via command line
from Blocks.Architectures import *  # For direct accessibility via command line


# Sets all Pytorch and Numpy random seeds
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Initializes run state
def init(args):
    # Set seeds
    set_seeds(args.seed)

    # Set device
    mps = getattr(torch.backends, 'mps', None)  # M1 MacBook speedup

    args.device = args.device or ('cuda' if torch.cuda.is_available()
                                  else 'mps' if mps and mps.is_available() else 'cpu')
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
    if hasattr(args, '_override_'):
        kwargs.update(args.pop('_override_'))  # For loading old models with new, overridden args

    while hasattr(args, '_default_'):  # Allow inheritance between shorthands
        args = args['_default_'] if isinstance(args['_default_'], str) else DictConfig(args.pop('_default_'), **args)

    if hasattr(args, '_target_') and args._target_:
        try:
            return hydra.utils.instantiate(args, **kwargs)  # Regular hydra
        except ImportError as e:
            if '(' in args._target_ and ')' in args._target_:  # Direct code execution
                args = args._target_
            else:
                args._target_ = 'Utils.' + args._target_  # Portal into Utils
                try:
                    return instantiate(args, i, **kwargs)
                except ImportError:
                    raise e  # Original error if all that doesn't work
        except TypeError as e:
            kwarg = re.search('got an unexpected keyword argument \'(.+?)\'', str(e))
            if kwarg:
                kwargs = {key: kwargs[key] for key in kwargs if key != kwarg.group(1)}
                return instantiate(args, i, **kwargs)  # Signature matching
            raise e  # Original error

    if isinstance(args, str):
        for key in kwargs:
            args = args.replace(f'kwargs.{key}', f'kwargs["{key}"]')  # Interpolation
        args = eval(args)  # Direct code execution

    return None if hasattr(args, '_target_') \
        else args(**{key: kwargs[key]
                     for key in kwargs.keys() & signature(args).parameters}) if isinstance(args, type) \
        else args[i] if isinstance(args, (list, nn.ModuleList)) \
        else args  # Additional useful ones


# Initializes model weights a la orthogonal
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# Initializes model optimizer. Default: AdamW + cosine annealing
def optimizer_init(params, optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None):
    params = list(params)

    # Optimizer
    optim = len(params) > 0 and (instantiate(optim, params=params, lr=getattr(optim, 'lr', lr)) or lr
                                 and AdamW(params, lr=lr, weight_decay=weight_decay))  # Default

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
def cnn_layer_feature_shape(in_height, in_width, kernel_size=1, stride=1, padding=0, dilation=1):
    assert in_height and in_width, f'Height and width must be positive integers, got {in_height}, {in_width}'
    if padding == 'same':
        return in_height, in_width
    if type(kernel_size) is not tuple:
        kernel_size = [kernel_size, kernel_size]
    if type(stride) is not tuple:
        stride = [stride, stride]
    if type(padding) is not tuple:
        padding = [padding, padding]
    if type(dilation) is not tuple:
        dilation = [dilation, dilation]
    kernel_size = [min(size, kernel_size[i]) for i, size in enumerate([in_height, in_width])]  # Assumes adaptive
    padding = [min(size, padding[i]) for i, size in enumerate([in_height, in_width])]  # Assumes adaptive
    out_height = math.floor(((in_height + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    out_width = math.floor(((in_width + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
    return out_height, out_width


# Compute the output shape of a whole CNN
def cnn_feature_shape(chw, *blocks, verbose=False):
    channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
    for block in blocks:
        if isinstance(block, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)):
            channels = block.out_channels if hasattr(block, 'out_channels') else channels
            height, width = cnn_layer_feature_shape(height, width,
                                                    kernel_size=block.kernel_size,
                                                    stride=block.stride,
                                                    padding=block.padding)
        elif isinstance(block, nn.Linear):
            channels = block.out_features  # Assumes channels-last if linear
        elif isinstance(block, nn.Flatten) and (block.start_dim == -3 or block.start_dim == 1):
            channels, height, width = channels * (height or 1) * (width or 1), None, None  # Placeholder height/width
        elif isinstance(block, nn.AdaptiveAvgPool2d):
            height, width = block.output_size
        elif hasattr(block, 'repr_shape'):
            chw = block.repr_shape(channels, height, width)
            channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
        elif hasattr(block, 'modules'):
            for layer in block.children():
                chw = cnn_feature_shape([channels, height, width], layer, verbose=verbose)
                channels, height, width = chw[0], chw[1] if len(chw) > 1 else None, chw[2] if len(chw) > 2 else None
        if verbose:
            print(block, (channels, height, width))

    feature_shape = tuple(size for size in (channels, height, width) if size is not None)

    return feature_shape


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
    def __init__(self, size=1, uniform=False):
        super().__init__()

        self.size = size
        self.uniform = uniform

    def repr_shape(self, *_):
        return self.size,

    def forward(self, *x):
        x = torch.randn((x[0].shape[0], self.size), device=x[0].device)

        if self.uniform:
            x.uniform_()

        return x


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes, null_value=0, one_value=1):
    # assert x.shape[-1] == 1  # Can check this
    x = x.squeeze(-1).unsqueeze(-1)  # Or do this
    x = x.long()
    shape = x.shape[:-1]
    nulls = torch.full([*shape, num_classes], null_value, dtype=x.dtype, device=x.device)
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

    Indexes a specific dimension "dim" in "item"  and any number of subsequent dimensions depending on "ind_dim".
    The index "ind" can share consecutive dimensions with "item" prior to "dim" or will be batched automatically.

    Relative coordinates, assume "dim" = "ind_dim":
    item: [0, ..., N, "dim", N + 2, ..., M], ind: [i, ..., N, "ind_dim", N + 2, ..., j], i ≤ N + 1, j ≤ M + 1
    --> [0, ..., N, "ind_dim", N + 2, ..., M]
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

        modules = nn.ModuleList()

        for _target_ in _targets_:
            modules.append(instantiate(OmegaConf.create({'_target_': _target_}), i, **kwargs))

            if 'input_shape' in kwargs:
                kwargs['input_shape'] = cnn_feature_shape(kwargs['input_shape'], modules[-1])

        self.Sequence = nn.Sequential(*modules)

    def repr_shape(self, *_):
        return cnn_feature_shape(_, self.Sequence)

    def forward(self, obs):
        return self.Sequence(obs)


# Swaps image dims between channel-last and channel-first format
class ChannelSwap(nn.Module):
    def repr_shape(self, *_):
        return _[-1], *_[1:-1], _[0]

    def forward(self, x, spatial2d=True):
        return x.transpose(-1, -3 if spatial2d and len(x.shape) > 3 else 1)  # Assumes 2D, otherwise Nd


# Convenient helper
ChSwap = ChannelSwap()


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
                model.eval()

    def __exit__(self, *args):
        for model, mode in zip(self.models, self.start_modes):
            if model is not None:
                model.train(mode)
        return False


# Converts data to torch Tensors and moves them to the specified device as floats
def to_torch(xs, device=None):
    return tuple(None if x is None
                 else torch.as_tensor(x, device=device).float() for x in xs)


# Pytorch incorrect (in this case) warning suppression
warnings.filterwarnings("ignore", message='.* skipping the first value of the learning rate schedule')


# Backward pass on a loss; clear the grads of models; update EMAs; step optimizers and schedulers
def optimize(loss, *models, clear_grads=True, backward=True, retain_graph=False, step_optim=True, epoch=0, ema=True):
    # Clear grads
    if clear_grads and loss is not None:
        for model in models:
            if model.optim:
                model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        loss.backward(retain_graph=retain_graph)

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
                model.optim.step()  # Step optimizer

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
