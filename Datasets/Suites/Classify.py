# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import datetime
import glob
import io
import itertools
import json
import random
from pathlib import Path

from tqdm import tqdm

from PIL.Image import Image

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import functional as F

from Utils import instantiate


class Classify:
    """
    A general-purpose environment:

    Must accept: **kwargs as init arg.

    Must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "render" function, -> image
    (4) "episode_done" attribute
    (5) "obs_spec" attribute which includes:
        - "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (6) "action-spec" attribute which includes:
        - "shape", "discrete_bins" (should be None if not discrete), "low", "high", and "discrete"
    (7) "exp" attribute containing the latest exp

    Recommended: Discrete environments should have a conversion strategy for adapting continuous actions (e.g. argmax)

    An "exp" (experience) is an AttrDict consisting of "obs", "action" (prior to adapting), "reward", "label", "step"
    numpy values which can be NaN. Must include a batch dim.

    ---

    Extended to accept a dataset config, which instantiates a Dataset. Datasets must:
    - extend Pytorch Datasets
    - include a "classes" (num classes) attribute
    - output (obs, label) pairs

    An "evaluate_episodes" attribute divides evaluation across batches since batch=episode

    """
    def __init__(self, dataset, task='MNIST', train=True, offline=True, generate=False, batch_size=32, num_workers=1,
                 low=None, high=None, frame_stack=None, action_repeat=None, seed=None, **kwargs):
        self.episode_done = False

        # Don't need once moved to replay (see below)
        dataset_ = dataset

        # Make env

        # Different datasets have different specs
        root_specs = [dict(root=f'./Datasets/ReplayBuffer/Classify/{task}_%s' %
                                ('Train' if train else 'Eval')), {}]
        train_specs = [dict(train=train),
                       dict(version='2021_' + 'train' if train else 'valid'), {}]
        download_specs = [dict(download=True), {}]
        transform_specs = [dict(transform=Transform()), {}]

        # Instantiate dataset
        for all_specs in itertools.product(root_specs, train_specs, download_specs, transform_specs):
            try:
                root_spec, train_spec, download_spec, transform_spec = all_specs
                specs = dict(**root_spec, **train_spec, **download_spec, **transform_spec)
                specs.update(kwargs)
                dataset = instantiate(dataset, **specs)
            except (TypeError, ValueError):
                continue
            break

        assert isinstance(dataset, Dataset), 'Dataset must be a Pytorch Dataset or inherit from a Pytorch Dataset'

        self.action_spec = {'shape': (1,),
                            'discrete_bins': len(dataset.classes),  # Dataset must include a "classes" attr
                            'low': 0,
                            'high': len(dataset.classes) - 1,
                            'discrete': True}

        self.batches = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn)

        self._batches = iter(self.batches)

        obs_shape = tuple(next(iter(self.batches))[0].shape[1:])
        obs_shape = (1,) * (3 - len(obs_shape)) + obs_shape  # 3D

        self.obs_spec = {'shape': obs_shape}

        """MOVE TO REPLAY"""

        replay_path = Path(f'./Datasets/ReplayBuffer/Classify/{task}_Buffer')

        stats_path = glob.glob(f'./Datasets/ReplayBuffer/Classify/{task}_Stats_*')

        # Offline and generate don't use training rollouts
        if (offline or generate) and not train and (not replay_path.exists() or not len(stats_path)):

            # But still need to create training replay & compute stats
            Classify(dataset_, task, True, offline, generate, batch_size, num_workers, None, None, None, None, seed,
                     **kwargs)

        # Create replay
        if train and (offline or generate) and not replay_path.exists():
            self.create_replay(replay_path)

        stats_path = glob.glob(f'./Datasets/ReplayBuffer/Classify/{task}_Stats_*')

        # Compute stats
        mean, stddev, low_, high_ = map(json.loads, stats_path[0].split('_')[-4:]) if len(stats_path) \
            else self.compute_stats(f'./Datasets/ReplayBuffer/Classify/{task}') if train else 0
        low, high = low if low is None else low_, high_ if high is None else high

        # No need
        if (offline or generate) and train:
            self.batches = self._batches = dataset = None
            return

        """---------------------"""

        self.obs_spec = {'shape': obs_shape,
                         'mean': mean,
                         'stddev': stddev,
                         'low': low,
                         'high': high}

        self.exp = None  # Experience

        self.evaluate_episodes = len(self.batches)

    def step(self, action):
        # Adapt to discrete!
        _action = self.adapt_to_discrete(action)

        correct = (self.exp.label == _action).astype('float32')

        self.exp.reward = correct
        self.exp.action = action  # Note: can store argmax instead

        self.episode_done = True

        return self.exp

    def reset(self):
        obs, label = [np.array(b, dtype='float32') for b in self.sample()]
        label = np.expand_dims(label, 1)

        batch_size = obs.shape[0]

        obs.shape = (batch_size, *self.obs_spec['shape'])

        self.episode_done = False

        # Create experience
        exp = {'obs': obs, 'action': None, 'reward': None, 'label': label, 'step': None}

        # Scalars/NaN to numpy
        for key in exp:
            if np.isscalar(exp[key]) or exp[key] is None or type(exp[key]) == bool:
                exp[key] = np.full([1, 1], exp[key], dtype=getattr(exp[key], 'dtype', 'float32'))
            elif len(exp[key].shape) in [0, 1]:  # Add batch dim
                exp[key].shape = (1, *(exp[key].shape or [1]))

        self.exp = AttrDict(exp)  # Experience

        return self.exp

    def render(self):
        # Assumes image dataset
        image = self.sample()[0] if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image) - 1)], dtype='uint8').transpose(1, 2, 0)

    def sample(self):
        try:
            return next(self._batches)
        except StopIteration:
            self._batches = iter(self.batches)
            return next(self._batches)

    def create_replay(self, path):
        path.mkdir(exist_ok=True, parents=True)

        for episode_ind, (obs, label) in enumerate(tqdm(self.batches, 'Creating a universal replay for this dataset. '
                                                                      'This only has to be done once')):
            obs, label = [np.array(b, dtype='float32') for b in (obs, label)]
            label = np.expand_dims(label, 1)

            batch_size = obs.shape[0]

            obs.shape = (batch_size, *self.obs_spec['shape'])

            dummy = np.full((batch_size, 1), np.NaN)
            missing = np.full((batch_size, *self.action_spec['shape'] + (self.action_spec['discrete_bins'],)), np.NaN)

            episode = {'obs': obs, 'action': missing, 'reward': dummy, 'label': label, 'step': dummy}

            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            episode_name = f'{timestamp}_{episode_ind}_{batch_size}.npz'

            with io.BytesIO() as buffer:
                np.savez_compressed(buffer, **episode)
                buffer.seek(0)
                with (path / episode_name).open('wb') as f:
                    f.write(buffer.read())

    def compute_stats(self, path):
        cnt = 0
        fst_moment, snd_moment = None, None
        low, high = np.inf, -np.inf

        for obs, _ in tqdm(self.batches, 'Computing mean, stddev, low, high for standardization/normalization. '
                                         'This only has to be done once'):

            b = obs.shape[0]
            _, c, h, w = (b, *self.obs_spec['shape'])
            obs = obs.view(b, c, h, w)
            fst_moment = torch.empty(c) if fst_moment is None else fst_moment
            snd_moment = torch.empty(c) if snd_moment is None else snd_moment
            nb_pixels = b * h * w
            sum_ = torch.sum(obs, dim=[0, 2, 3])
            sum_of_square = torch.sum(obs ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

            low, high = min(obs.min(), low), max(obs.max(), high)

        mean, stddev = fst_moment.tolist(), torch.sqrt(snd_moment - fst_moment ** 2).tolist()
        open(path + f'_Stats_{mean}_{stddev}_{low}_{high}', 'w')  # Save stat values for future reuse

        return mean, stddev, low.item(), high.item()

    def adapt_to_discrete(self, action):
        shape = self.action_spec['shape']

        try:
            action = action.reshape(len(action), *shape)  # Assumes a batch dim
        except (ValueError, RuntimeError):
            try:
                action = action.reshape(len(action), -1, *shape)  # Assumes a batch dim
            except:
                raise RuntimeError(f'Discrete environment could not broadcast or adapt action of shape {action.shape} '
                                   f'to expected batch-action shape {(-1, *shape)}')
            action = action.argmax(1)

        discrete_bins, low, high = self.action_spec['discrete_bins'], self.action_spec['low'], self.action_spec['high']

        # Round to nearest decimal/int corresponding to discrete bins, high, and low
        return np.round((action - low) / (high - low) * (discrete_bins - 1)) / (discrete_bins - 1) * (high - low) + low


class Transform:
    def __call__(self, sample):
        return F.to_tensor(sample) if isinstance(sample, Image) else sample


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(_dict)

