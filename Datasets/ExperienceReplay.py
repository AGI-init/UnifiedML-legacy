# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import random
import glob
import shutil
import atexit
import uuid
import warnings
import resource
from pathlib import Path
import datetime
import io
import traceback
from time import sleep

from omegaconf import OmegaConf

import numpy as np

from multiprocessing.shared_memory import SharedMemory, ShareableList

import torch
from torch.utils.data import IterableDataset, Dataset
from torch.multiprocessing import Pipe, Manager

from torchvision.transforms import transforms


class ExperienceReplay:
    def __init__(self, batch_size, num_workers, capacity, suite, task, offline, generate, stream, save, load, path, env,
                 obs_spec, action_spec, frame_stack=1, nstep=0, discount=1, meta_shape=(0,), transform=None):
        # Path and loading

        exists = glob.glob(path + '*/')

        offline = offline or generate

        if load or offline:
            if suite == 'classify':
                if env.subset:
                    task += '_Classes_' + '_'.join(map(str, env.subset))  # Subset of classes
                if getattr(env.transform, '_target_', env.transform) is not None:
                    task += '_Transformed'  # Pre-transformed by environment (fixed transformations)
                standard = f'./Datasets/ReplayBuffer/Classify/{task}_Buffer'
                if len(exists) == 0:
                    exists = [standard + '/']
                    print('All data loaded. Training underway.')
                else:
                    if path != standard:
                        warnings.warn(f'Loading a saved replay of a classify task from a previous online session. '
                                      f'For the standard offline dataset, set replay.path="{standard}" '  # If exists
                                      f'or delete the saved buffer in {path}.')
            assert len(exists) > 0, f'\nNo existing replay buffer found in path: {path}.\ngenerate=true, ' \
                                    f'offline=true, & replay.load=true all assume the presence of a saved replay ' \
                                    f'buffer. \nTry replay.save=true first, then you can try again with one ' \
                                    f'of those 3, \nor set those to false, or choose a different path via replay.path=.'
            self.path = Path(sorted(exists)[-1])
            save = offline or save
        elif not stream:
            self.path = Path(path + '_' + str(datetime.datetime.now()))
            self.path.mkdir(exist_ok=True, parents=True)

        if not save and hasattr(self, 'path'):
            # Delete replay on terminate
            atexit.register(lambda p: (shutil.rmtree(p), print('Deleting replay')), self.path)

        # Data specs

        self.frame_stack = frame_stack or 1
        obs_spec.shape[0] //= self.frame_stack

        self.specs = {'obs': obs_spec, 'action': action_spec,
                      'meta': {'shape': meta_shape},  # Shape of optional addable/update-able metadata from Agent
                      **{name: {'shape': (1,)} for name in ['reward', 'label', 'step']}}

        # Episode traces (temporary in-RAM buffer until full episode ready to be stored and cached)

        self.episode = {name: [] for name in self.specs}
        self.episode_len = 0
        self.episodes_stored = 0 if stream else len(list(self.path.glob('*.npz')))
        self.save = save

        self.offline = offline  # Dataset doesn't grow
        self.stream = stream  # Streaming from Environment directly

        # Data transform

        if transform is not None:
            if self.stream:
                warnings.warn('Command-line transforms (`transform=`) are not supported for streaming (`stream=true`). '
                              'Try a custom dataset instead via `Dataset=` defined with your preferred transforms.')
            if isinstance(transform, str):
                transform = OmegaConf.create(transform)
            if 'RandomCrop' in transform and 'size' not in transform['RandomCrop']:
                transform['RandomCrop']['size'] = obs_spec['shape'][-2:]
            if 'Normalize' in transform:
                warnings.warn('"Normalizing" via transform. This may be redundant and dangerous if standardize=true, '
                              'which is the default.')
            # Can pass in a dict of torchvision transform names and args
            transform = transforms.Compose([getattr(transforms, t)(**transform[t]) for t in transform])

        # Future steps to compute cumulative reward from
        self.nstep = 0 if suite == 'classify' or generate else nstep

        self.epoch = 1

        if self.stream:
            if self.nstep > 0:
                self.nstep = 0
                warnings.warn('Setting nstep=0. Streaming does not currently support N-step cumulative reward.')
            return

        """
        ---Parallelized experience loading--- 
        
        Either Online or Offline. "Online" means the data size grows.
        
          Offline, data is adaptively pre-loaded onto CPU RAM from hard disk before training. Caching on RAM is faster 
          than loading from hard disk, epoch by epoch. Caching is done with truly-shared RAM memory for Offline. 

          The disadvantage of CPU pre-loading is the dependency on more CPU RAM. The "capacity=" a.k.a. "RAM_capacity=" 
          adapts how many experiences get stored on RAM. 
          
          The rest gets memory-mapped on hard disk if Offline. Memory mapping is an efficient hard disk loading format.

          If Online, data is also cached on RAM, after storing to hard disk. "capacity=" controls total capacity for 
          training. Online doesn't currently support hard disk caching. Saving is supported with "replay.save=true".
        """

        # CPU workers
        self.num_workers = max(1, min(num_workers, os.cpu_count()))

        # RAM capacity per worker. Max num experiences allotted per CPU worker
        capacity = capacity // self.num_workers if capacity not in [-1, 'inf'] else np.inf

        # For sending data to workers directly
        pipes, self.pipes = zip(*[Pipe(duplex=False) for _ in range(self.num_workers)])

        self.experiences = (Offline if offline else Online)(path=self.path,
                                                            capacity=capacity,
                                                            specs=self.specs,
                                                            fetch_per=1000,
                                                            pipes=pipes,
                                                            save=save,
                                                            frame_stack=self.frame_stack,
                                                            nstep=self.nstep,
                                                            discount=discount,
                                                            transform=transform)

        # Batch loading

        self.batches = torch.utils.data.DataLoader(dataset=self.experiences,
                                                   batch_size=batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True,
                                                   shuffle=offline,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=True)

        # Replay

        self._replay = None

    # Allows iteration via "next" (e.g. batch = next(replay))
    def __next__(self):
        return self.sample()

    # Samples stored experiences or pulls from a data stream, optionally includes trajectories, returns a batch
    def sample(self, trajectories=False):
        if self.stream:
            # Streaming
            return [self.stream.get(key, torch.empty([1, 0]))
                    for key in ['obs', 'action', 'reward', 'discount', 'next_obs', 'label', *[None] * 4 * trajectories,
                                'step', 'ids', 'meta']]  # Return contents of the data stream
        else:
            # Sampling
            try:
                sample = next(self.replay)
            except StopIteration:
                self.epoch += 1
                self._replay = None  # Reset iterator when depleted
                sample = next(self.replay)
            return *sample[:10 if trajectories else 6], *sample[10:]  # Return batch, w(/o) future-trajectories

    # Initial iterator, allows replay iteration
    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)  # Recreates the iterator when exhausted
        return self._replay

    # Tracks single episode "trace" in memory buffer
    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # An "episode" or part of episode of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for name, spec in self.specs.items():
                # Missing data
                if name not in exp or exp[name] is None:
                    exp[name] = np.zeros((1, 0))

                # Add batch dimension to singles / convert to numpy
                if np.isscalar(exp[name]) or not isinstance(exp[name], np.ndarray):
                    exp[name] = np.full((1, *spec['shape']), exp[name], dtype=getattr(exp[name], 'dtype', 'float32'))
                elif len(exp[name].shape) in [0, 1]:
                    exp[name].shape = (1, exp[name].size)

                # Expands attributes that are unique per batch (such as 'step')
                batch_size = exp.get('obs', exp['action']).shape[0]
                if 1 == exp[name].shape[0] < batch_size:
                    exp[name] = np.repeat(exp[name], batch_size, axis=0)

                # Validate consistency - disabled for discrete/continuous conversions
                # assert spec['shape'] == exp[name].shape[1:], \
                #     f'Unexpected shape for "{name}": {spec["shape"]} vs. {exp[name].shape[1:]}'
                spec['shape'] = exp[name].shape[1:]

                # Add the experience
                self.episode[name].append(exp[name])

            if self.stream:
                self.stream = exp  # For streaming directly from Environment

        # Count experiences in episode
        self.episode_len += len(experiences)

        if store:
            self.store_episode()  # Stores them in file system

    # Stores episode (to file in system)
    def store_episode(self):
        if self.episode_len == 0:
            return

        for name, spec in self.specs.items():
            # Concatenate into one big episode batch. Presumes a pre-existing batch dimension in each experience
            self.episode[name] = np.concatenate(self.episode[name], axis=0).astype(np.float32)

        self.episode_len = len(self.episode['obs'])

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        num_episodes = len(self)
        episode_name = f'{timestamp}_{num_episodes}_{self.episode_len}.npz'

        # Save episode
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **self.episode)
            buffer.seek(0)
            with (self.path / episode_name).open('wb') as f:
                f.write(buffer.read())

        self.episode = {name: [] for name in self.specs}
        self.episode_len = 0
        self.episodes_stored += 1

    def clear(self):
        self.episode = {name: [] for name in self.specs}
        self.episode_len = 0

    # Update experiences (in workers) by IDs (experience index and worker ID) and dict like {spec: update value}
    def rewrite(self, updates, ids):
        assert isinstance(updates, dict), f'expected \'updates\' to be dict, got {type(updates)}'

        updates = {key: updates[key].detach() for key in updates}
        exp_ids, worker_ids = ids.detach().int().T

        # Send update to dedicated worker
        for worker_id in torch.unique(worker_ids):
            worker = worker_ids == worker_id
            update = {key: updates[key][worker] for key in updates}
            self.pipes[worker].send((update, exp_ids[worker]))

    def __len__(self):
        # Infinite if stream, else the number of episodes stored if constant or some deleted, else the episodes in path
        return int(5e11) if self.stream else self.episodes_stored if self.offline or not self.save \
            else len(list(self.path.glob('*.npz')))


# How to initialize each worker
def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(int(seed))


# CPU workers that can iteratively and efficiently builds/updates batches of experience in parallel (from files/RAM)
class Experiences:
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, offline, frame_stack, nstep, discount, transform):

        # Dataset construction via parallel workers

        self.path = path

        self.episode_names = []
        self.episodes = SharedDict(specs) if offline else dict()  # Episodes fetched on CPU

        # Or can use Python's built-in shared memory, which serializes and de-serializes (isn't truly-shared, slower)
        # self.episodes = Manager().dict() if offline else dict()

        self.experience_indices = []
        self.capacity = capacity
        self.deleted_indices = 0

        self.specs = specs

        self.fetch_per = fetch_per  # How often to fetch
        self.samples_since_last_fetch = fetch_per

        # Multiprocessing pipes
        self.pipes = pipes  # Receiving data directly from Replay

        self.save = save
        self.offline = offline

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.discount = discount

        self.transform = transform

        # If Offline, share RAM across CPU workers
        if self.offline:
            self.num_experiences = sum([int(episode_name.stem.split('_')[-1]) - (self.nstep or 0)
                                        for episode_name in self.path.glob('*.npz')])

        self.initialized = False

    @property
    def worker_id(self):
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0

    @property
    def num_workers(self):
        return len(self.pipes)

    @property
    def pipe(self):
        return self.pipes[self.worker_id]

    def init_worker(self):
        if isinstance(self.episodes, SharedDict):
            atexit.register(self.episodes.cleanup)

        self.worker_fetch_episodes()  # Load in existing data

        # If Offline, share RAM across CPU workers
        if self.offline:
            self.episode_names = sorted(self.path.glob('*.npz'))

            self.experience_indices = sum([list(enumerate([episode_name] * (int(episode_name.stem.split('_')[-1])
                                                                            - (self.nstep or 0))))
                                           for episode_name in self.episode_names], [])  # Slightly redundant per worker

        self.initialized = True

    def load_episode(self, episode_name):
        try:
            with episode_name.open('rb') as episode_file:
                episode = np.load(episode_file)
                episode = {key: episode[key] for key in episode.keys()}
        except:
            return False

        offset = self.nstep or 0
        episode_len = len(episode['obs']) - offset
        episode = {name: episode.get(name, np.zeros((episode_len, 0))) for name, spec in self.specs.items()}

        episode['id'] = len(self.experience_indices)
        self.experience_indices += list(enumerate([episode_name] * episode_len))

        self.episode_names.append(episode_name)
        self.episode_names.sort()

        # If Offline replay exceeds RAM ("capacity=" flag) (approximately), keep episode on hard disk
        if self.offline and episode_len + len(self.experience_indices) > self.capacity:
            for spec in episode:
                if isinstance(episode[spec], np.ndarray) and episode[spec].size > 1:
                    path = self.path / f'{episode_name.stem}_{spec}.dat'
                    mmap_file = np.memmap(path, 'float32', 'w+', shape=episode[spec].shape)
                    mmap_file[:] = episode[spec][:]
                    mmap_file.flush()  # Write episode to hard disk
                    episode[spec] = mmap_file  # Replace episode with memory mapped link for efficient retrieval

            self.episodes[episode_name] = episode
            return True

        self.episodes[episode_name] = episode

        if not self.save:
            episode_name.unlink(missing_ok=True)  # Deletes file

        # Deleting experiences upon overfill if Online
        while episode_len + len(self) > self.capacity and not self.offline:
            early_episode_name = self.episode_names.pop(0)
            early_episode = self.episodes.pop(early_episode_name)
            early_episode_len = len(early_episode['obs']) - offset
            self.deleted_indices += early_episode_len  # To derive a consistent experience index even as data deleted

        return True

    # Populates workers with new data
    def worker_fetch_episodes(self):
        if self.samples_since_last_fetch < self.fetch_per:
            return

        self.samples_since_last_fetch = 0

        episode_names = sorted(self.path.glob('*.npz'), reverse=True)  # Episodes
        num_fetched = 0
        # Find new episodes
        for episode_name in episode_names:
            episode_idx, episode_len = [int(x) for x in episode_name.stem.split('_')[1:]]
            if episode_idx % self.num_workers != self.worker_id:  # Each worker stores their own dedicated data
                continue
            if episode_name in self.episodes.keys():  # Don't store redundantly
                break
            # if num_fetched + episode_len > self.capacity:  # Don't overfill  (This is already accounted for)
            #     break
            num_fetched += episode_len
            if not self.load_episode(episode_name):
                break  # Resolve conflicts

    # Can update/write data based on piped update specs
    def worker_fetch_updates(self):
        while self.pipe.poll():
            try:
                updates, exp_ids = self.pipe.recv()
            except EOFError:
                break

            # Iterate through each update spec
            for key in updates:
                for update, exp_id in zip(updates[key], exp_ids):
                    # Get corresponding experience and episode
                    idx, episode_name = self.experience_indices[exp_id]

                    # Update experience in replay
                    if episode_name in self.episodes:
                        self.episodes[episode_name][key][idx] = update.numpy()  # Note: Only compatible with numpy data

    def sample(self, episode_names, metrics=None):
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    # N-step cumulative discounted rewards
    def process(self, episode, idx=None):
        offset = self.nstep or 0
        episode_len = len(episode['obs']) - offset
        if idx is None:
            idx = np.random.randint(episode_len)

        # Frame stack
        def frame_stack(traj_o, idx):
            frames = traj_o[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):  # If not enough frames, reuse the first
                frames = np.concatenate([traj_o[:1], frames], 0)
            frames = frames.reshape(frames.shape[1] * self.frame_stack, *frames.shape[2:])
            return frames

        # Present
        obs = frame_stack(episode['obs'], idx)
        label = episode['label'][idx]
        step = episode['step'][idx]

        exp_id, worker_id = episode['id'] + idx, self.worker_id
        ids = np.array([exp_id, worker_id])

        meta = episode['meta'][idx]  # Agent-writable Metadata

        # Future
        if self.nstep:
            # Transition
            action = episode['action'][idx + 1]
            next_obs = frame_stack(episode['obs'], idx + self.nstep)

            # Trajectory
            traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
                                     for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
            traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
            traj_r = episode['reward'][idx + 1:idx + self.nstep + 1]
            traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(self.nstep + 1)
            reward = np.dot(discounts[:-1], traj_r)
            discount = discounts[-1:]
        else:
            action, reward = episode['action'][idx], episode['reward'][idx]

            next_obs = traj_o = traj_a = traj_r = traj_l = np.zeros(0)
            discount = np.array([1.0])

        # Transform
        if self.transform is not None:
            obs = self.transform(torch.as_tensor(obs))

        return obs, action, reward, discount, next_obs, label, traj_o, traj_a, traj_r, traj_l, step, ids, meta

    def fetch_sample_process(self, idx=None):
        # Populate workers with up-to-date data
        if not self.initialized:
            self.init_worker()
        try:
            if not self.offline:
                self.worker_fetch_episodes()
            self.worker_fetch_updates()
        except:
            traceback.print_exc()

        self.samples_since_last_fetch += 1

        # Sample or index an experience
        if idx is None:
            episode_name = self.sample(self.episode_names)
        else:
            idx, episode_name = self.experience_indices[idx + self.deleted_indices]

        episode = self.episodes[episode_name]

        return self.process(episode, idx)  # Process episode into a compact experience

    def __len__(self):
        return self.num_experiences if self.offline else len(self.experience_indices) - self.deleted_indices


# Loads Experiences with an Iterable Dataset
class Online(Experiences, IterableDataset):
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, frame_stack, nstep=0, discount=1, transform=None):
        super().__init__(path, capacity, specs, fetch_per, pipes, save, False, frame_stack, nstep, discount, transform)

    def __iter__(self):
        # Keep fetching, sampling, and building batches
        while True:
            yield self.fetch_sample_process()  # Yields a single experience


# Loads Experiences with a Map Style Dataset
class Offline(Experiences, Dataset):
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, frame_stack, nstep=0, discount=1, transform=None):
        super().__init__(path, capacity, specs, fetch_per, pipes, save, True, frame_stack, nstep, discount, transform)

    def __getitem__(self, idx):
        # Retrieve a single experience by index
        return self.fetch_sample_process(idx)


# Truly-shared RAM and memory-mapped hard disk data usable across parallel CPU workers
class SharedDict:
    def __init__(self, specs):
        self.specs = specs

        self.dict_id = str(uuid.uuid4())[:8]

        self.created = {}

        # Shared memory can create a lot of file descriptors
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Increase soft limit to hard limit just in case
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

    def __setitem__(self, key, value):
        assert isinstance(value, dict), 'Shared Memory must be a episode dict'

        num_episodes = key.stem.split('/')[-1].split('_')[1]

        for spec, data in value.items():
            name = self.dict_id + num_episodes + spec

            mmap = isinstance(data, np.memmap)  # Whether to memory map

            # Shared integers
            if spec == 'id':
                mem = self.create([data], name)
                mem[0] = data
                mem.shm.close()
            # Shared numpy
            elif data.nbytes > 0:
                # Hard disk memory-map link
                if mmap:
                    mem = self.create(list(str(data.filename)), name + 'mmap')
                    mem.shm.close()
                # Shared RAM memory
                else:
                    mem = self.create(data, name)  # Create
                    mem_ = np.ndarray(data.shape, dtype=data.dtype, buffer=mem.buf)
                    mem_[:] = data[:]  # Set data
                    mem.close()
                    mem = self.create([0], name + 'mmap')  # Set to False, no memory mapping
                    mem.shm.close()

            # Data shape
            mem = self.create([1] if np.isscalar(data) else list(data.shape), name + 'shape')
            mem.shm.close()

    def create(self, data, name=''):
        # Two ways to create a truly shared RAM memory in Python
        method = (ShareableList if isinstance(data, list) else SharedMemory)

        # Try to create shared memory link
        try:
            mem = method(data, name=name) if isinstance(data, list) \
                else method(create=True, name=name,  size=data.nbytes)

            self.created.update({name: method})
        except FileExistsError:
            mem = method(name=name)  # But if exists, retrieve existing shared memory link

        return mem

    def __getitem__(self, key):
        # Account for potential delay
        for _ in range(2400):
            try:
                return self._getitem(key)
            except FileNotFoundError as e:
                sleep(1)
        raise(e)

    def _getitem(self, key):
        num_episodes = key.stem.split('/')[-1].split('_')[1]

        episode = {}

        for spec in self.keys():
            name = self.dict_id + num_episodes + spec

            # Integer
            if spec == 'id':
                mem = ShareableList(name=name)
                episode[spec] = int(mem[0])
                mem.shm.close()
            # Numpy array
            else:
                # Shape
                mem = ShareableList(name=name + 'shape')
                shape = tuple(mem)
                mem.shm.close()

                if 0 in shape:
                    episode[spec] = np.zeros(shape)  # Empty array
                else:
                    # Whether memory mapped
                    mem = ShareableList(name=name + 'mmap')
                    is_mmap = list(mem)
                    mem.shm.close()
                    # View of memory
                    episode[spec] = SharedMem(name, shape, mmap_name=None if 0 in is_mmap else ''.join(is_mmap))

        return episode

    def keys(self):
        return self.specs.keys() | {'id'}

    def cleanup(self):
        for name, method in self.created.items():
            mem = method(name=name)

            if isinstance(mem, ShareableList):
                mem = mem.shm

            mem.close()
            mem.unlink()  # Unlink shared memory, assumes each worker is uniquely assigned the episodes to create()


# A special view into shared memory or memory mapped data that handles index-based reads and writes efficiently
class SharedMem:
    def __init__(self, name, shape, mmap_name=None):
        self.name, self.shape, self.mmap_name = name, shape, mmap_name

    def __getitem__(self, idx):
        if self.mmap_name is None:
            # Truly-shared RAM access
            mem = SharedMemory(name=self.name)

            value = np.ndarray(shape=self.shape, dtype=np.float32, buffer=mem.buf)[idx].copy()

            mem.close()

            return value
        else:
            # Read from memory-mapped hard disk file rather than shared RAM
            return np.memmap(self.mmap_name, np.float32, 'r+', shape=self.shape)[idx]

    def __setitem__(self, idx, value):
        if self.mmap_name is None:
            # Shared RAM memory
            mem = SharedMemory(name=self.name)

            value = np.ndarray(shape=self.shape, dtype=np.float32, buffer=mem.buf)  # Un-mutable shape!
            value[idx] = value

            mem.close()
        else:
            # Hard disk memory-map link
            mem = np.memmap(self.mmap_name, np.float32, 'r+', shape=self.shape)
            mem[idx] = value
            mem.flush()  # Write to hard disk

    def __len__(self):
        mem = ShareableList(name=self.name + 'shape')  # Shape
        size = mem[0]  # Batch dim
        mem.shm.close()
        return size
