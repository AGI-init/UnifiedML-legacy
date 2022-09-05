# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import csv
import datetime
import re
from pathlib import Path
from termcolor import colored

import numpy as np

import torch


def shorthand(log_name):
    return ''.join([s[0].upper() for s in re.split('_|[ ]', log_name)] if len(log_name) > 3 else log_name.upper())


def format(log, log_name):
    k = shorthand(log_name)

    if 'time' in log_name.lower():
        log = str(datetime.timedelta(seconds=int(log)))
        return f'{k}: {log}'
    elif float(log).is_integer():
        log = int(log)
        return f'{k}: {log}'
    else:
        return f'{k}: {log:.04f}'


class Logger:
    def __init__(self, task, seed, generate=False, path='.', aggregation='mean', wandb=False):

        self.path = path
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.task = task
        self.seed = seed
        self.generate = generate

        self.logs = {}

        self.aggregation = aggregation  # mean, median, last, max, min, or sum
        self.default_aggregations = {'step': np.ma.max, 'frame': np.ma.max, 'episode': np.ma.max, 'epoch': np.ma.max,
                                     'time': np.ma.max, 'fps': np.ma.mean}

        self.wandb = 'uninitialized' if wandb \
            else None

    def log(self, log=None, name="Logs", dump=False):
        if log is not None:

            if name not in self.logs:
                self.logs[name] = {}

            logs = self.logs[name]

            for log_name, item in log.items():
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                logs[log_name] = logs[log_name] + [item] if log_name in logs else [item]

        if dump:
            self.dump_logs(name)

    def dump_logs(self, name=None):
        if name is None:
            # Iterate through all logs
            for name in self.logs:
                for log_name in self.logs[name]:
                    agg = self.aggregate(log_name)
                    self.logs[name][log_name] = agg(self.logs[name][log_name])
                self._dump_logs(self.logs[name], name=name)
                del self.logs[name]
        else:
            # Iterate through just the named log
            if name not in self.logs:
                return
            for log_name in self.logs[name]:
                agg = self.aggregate(log_name)
                self.logs[name][log_name] = agg(self.logs[name][log_name])
            self._dump_logs(self.logs[name], name=name)
            self.logs[name] = {}
            del self.logs[name]

    # Aggregate list of scalars or batched-values of arbitrary lengths
    def aggregate(self, log_name):
        def last(data):
            data = np.array(data).flat
            return data[len(data) - 1]

        agg = self.default_aggregations.get(log_name,
                                            np.ma.mean if self.aggregation == 'mean'
                                            else np.ma.median if self.aggregation == 'median'
                                            else last if self.aggregation == 'last'
                                            else np.ma.max if self.aggregation == 'max'
                                            else np.ma.min if self.aggregation == 'min'
                                            else np.ma.sum)

        def size_agnostic_agg(stats):
            stats = [(stat,) if np.isscalar(stat) else stat.flatten() for stat in stats]

            masked = np.ma.empty((len(stats), max(map(len, stats))))
            masked.mask = True
            for m, stat in zip(masked, stats):
                m[:len(stat)] = stat
            return agg(masked)

        return agg if agg == last else size_agnostic_agg

    def _dump_logs(self, logs, name):
        self.dump_to_console(logs, name=name)
        self.dump_to_csv(logs, name=name)
        if self.wandb is not None:
            self.log_wandb(logs, name=name)

    def dump_to_console(self, logs, name):
        name = colored(name, 'yellow' if name.lower() == 'train' else 'green' if name.lower() == 'eval' else None,
                       attrs=['dark'] if name.lower() == 'seed' else None)
        pieces = [f'| {name: <14}']
        for log_name, log in logs.items():
            pieces.append(format(log, log_name))
        print(' | '.join(pieces))

    def remove_old_entries(self, logs, file_name):
        rows = []
        with file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= logs['step']:
                    break
                rows.append(row)
        with file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=logs.keys(),
                                    extrasaction='ignore',
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dump_to_csv(self, logs, name):
        logs = dict(logs)

        assert 'step' in logs

        if self.generate:
            name = 'Generate_' + name

        file_name = Path(self.path) / f'{self.task}_{self.seed}_{name}.csv'

        write_header = True
        if file_name.exists():
            write_header = False
            self.remove_old_entries(logs, file_name)

        file = file_name.open('a')
        writer = csv.DictWriter(file,
                                fieldnames=logs.keys(),
                                restval=0.0)
        if write_header:
            writer.writeheader()

        writer.writerow(logs)
        file.flush()

    def log_wandb(self, logs, name):
        if self.wandb == 'uninitialized':
            import wandb

            experiment, agent, suite = self.path.split('/')[2:5]

            if self.generate:
                agent = 'Generate_' + agent

            wandb.init(project=experiment, name=f'{agent}_{suite}_{self.task}_{self.seed}', dir=self.path)

            for file in ['', '*/', '*/*/', '*/*/*/']:
                try:
                    wandb.save(f'./Hyperparams/{file}*.yaml')
                except Exception:
                    pass

            self.wandb = wandb

        measure = 'reward' if 'reward' in logs else 'accuracy'
        if measure in logs:
            logs[f'{measure} ({name})'] = logs.pop(f'{measure}')

        self.wandb.log(logs, step=int(logs['step']))
