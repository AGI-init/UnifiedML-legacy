# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import json
import re
import sys
from typing import MutableSequence
import glob
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import matplotlib.pyplot as plt
from matplotlib import ticker, dates
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def plot(path, plot_experiments=None, plot_agents=None, plot_suites=None, plot_tasks=None, steps=np.inf,
         write_tabular=False, plot_bar=True, plot_train=False, title='UnifiedML', x_axis='Step', verbose=False):

    path = Path(path + f'/{"Train" if plot_train else "Eval"}')
    path.mkdir(parents=True, exist_ok=True)

    # Make sure non empty and lists, and gather names
    empty = True
    specs = [plot_experiments, plot_agents, plot_suites, plot_tasks]
    plot_name = ''
    for i, spec in enumerate(specs):
        if spec is not None:
            empty = False
            if not isinstance(spec, MutableSequence):
                specs[i] = [spec]
            # Plot name
            plot_name += "_".join(specs[i] if i == 0 or len(specs[i]) < 5 else (specs[i][:5] + ['etc'])) + '_'
    plot_name = plot_name.strip('.')
    if empty:
        return

    # Style

    # RdYlBu, Set1, Set2, Set3, gist_stern, icefire, tab10_r, Dark2
    possible_palettes = ['Accent', 'RdYlBu', 'Set1', 'Set2', 'Set3', 'gist_stern', 'icefire', 'tab10_r', 'Dark2']
    # Note: finite number of color palettes: could error out if try to plot a billion tasks in one figure
    palette_colors = sum([sns.color_palette(palette) for palette in possible_palettes], [])

    sns.set_theme(font_scale=0.7,
                  rc={
                      'legend.loc': 'lower right', 'figure.dpi': 400,
                      'legend.fontsize': 5.5, 'legend.title_fontsize': 5.5,
                      # 'axes.titlesize': 4, 'axes.labelsize': 4, 'font.size': 4,
                      # 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                      # 'figure.titlesize': 4
                  })

    # All CSVs from path, recursive
    csv_names = glob.glob('./Benchmarking/*/*/*/*.csv', recursive=True)

    csv_list = []
    # max_csv_list = []  # Unused
    found_suite_tasks = set()
    found_suites = set()
    min_steps = steps

    # Data recollection/parsing
    for csv_name in csv_names:
        # Parse files
        experiment, agent, suite, task_seed_eval = csv_name.split('/')[2:]
        task_seed = task_seed_eval.split('_')
        suite_task, seed, eval = '_'.join(task_seed[:-2]), task_seed[-2], task_seed[-1].replace('.csv', '')

        # Map suite names to properly-cased names
        suite = {k.lower(): k for k in ['Atari', 'DMC', 'Classify']}[suite.lower()]

        # Whether to include this CSV
        include = True

        if (plot_train is (eval.lower() == 'eval')) or eval.lower() == 'seed':
            include = False

        datums = [experiment, agent, suite.lower(), suite_task]
        for i, spec in enumerate(specs):
            if spec is not None and not re.match('^(%s)+$' % '|'.join(spec).replace('(', r'\(').replace(
                    ')', r'\)').replace('+', r'\+'), datums[i], re.IGNORECASE):
                if i == 3 and re.match('^.*(%s)+$' % '|'.join(spec).replace('(', r'\(').replace(
                        ')', r'\)').replace('+', r'\+'), datums[i], re.IGNORECASE):
                    break
                include = False

        if not include:
            continue

        # Add CSV
        csv = pd.read_csv(csv_name)

        length = int(csv['step'].max())
        if length == 0:
            continue

        # TODO assumes all step brackets are shared
        # Min number of steps  TODO per suite, task
        min_steps = min(min_steps, length)

        found_suite_task = suite_task + ' (' + suite + ')'
        csv['Agent'] = agent + ' (' + experiment + ')'
        csv['Suite'] = suite
        csv['Task'] = found_suite_task

        if verbose and length < steps != np.inf:
            print(f'[Experiment {experiment} Agent {agent} Suite {suite} Task {suite_task} Seed {seed}] '
                  f'has {length} steps.')

        # Rolling max per run (as in CURL, SUNRISE) This was critiqued heavily in https://arxiv.org/pdf/2108.13264.pdf
        # max_csv = csv.copy()
        # max_csv['reward'] = max_csv[['reward', 'step']].rolling(length, min_periods=1, on='step').max()['reward']

        csv_list.append(csv)
        # max_csv_list.append(max_csv)
        found_suite_tasks.update({found_suite_task})
        found_suites.update({suite})

    # Non-empty check
    if len(csv_list) == 0:
        return

    df = pd.concat(csv_list, ignore_index=True)
    # max_df = pd.concat(max_csv_list, ignore_index=True)  # Unused
    found_suite_tasks = np.sort(list(found_suite_tasks))

    tabular_mean = {}
    tabular_median = {}
    tabular_normalized_mean = {}
    tabular_normalized_median = {}

    universal_hue_order, handles = np.sort(df.Agent.unique()), {}
    palette = {agent: color for agent, color in zip(universal_hue_order, palette_colors[:len(universal_hue_order)])}

    x_axis = x_axis.capitalize()

    # PLOTTING (tasks)

    # Dynamically compute num columns/rows
    num_rows = int(np.floor(np.sqrt(len(found_suite_tasks))))
    while len(found_suite_tasks) % num_rows != 0:
        num_rows -= 1
    num_cols = len(found_suite_tasks) // num_rows

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 3 * num_rows))

    # Title
    if title is not None:
        fig.suptitle(title)

    # Plot tasks
    for i, suite_task in enumerate(found_suite_tasks):
        task_data = df[df['Task'] == suite_task]

        # Capitalize column names
        task_data.columns = [' '.join([c_name.capitalize() for c_name in col_name.split('_')])
                             for col_name in task_data.columns]

        if steps < np.inf:
            task_data = task_data[task_data['Step'] <= steps]

        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 and num_cols > 1 else axs[col] if num_cols > 1 \
            else axs[row] if num_rows > 1 else axs

        # Format title
        ax_title = ' '.join([task_name[0].upper() + task_name[1:] for task_name in suite_task.split('_')])

        suite = ax_title.split('(')[1].split(')')[0]
        task = ax_title.split(' (')[0]

        _x_axis = x_axis if x_axis in task_data.columns else 'Step'
        y_axis = 'Accuracy' if 'classify' in suite.lower() else 'Reward'

        if _x_axis == 'Time':
            task_data['Time'] = pd.to_datetime(task_data['Time'], unit='s')

        if write_tabular or plot_bar:
            # Aggregate tabular data over all seeds/runs
            for agent in task_data.Agent.unique():
                for tabular in [tabular_mean, tabular_median]:
                    if agent not in tabular:
                        tabular[agent] = {}
                    if suite not in tabular[agent]:
                        tabular[agent][suite] = {}
                scores = task_data.loc[(task_data['Step'] == min_steps) & (task_data['Agent'] == agent), y_axis]
                tabular_mean[agent][suite][task] = scores.mean()
                tabular_median[agent][suite][task] = scores.median()
                for t in low:
                    if t.lower() in suite_task.lower():
                        for tabular in [tabular_normalized_mean, tabular_normalized_median]:
                            if agent not in tabular:
                                tabular[agent] = {}
                            if suite not in tabular[agent]:
                                tabular[agent][suite] = {}
                        normalized = (scores - low[t]) / (high[t] - low[t])
                        tabular_normalized_mean[agent][suite][task] = normalized.mean()
                        tabular_normalized_median[agent][suite][task] = normalized.median()
                        break

        # No need to show Agent in legend if all same
        short_palette = palette
        if len(task_data.Agent.str.split('(').str[0].unique()) == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                task_data['Agent'] = task_data.Agent.str.split('(').str[1:].str.join('(').str.split(')').str[:-1].str.join(')')
                short_palette = {')'.join('('.join(agent.split('(')[1:]).split(')')[:-1]): palette[agent] for agent in palette}

        hue_order = np.sort(task_data.Agent.unique())
        sns.lineplot(x=_x_axis, y=y_axis, data=task_data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax,
                     palette=short_palette
                     )
        ax.set_title(f'{ax_title}')

        if _x_axis == 'Time':
            ax.set_xlabel("Time (h)")
            ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
            # For now, group x axis into bins only for time
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        if 'classify' in suite.lower():
            ax.set_ybound(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

        # Legend in subplots
        ax.legend(frameon=False).set_title(None)

        ax.tick_params(axis='x', rotation=20)

        # Legend next to subplots
        # ax.legend(loc=2, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, frameon=False).set_title('Agent')

        # Data for universal legend (Note: need to debug if not showing Agent)
        # handle, label = ax.get_legend_handles_labels()
        # handles.update({l: h for l, h in zip(label, handle)})
        # ax.legend().remove()

    # Universal legend
    # axs[num_cols - 1].legend([handles[label] for label in hue_order], hue_order, loc=2, bbox_to_anchor=(1.05, 1.05),
    #                          borderaxespad=0, frameon=False).set_title('Agent')

    plt.tight_layout()
    plt.savefig(path / (plot_name + 'Tasks.png'))

    plt.close()

    # PLOTTING (suites)

    num_cols = len(found_suites)

    # Create subplots
    fig, axs = plt.subplots(1, num_cols, figsize=(4.5 * num_cols, 3))

    # Title
    if title is not None:
        fig.suptitle(title)

    # Sort suites
    found_suites = [found for s in ['Atari', 'DMC', 'Classify'] for found in found_suites if s in found]

    # Plot suites
    for col, suite in enumerate(found_suites):
        task_data = df[df['Suite'] == suite]

        # Capitalize column names
        task_data.columns = [' '.join([c_name.capitalize() for c_name in col_name.split('_')])
                             for col_name in task_data.columns]

        if steps < np.inf:
            task_data = task_data[task_data['Step'] <= steps]

        _x_axis = x_axis if x_axis in task_data.columns else 'Step'
        y_axis = 'Accuracy' if 'classify' in suite.lower() else 'Reward'

        if _x_axis == 'Time':
            task_data['Time'] = pd.to_datetime(task_data['Time'], unit='s')

        # No need to show Agent in legend if all same
        short_palette = palette
        if len(task_data.Agent.str.split('(').str[0].unique()) == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                task_data['Agent'] = task_data.Agent.str.split('(').str[1:].str.join('(').str.split(')').str[:-1].str.join(')')
                short_palette = {')'.join('('.join(agent.split('(')[1:]).split(')')[:-1]): palette[agent] for agent in palette}

        # High-low-normalize
        for suite_task in task_data.Task.unique():
            for t in low:
                if t.lower() in suite_task.lower():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=SettingWithCopyWarning)

                        task_data.loc[task_data['Task'] == suite_task, y_axis] -= low[t]
                        task_data.loc[task_data['Task'] == suite_task, y_axis] /= high[t] - low[t]
                        continue

        ax = axs[col] if num_cols > 1 else axs

        hue_order = np.sort(task_data.Agent.unique())
        sns.lineplot(x=_x_axis, y=y_axis, data=task_data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax,
                     palette=short_palette
                     )
        ax.set_title(f'{suite}')

        if _x_axis == 'Time':
            ax.set_xlabel("Time (h)")
            ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
            # For now, group x axis into bins only for time
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        if suite.lower() == 'atari':
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Human-Normalized Score')
        elif suite.lower() == 'dmc':
            ax.set_ybound(0, 1000)
        elif suite.lower() == 'classify':
            ax.set_ybound(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

        # Legend in subplots
        ax.legend(frameon=False).set_title(None)

        ax.tick_params(axis='x', rotation=20)

        # Legend next to subplots
        # ax.legend(loc=2, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, frameon=False).set_title('Agent')

        # ax.legend().remove()

    # Universal legend
    # axs[num_cols - 1].legend([handles[label] for label in hue_order], hue_order, loc=2, bbox_to_anchor=(1.05, 1.05),
    #                          borderaxespad=0, frameon=False).set_title('Agent')

    plt.tight_layout()
    plt.savefig(path / (plot_name + 'Suites.png'))

    plt.close()

    # Tabular data
    if write_tabular:
        f = open(path / (plot_name + f'{int(min_steps)}-Steps_Tabular.json'), "w")  # TODO name after steps if provided
        tabular_data = {'Mean': tabular_mean,
                        'Median': tabular_median,
                        'Normalized Mean': tabular_normalized_mean,
                        'Normalized Median': tabular_normalized_median}
        # Aggregating across suites
        for agg_name, agg in zip(['Mean', 'Median'], [np.mean, np.median]):
            for name, tabular in zip(['Mean', 'Median', 'Normalized-Mean', 'Normalized-Median'],
                                     [tabular_mean, tabular_median,
                                      tabular_normalized_mean, tabular_normalized_median]):
                tabular_data.update({
                    f'{agg_name} {name}': {
                        agent: {
                            suite:
                                agg([val for val in tabular[agent][suite].values()])
                            for suite in tabular[agent]}
                        for agent in tabular}
                })
        json.dump(tabular_data, f, indent=2)
        f.close()

    # Consistent x axis across all tasks for bar plot since tabular data only records w.r.t. min step
    min_time = df.loc[df['step'] == min_steps, x_axis.lower()].unique()
    if len(min_time) > 1:
        x_axis = 'Step'
        min_time = min_steps
    else:
        min_time = min_time[0]

    # Bar plot
    if plot_bar:
        bar_data = {suite_name: {'Task': [], 'Median': [], 'Agent': []} for suite_name in found_suites}
        for agent in tabular_median:
            for suite in tabular_median[agent]:
                for task in tabular_median[agent][suite]:
                    median = tabular_median
                    for t in low:
                        if t.lower() == suite.lower() or t.lower() == task.lower():
                            median = tabular_normalized_median
                            break
                    bar_data[suite]['Task'].append(task)
                    bar_data[suite]['Median'].append(median[agent][suite][task])
                    bar_data[suite]['Agent'].append(agent)

        # Max agents for a task
        max_agents = max([len(set([bar_data[suite]['Agent'][i] for i, _ in enumerate(bar_data[suite]['Agent'])
                                   if bar_data[suite]['Task'][i] == task])) for suite in bar_data
                          for task in set(bar_data[suite]['Task'])])

        # Create subplots
        fig, axs = plt.subplots(1, num_cols, figsize=(1.5 * max(max_agents, 3) * num_cols, 3))

        # Title
        if title is not None:
            fig.suptitle(title)

        for col, suite in enumerate(bar_data):
            task_data = pd.DataFrame(bar_data[suite])

            ax = axs[col] if num_cols > 1 else axs

            # No need to show Agent in legend if all same
            short_palette = palette
            if len(task_data.Agent.str.split('(').str[0].unique()) == 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                    task_data['Agent'] = task_data.Agent.str.split('(').str[1:].str.join('(').str.split(')').str[:-1].str.join(')')
                    short_palette = {')'.join('('.join(agent.split('(')[1:]).split(')')[:-1]): palette[agent] for agent in palette}

            hue_order = np.sort(task_data.Agent.unique())
            sns.barplot(x='Task', y='Median', ci='sd', hue='Agent', data=task_data, ax=ax, hue_order=hue_order,
                        palette=short_palette
                        )

            if x_axis.lower() == 'time':
                time_str = pd.to_datetime(min_time, unit='s').strftime('%H:%M:%S')
                ax.set_title(f'{suite} (@{time_str}h)')
            else:
                ax.set_title(f'{suite} (@{min_time:.0f} {x_axis}s)')

            if suite.lower() == 'atari':
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Median Human-Normalized')
            elif suite.lower() == 'dmc':
                ax.set_ybound(0, 1000)
                ax.set_ylabel('Median Reward')
            elif suite.lower() == 'classify':
                ax.set_ybound(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate('{:.0f}'.format(height) if suite.lower() == 'dmc' else f'{height:.0%}',
                            (x + width/2, y + height), ha='center', size=max(min(24 * width, 7), 5),  # No max(keep, 5)?
                            # color='#498057'
                            # color='#3b423d'
                            )

            ax.tick_params(axis='x', rotation=20)
            ax.set(xlabel=None)

            # Legend in subplots
            # ax.legend(frameon=False).set_title(None)

            # Legend next to subplots
            ax.legend(loc=2, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, frameon=False).set_title('Agent')

            # ax.legend().remove()

        # Universal legend
        # axs[num_cols - 1].legend([handles[label] for label in hue_order], hue_order, loc=2, bbox_to_anchor=(1.05, 1.05),
        #                          borderaxespad=0, frameon=False).set_title('Agent')

        plt.tight_layout()
        plt.savefig(path / (plot_name + 'Bar.png'))

        plt.close()

# Lows and highs for normalization

atari_random = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
}
atari_human = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}

low = {**atari_random}
high = {**atari_human}


if __name__ == "__main__":

    @hydra.main(config_path='Hyperparams', config_name='args')
    def main(args):
        OmegaConf.set_struct(args, False)
        del args.plotting['_target_']
        if 'path' not in sys_args:
            if isinstance(args.plotting.plot_experiments, str):
                args.plotting.plot_experiments = [args.plotting.plot_experiments]
            args.plotting.path = f"./Benchmarking/{'_'.join(args.plotting.plot_experiments)}/Plots"
        if 'steps' not in sys_args:
            args.plotting.steps = np.inf
        plot(**args.plotting)

    # Format path names
    # e.g. Checkpoints/Agents.DQNAgent -> Checkpoints/DQNAgent
    OmegaConf.register_new_resolver("format", lambda name: name.split('.')[-1])

    sys_args = []
    for i in range(1, len(sys.argv)):
        sys_args.append(sys.argv[i].split('=')[0].strip('"').strip("'"))
        sys.argv[i] = 'plotting.' + sys.argv[i] if sys.argv[i][0] != "'" and sys.argv[i][0] != '"' \
            else sys.argv[i][0] + 'plotting.' + sys.argv[i][1:]

    main()
