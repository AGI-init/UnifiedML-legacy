atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]

if __name__ == '__main__':
    out = ""
    for task in atari_tasks:
        f = open(f"./{task.lower()}.yaml", "w")
        f.write(r"""defaults:
      - _self_
    
Env: Datasets.Suites.Atari.Atari
suite: atari
task_name: {}
discrete: true
action_repeat: 4
truncate_episode_steps: 250
nstep: 10
frame_stack: 3
train_steps: 500000
stddev_schedule: 'linear(1.0,0.1,20000)'

# Atari has two augmentations
aug:
    _target_: Utils.Sequential
    _targets_: [RandomShiftsAug, IntensityAug]
    pad: 4
    noise: 0.05
""".format(task))
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
