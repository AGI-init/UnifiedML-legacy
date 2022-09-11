![alt text](https://i.imgur.com/rjw4eFg.png)

[comment]: <> ([![]&#40;https://img.shields.io/badge/State_of_the_Art-Data--Efficient_RL-blue.svg??style=flat&logo=google-analytics&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/Modern,_Academic--Standard-Image_Classification-blue.svg??style=flat&logo=instatus&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/Technically--Working-Generative_Modeling-blue.svg??style=flat&logo=angellist&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/In--Progress-Multi_Modalities-red.svg??style=flat&logo=plex&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/Unified_in_one_Framework-Seamless,_General.svg??style=flat&logo=immer&#41;]&#40;&#41;)

### Quick Links

- [Setup](#wrench-setting-up)

- [Examples](#mag-full-tutorials)

- [Agents and performances](#bar_chart-agents--performances)

[comment]: <> (- [How Is This Possible?]&#40;#interrobang-how-is-this-possible&#41;)

[comment]: <> (- [Contributing]&#40;#people_holding_hands-contributing&#41; &#40;Best way: [please donate]&#40;&#41;!&#41;)

[comment]: <> (# )

[comment]: <> (> A library for **reinforcement learning**, **supervised learning**, and **generative modeling**. And eventually, full-general intelligence.)

# :runner: Running The Code

To start a train session, once [installed](#wrench-setting-up):

```console
python Run.py
```

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, generated images, and videos are automatically stored in: ```./Benchmarking```.

![alt text](https://i.imgur.com/2jhOPib.gif)

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, à la carte!

Drink up! :beers:

# :pen: Paper & Citing

For detailed documentation, [see our :scroll:](https://arxiv.com).

[comment]: <> ([![arXiv]&#40;https://img.shields.io/badge/arXiv-<NUMBER>.<NUMBER>-b31b1b.svg?style=flat&#41;]&#40;https://arxiv.org/abs/<NUMBER>.<NUMBER>&#41;)

[comment]: <> (```)

[comment]: <> (@inproceedings{cool,)

[comment]: <> (  title={bla},)

[comment]: <> (  author={Sam Lerman and Chenliang Xu},)

[comment]: <> (  booktitle={bla},)

[comment]: <> (  year={2022},)

[comment]: <> (  url={https://openreview.net})

[comment]: <> (})

[comment]: <> (```)

```bibtex
@article{cool,
  title   = {UnifiedML: A Unified Framework For Intelligence Training},
  author  = {Lerman, Sam and Xu, Chenliang},
  journal = {arXiv preprint arXiv:2203.08913},
  year    = {2022}
}
```

[comment]: <> (```bibtex)

[comment]: <> (@inproceedings{UML,)

[comment]: <> (  title={UnifiedML: A Unified Framework For Intelligence Training},)

[comment]: <> (  author={Lerman, Sam and Xu, Chenliang},)

[comment]: <> (  booktitle={booktitle},)

[comment]: <> (  year={2022},)

[comment]: <> (  url={https://openreview.net})

[comment]: <> (})

[comment]: <> (```)

If you use this work, please give us a star :star: and be sure to cite the above!

An acknowledgment to [Denis Yarats](https://github.com/denisyarats), whose excellent [DrQV2 repo](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

# :open_umbrella: Unified Learning?

Yes.

All agents support discrete and continuous control, classification, and generative modeling.

[comment]: <> (All agents and even tasks support discrete and continuous control, online and offline RL, imitation learning, classification, regression, and generative modeling.)

See example scripts of various configurations [below](#mag-full-tutorials).

# :wrench: Setting Up

Let's get to business.

## 1. Clone The Repo

```console
git clone git@github.com:agi-init/UnifiedML.git
cd UnifiedML
```

## 2. Gemme Some Dependencies

All dependencies can be installed via [Conda](https://docs.conda.io/en/latest/miniconda.html):

```console
conda env create --name ML --file=Conda.yml
```

[comment]: <> (For GPU support, you may have to [pip install Pytorch]&#40;&#41; depending on your CUDA version.)

[comment]: <> (> CUDA 11.6 example: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`  )

[comment]: <> (:tada:)

[comment]: <> (:tada: Got me some dependencies :tada:)

[comment]: <> (For CUDA 11+, also try:)

[comment]: <> (```console)

[comment]: <> (# 11.3)

[comment]: <> (pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113)

[comment]: <> (# 11.6)

[comment]: <> (pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116)

[comment]: <> (```)

[comment]: <> (See [here]&#40;https://pytorch.org/get-started/locally/&#41;.)

[comment]: <> (as per the [Pytorch installation instructions]&#40;https://pytorch.org/get-started/locally/&#41;.)

## 3. Activate Your Conda Env.

```console
conda activate ML
```

#

> &#9432; Depending on your CUDA version, you may need to additionally install Pytorch with CUDA via pip from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) after activating your Conda environment.
>
> For example, for CUDA 11.6:
> ```console
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
> ```

[comment]: <> (For GPU support, you may need to install Pytorch with CUDA from https://pytorch.org/get-started/locally/.)

[comment]: <> (```console)

[comment]: <> (conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch)

[comment]: <> (```)

# :joystick: Installing The Suites

## 1. Atari Arcade

<p align="left">
<img src="https://i.imgur.com/ppm4LJw.jpg" width="320">
<br><i>A collection of retro Atari games.</i>
</p>

You can install via ```AutoROM``` if you accept the license.

```console
pip install autorom
AutoROM --accept-license
```
Then:
```console
mkdir ./Datasets/Suites/Atari_ROMS
AutoROM --install-dir ./Datasets/Suites/Atari_ROMS
ale-import-roms ./Datasets/Suites/Atari_ROMS
```

## 2. DeepMind Control

Comes pre-installed! For any issues, consult the [DMC repo](https://github.com/deepmind/dm_control).

<p align="left">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rAai4QzcYbs" target="_blank"><i>:arrow_forward: Click to play</i></a><br>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rAai4QzcYbs" target="_blank">
<img src="https://i.imgur.com/vzNmMMQ.png" alt="Play video" width="310" />
</a>
<br><i>Video of different tasks in action.</i>
</p>

## 3. Classify

<p align="left">

[comment]: <> (<img src="https://i.imgur.com/F633xwk.png" width="320">)

[comment]: <> (<br><i>Alpaca or llama? Donkey or mule? Roses or kale? — iNaturalist</i><br><br>)
<img src="https://i.imgur.com/N1st6uO.png" width="320">
<br><i>Eight different ladybug species in the iNaturalist dataset.</i>

[comment]: <> (<br><br>)

[comment]: <> (<img src="https://i.imgur.com/etoaz2b.png" width="320">)

[comment]: <> (<br><i>Samples of images from the CIFAR-100 dataset.</i>)

[comment]: <> (<br><br><img src="https://i.imgur.com/E1v1jvm.jpg" width="320">)

[comment]: <> (<br><i>Samples of images from the Tiny-ImageNet dataset.</i>)
</p>

[comment]: <> (Comes preinstalled.)

[comment]: <> (No additional preparation needed. All datasets download automatically.)

[All datasets](Hyperparams/task/classify) come ready-to-use :white_check_mark:

[comment]: <> (<hr class="solid">)

That's it. 

[comment]: <> (~)

> :bulb: Train Atari example: ```python Run.py task=atari/mspacman```
>
> :bulb: Train DMC example: ```python Run.py task=dmc/cheetah_run```
>
> :bulb: Train Classify example: ```python Run.py task=classify/mnist```

[comment]: <> (:point_right:)


[comment]: <> (> :bulb: **Train Atari example**: ```python Run.py task=atari/mspacman```)

[comment]: <> (>)

[comment]: <> (> :bulb: **Train DMC example**: ```python Run.py task=dmc/cheetah_run```)

[comment]: <> (>)

[comment]: <> (> :bulb: **Train Classify example**: ```python Run.py task=classify/mnist```)

[comment]: <> (<hr>)


[comment]: <> (> > :bulb: **Train GAN example**: ```python Run.py task=classify/mnist generate=true```)
 
[comment]: <> (<hr class="solid">)

[comment]: <> (All datasets come preinstalled :white_check_mark:)

[comment]: <> (#)

# :file_cabinet: Key files

```Run.py``` handles learning and evaluation loops, saving, distributed training, logging, plotting.

```Environment.py``` handles rollouts.

```./Agents``` contains self-contained agents.

[comment]: <> (```Run.py``` handles learning and evaluation loops, saving, distributed training, logging, plotting :play_or_pause_button: :repeat:)

[comment]: <> (:earth_africa: ```Environment.py``` handles rollouts)

[comment]: <> (:robot: ```./Agents``` contains self-contained agents :alien: :space_invader:)
 
[comment]: <> (<hr>)

#

# :mag: Full Tutorials

### RL

<details>
<summary>
:mag: <i>Click to interact</i>
</summary>
<br>

[comment]: <> (* Achieves [top scores]&#40;#bar_chart-agents--performances&#41; in data-efficient RL across Atari and DMC.)

[comment]: <> (❖)
**Train a** [```DQN Agent```](Agents/DQN.py) **to play Ms. Pac-Man**:

```console

python Run.py task=atari/mspacman

```

[comment]: <> (* This agent is the library's default &#40;```Agent=```[```Agents.DQNAgent```]&#40;Agents/DQN.py&#41;&#41;.)
* This agent is the library's default (```Agent=Agents.DQNAgent```).
* Our implementation expands on [ensemble Q-learning](https://arxiv.org/abs/1802.09477v3) with [data regularization](https://arxiv.org/pdf/2004.13649.pdf) and [Soft-DQN](https://arxiv.org/pdf/2007.14430.pdf) ([```here```](Losses/QLearning.py#L43)).
* [Original Nature DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).

——❖——

[comment]: <> (Maybe put below in collapsed)

**Humanoid from pixels** with [```DrQV2 Agent```](Agents/DrQV2.py), [a state of the art algorithm for continuous control from images](https://arxiv.org/abs/2107.09645):
```console
python Run.py Agent=Agents.DrQV2Agent task=dmc/humanoid_walk
```

[comment]: <> (❖)

**For self-supervision**, [```SPR Agent```](Agents/SPR.py) in Atari:
```console
python Run.py Agent=Agents.SPRAgent task=atari/boxing
```

The [original SPR paper](https://arxiv.org/abs/2007.05929) used a [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) backbone. We use a weaker [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) backbone for now for the sake of simplicity.

[comment]: <> ([AC2]&#40;paper&#41; Agent in DMC:)

**When in doubt**: our [```AC2 Agent```](Agents/Lermanbots/AC2.py). Pretty much the best of all worlds among this collection of algorithms.
```console
python Run.py Agent=Agents.AC2Agent task=dmc/walker_walk +agent.depth=5 +agent.num_actors=5 +agent.num_critics=5 nstep=5
```

```+agent.depth=5``` activates a self-supervisor to predict temporal dynamics for up to 5 timesteps ahead.

```+agent.num_actors=5 +agent.num_critics=5``` activates actor-critic ensembling.

[comment]: <> (——❖——)

[comment]: <> (#)

[comment]: <> (Collapse up to here, maybe remove video path/gif)

⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽

[comment]: <> (——❖——)

[comment]: <> (✞———————❖———————✞)

As of now, all agents are *visual*, that is, they observe pixel-based inputs.

Save videos with ```log_video=true```.

:clapper: :movie_camera: -> ```Benchmarking/<experiment>/<agent>/<suite>/<task>_<seed>_Video_Image/```

<p>
<img src="https://qiita-image-store.s3.amazonaws.com/0/3180/8c235a00-cd55-41a2-a605-a4a2e9b0240f.gif" data-canonical-src="https://qiita-image-store.s3.amazonaws.com/0/3180/8c235a00-cd55-41a2-a605-a4a2e9b0240f.gif" width="64" height="84" />
</p>

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (Maybe collapse all this up above, putting discretization note first)

Check out [args.yaml](Hyperparams/args.yaml) for the full array of configurable options available, including
* N-step rewards (```nstep=```)
* Frame stack (```frame_stack=```)
* Action repeat (```action_repeat=```)
* & more, with [per-task](Hyperparams/task) defaults in ```/Hyperparams/task``` — please [share your hyperparams](https://github.com/agi-init/UnifiedML/discussions) if you discover new or better ones!

[comment]: <> (#)

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (✞———————❖———————✞)

[comment]: <> (—————)

[comment]: <> (Basic RL features are configurable:)

[comment]: <> (- N-step reward via ```nstep=```)

[comment]: <> (- Action repeat via ```action_repeat=```)

[comment]: <> (- Frame stack via ```frame_stack=```)

[comment]: <> (- Exploration schedule via ```'stddev_schedule= '```)

[comment]: <> (Or keep the loosely-optimized per-task defaults already specified in ```Hyperparams/task/```.)
  
[comment]: <> (Actor-Critic ensembling is also supported for some agents like ```AC2Agent```:)

[comment]: <> (- ```Agent=Agents.AC2Agent +agent.num_actors=```)

[comment]: <> (- ```Agent=Agents.AC2Agent +agent.num_critics=```)

[comment]: <> (✞———————❖———————✞)

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺)

[comment]: <> (<br>)

[comment]: <> (——❖——)

[comment]: <> (#)

&#9432; ***Experimental***: If you'd like to **discretize** a continuous domain, pass in ```discrete=true``` and specify the number of discrete bins per action dimension via ```num_actions=```. If you'd like to **continuous-ize** a discrete domain, pass in ```discrete=false```.

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (#)

[comment]: <> (<br>)

[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (Maybe remove all this)

#

[comment]: <> (——❖——)

> :bulb: *The below sections describe many features in other domains, but chances are those features will work in RL as well. For example, a cosine annealing learning rate schedule can be toggled with: ```lr_decay_epochs=100```. It will anneal per-episode rather than per-epoch. Different model architectures, image transforms, EMAs, and more are all supported across domains!*
> 
> The vast majority of this hasn't been tested outside of its respective domain (CV, RL, etc.), so the research opportunity is a lot!

[comment]: <> (#)
[comment]: <> (⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽)

[comment]: <> (Enjoy :thumbsup:)

[comment]: <> (More in-depth logs can be toggled with ```agent.log=true```.)

[comment]: <> (Options like ```nstep=```, ```action_repeat=```, ```frame_stack=``` let you customize the training further, as well as plenty of [other hyperparams]&#40;Hyperparams/args.yaml&#41;.)

</details>

### Classification 

<details>
<summary>
:mag: <i>Click to ascertain</i>
</summary>
<br>

CNN on MNIST:

```console
python Run.py task=classify/mnist 
```
[comment]: <> (Since this is *Unified*ML, there are a few noteworthy variations.)

*Note:* ```RL=false``` is the default for ```classify``` tasks. Keeps training at **standard** supervised-only classification.

**Variations**

Since this is *Unified*ML, there are a couple noteworthy variations. You can ignore these if you are only interested in standard classification via cross-entropy supervision only.

1. With ```RL=true```, an **augmented RL** update joins the supervised learning update $\text{s.t. } reward = -error$ (**experimental**).

2. Alternatively, and interestingly, ```supervise=false RL=true``` will *only* supervise via RL $reward = -error$. This is **pure-RL** training and actually works!

Classify environments can actually be great testbeds for certain RL problems since they give near-instant and clear performance feedback.

[comment]: <> (*Note:* ```RL=false``` sets training to standard supervised-only classification. Without ```RL=false```, an additional RL update joins the supervised learning update s.t. $reward = -error$.)

[comment]: <> (Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL $reward = -error$ &#40;**experimental**&#41;. This is pure-RL training and actually works.)

[comment]: <> (with a simple CNN and some small random crop transforms.)

[comment]: <> (*Note:* ```RL=false``` sets training to standard supervised-only classification.)

[comment]: <> (Without ```RL=false```, an **Augmented RL** update joins the supervised learning update s.t. $reward = -error$.)

[comment]: <> (**Pure-RL** Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL $reward = -error$ &#40;*experimental*&#41;. This is pure-RL training and actually works.)

[comment]: <> (The latent optimization could also be done over a learned parameter space as in POPLIN &#40;Wang and Ba, 2019&#41;, which lifts the domain of the optimization problem eq. &#40;1&#41; from Y to the parameter space of a fully-amortized neural network. This leverages the insight that the parameter space of over-parameterized neural networks can induce easier non-convex optimization problems than in the original space, which is also studied in Hoyer et al. &#40;2019&#41;.)

**Important features** 

Many popular features are unified in this library and generalized across RL/CV/generative domains, with more being added: 

* Evaluation with [exponential moving average (EMA)](https://arxiv.org/pdf/1803.05407.pdf) of params can be toggled with the ```ema=true``` flag; customize the decay rate with ```ema_decay=```. 
  
* See [Custom Architectures](#custom-architectures) for mix-and-matching custom or pre-defined (*e.g.* ViT, ResNet50) architectures via the command line syntax. 
  
* Different optimizations [can be configured](#custom-optimization) too.
  
* As well as [Custom Datasets](#custom-datasets). 

* Ensembling is supported for some agents (e.g., ```Agent=Agents.AC2Agent +agent.num_actors=```)
  
* Training with [weight decay](https://arxiv.org/abs/1711.05101) can be toggled via ```weight_decay=```. 
  
* A [cosine annealing learning rate schedule](https://arxiv.org/abs/1608.03983) can be applied for $N$ epochs (or episodes in RL) with ```lr_decay_epochs=```. 
  
* And [TorchVision transforms](https://pytorch.org/vision/stable/transforms.html) can be passed in as dicts via ```transform=```. 
  
For example,

```console
python Run.py task=classify/cifar10 ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{p:0.5}}" Eyes=Blocks.Architectures.ResNet18
```

The above returns a $93$% on CIFAR-10 with a ResNet18, which is pretty good. Changing datasets/architectures is as easy as modifying the corresponding parts ```task=``` and ```Eyes=``` of the above script.

And if you set ```supervise=false RL=true```, we get a $94$%... vis-à-vis pure-RL. 

[comment]: <> (Rollouts fill up data in an online fashion, piecemeal, until depletion &#40;all data is processed&#41; and gather metadata like past predictions, which may be useful for curriculum learning.)

[comment]: <> (Automatically toggles ```offline=true``` by default, but can be set to ```false``` if past predictions or "streaming" data is needed.)

This library is meant to be useful for academic research, and out of the box supports [many datasets](Hyperparams/task/classify), including 
* Tiny-ImageNet (```task=classify/tinyimagenet```), 
* iNaturalist, (```task=classify/inaturalist```),
* CIFAR-100 (```task=classify/cifar100```), 
* & [more](Hyperparams/task/classify), normalized and no manual preparation needed

</details>

### Offline RL

<details>
<summary>
:mag: <i>Click to recall</i>
</summary>
<br>

From a saved experience replay, sans additional rollouts:

```console
python Run.py task=atari/breakout offline=true
```

Assumes a replay [is saved](#saving).

Implicitly treats ```replay.load=true``` and ```replay.save=true```, and only does evaluation rollouts.

Is true by default for classification, where replays are automatically downloaded.

</details>

[comment]: <> (### Imitation Learning)

[comment]: <> (<details>)

[comment]: <> (<summary>)

[comment]: <> (:mag: <i>Click to recall</i>)

[comment]: <> (</summary>)

[comment]: <> (<br>)

[comment]: <> (The conversion to imitation is really simple. The action gets set as the label and is either argmaxed or one-hotted depending on whether the environment is discrete or continuous, or whether classifying &#40;overriding with ```classify=true```&#41; or doing regression &#40;```classify=false```&#41;.)

[comment]: <> (```console)

[comment]: <> (python Run.py task=atari/breakout imitate=true)

[comment]: <> (```)

[comment]: <> (Assumes a replay [is saved]&#40;#saving&#41; to load and imitate based on.)

[comment]: <> (Implicitly treats ```replay.load=true``` and ```replay.save=true```. The load path can of course be configured &#40;```replay.path```&#41;.)

[comment]: <> (</details>)

### Generative Modeling

<details>
<summary>
:mag: <i>Click to synthesize</i>
</summary>
<br>

Via the ```generate=true``` flag:
```console
python Run.py task=classify/mnist generate=true
```

<p align="left">
<img src="https://i.imgur.com/HEudCOX.png" width="180">
<br><i>Synthesized MNIST images, conjured up and imagined by a simple MLP.</i>
</p>

Saves to ```./Benchmarking/<experiment>/<Agent name>/<task>_<seed>_Video_Image/```.

Defaults can be easily modified with custom architectures or even datasets as elaborated in [Custom Architectures](#custom-architectures) and [Custom Datasets](#custom-dataset). Let's try the above with a CNN Discriminator:

```console
python Run.py task=classify/mnist generate=true Discriminator=CNN +agent.num_critics=1
```

```+agent.num_critics=1``` uses only a single Discriminator rather than ensembling as is done in RL. See [paper]() or [How Is This Possible?](#interrobang-how-is-this-possible) for more details on the unification between Critic and Discriminator. Not all Agents support custom critic ensembling, and those will default to 2.

Or a ResNet18:

```console
python Run.py task=classify/mnist generate=true Discriminator=Resnet18
```

Let's speed up training by turning off the default image augmentation, which is overkill anyway for this simple case:

```console
python Run.py task=classify/mnist generate=true Aug=Identity +agent.num_critics=1
```

```Aug=Identity``` substitutes the default random cropping image-augmentation with the Identity function, thereby disabling it.

Generative mode implicitly treats training as [offline](#offline-rl), and assumes a replay [is saved](#saving) that can be loaded. As long as a dataset is available or a replay has [been saved](#saving), ```generate=true``` will work for any defined visual task, making it a powerful hyper-parameter that can just work. For now, only visual (image) tasks are compatible. 

[comment]: <> (TODO: set defualts for generate in Run.py/Environment.py automatically)
Can even work with RL tasks (due to frame stack, the generated images are technically multi-frame videos).

```console
python Run.py task=atari/breakout generate=true
```

Make sure you have [saved a replay](#saving) that can be loaded before doing this.

</details>

[comment]: <> (ensemble could help this:)

[comment]: <> (Extensions. Analyzing and extending the amortization components has been a key development in AVI methods. Cremer et al. &#40;2018&#41; investigate suboptimality in these models are categorize it as coming from an amortization gap where the amortized model for eq. &#40;30&#41; does not properly solve it, or the approximation gap where the variational posterior is incapable of approximating the true distribution. Semi-amortization plays a crucial role in addressing the amortization gap and is explored in the semi-amortized VAE &#40;SAVAE&#41; by)

[comment]: <> (Kim et al. &#40;2018&#41; and iterative VAE &#40;IVAE&#41; by Marino et al. &#40;2018&#41;.)

### Saving
<details>
<summary>
:mag: <i>Click to load</i>
</summary>
<br>

**Agents** can be saved periodically or loaded with the ```save_per_steps=``` or ```load=true``` flags, and are automatically saved at end of training with ```save=true``` by default.

```console
python Run.py save_per_steps=100000 load=true
```

An **experience replay** can be saved or loaded with the ```replay.save=true``` or ```replay.load=true``` flags.

```console
python Run.py replay.save=true replay.load=true
```

Agents and replays save to ```./Checkpoints``` and ```./Datasets/ReplayBuffer``` respectively per *a unique experiment*, otherwise overriding.

*A unique experiment* is distinguished by the flags: ```experiment=```, ```Agent=```, ```task=```, and ```seed=```.

Replays also save uniquely w.r.t. a date-time. In case of multiple saved replays per a unique experiment, the most recent is loaded.

Careful, without ```replay.save=true``` a replay, whether new or loaded, will be deleted upon terminate, except for the default offline classification replays.

</details>

### Distributed

<details>
<summary>
:mag: <i>Click to disperse</i>
</summary>
<br>

The simplest way to do distributed training is to use the ```parallel=true``` flag,

```console
python Run.py parallel=true 
```

which automatically parallelizes the Encoder's "Eyes" across all visible GPUs. The Encoder is usually the most compute-intensive architectural portion.

To share whole agents across multiple parallel instances and/or machines,

<details>

<summary><i>Click to expand :open_book: </i></summary>

<br>

you can use the ```load_per_steps=``` flag.

For example, a data-collector agent and an update agent,

```console

python Run.py learn_per_steps=0 replay.save=true load_per_steps=1

```

```console

python Run.py offline=true replay.offline=false replay.save=true replay.load=true save_per_steps=2

```

in concurrent processes.

Since both use the same experiment name, they will save and load from the same agent and replay, thereby emulating distributed training. Just make sure the replay from the first script is created before launching the second script. **Highly experimental!**

Here is another example of distributed training, via shared replays:

```console
python Run.py replay.save=true 
```

Then, in a separate process, after that replay has been created:

```console
python Run.py replay.load=true replay.save=true 
```

[comment]: <> (It's a bit finicky; there are a few timing delicacies that I don't account for. I recommend to wait until at least 1 episode for the first script's replay to be created before launching the second script. This is not meant as a deployable means of distributed training. It just happens to work, incidentally, sort of.)

</details>

</details>

### Recipes

<details>
<summary>
:mag: <i>Learn to cook</i>
</summary>
<br>

```console
python Run.py Eyes=Sequential +eyes._targets_="[CNN, Transformer]" task=classify/mnist
```

```console
python Run.py task=classify/mnist Pool=Sequential +pool._targets_="[Transformer, AvgPool]" +pool.positional_encodings=false
```

```console
python Run.py task=classify/mnist Pool=Residual +pool.model=Transformer +pool.depth=2
```

```console
python Run.py task=classify/mnist Pool=Sequential +pool._targets_="[ChannelSwap, Residual]" +'pool.model="MLP(kwargs.input_shape[-1])"' +'pool.down_sample="MLP(input_shape=kwargs.input_shape[-1])"'
```

```console
python Run.py task=classify/mnist Pool=RN
```

```console
python Run.py task=classify/mnist Pool=Sequential +pool._targets_="[RN, AvgPool]"
```

```console
python Run.py task=classify/mnist Eyes=Perceiver +eyes.depths="[3, 3, 2]"  +eyes.num_tokens=128
```

```console
python Run.py task=classify/mnist Predictor=Perceiver +predictor.token_dim=32
```

```console
python Run.py task=classify/mnist Predictor=Perceiver train_steps=2
python Run.py task=dmc/cheetah_run Predictor=load +predictor.path=./Checkpoints/Exp/DQNAgent/classify/MNIST_1.pt +predictor.attr=actor.Pi_head +predictor.device=cpu save=false
```

```console
python Run.py task=classify/mnist Eyes=Identity Predictor=Perceiver +predictor.depths=10
```

[comment]: <> (Doesn't work:)

[comment]: <> (```console)

[comment]: <> (python Run.py experiment='Q-Learning-Target_expected+entropy_Intensity+Shift' Aug=Sequential '+aug._targets_="[IntensityAug&#40;0.05&#41;, RandomShiftsAug&#40;4&#41;]"')

[comment]: <> (```)

[comment]: <> (python Run.py                     task=classify/custom                     Dataset=XRD.XRD                     Aug=Identity                     Trunk=Identity                     Eyes=ViT                    batch_size=2                  Optim=SGD                     lr=1e-3                     standardize=false                     norm=false                     task_name='Soup-50-50_${dataset.num_classes}-Way'                     experiment='CNN_optim_SGD_batch_size_${batch_size}_lr_1e-3'                     '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'                     +'dataset.train_eval_splits=[1, 0.5]'                     +dataset.num_classes=7                     train_steps=5e5 num_workers=1  +eyes.depth=1)


```console
python Run.py Aug=Sequential +aug._targets_="[IntensityAug, RandomShiftsAug]" +aug.scale=0.05 aug.pad=4
```

</details>

### Custom Architectures

<details>
<summary>
:mag: <i>Click to construct</i>
</summary>
<br>

A rich and expressive command line syntax is available for selecting and customizing architectures such as those defined in [```./Blocks/Architectures```](Blocks/Architectures).

ResNet18 on CIFAR-10:

```console
python Run.py task=classify/cifar10 Eyes=ResNet18 
```

[comment]: <> (TODO: MiniViT, ViT)
Atari with ViT:

```console
python Run.py Eyes=ViT +eyes.patch_size=7
```

[comment]: <> (TODO: Eyes, Ears, etc. recipes -> hands)
Shorthands like ```Aug```, ```Eyes```, and ```Pool``` make it easy to plug and play custom architectures. All of an agent's architectural parts can be accessed, mixed, and matched with their [corresponding recipe shorthands](Hyperparams/args.yaml#L166).

Generally, the rule of thumb is capital names for paths to classes (such as ```Eyes=Blocks.Architectures.MLP```) and lowercase names for shortcuts to tinker with model args (such as ```+eyes.depth=1```).

Architectures imported in [Blocks/Architectures/\_\_init\_\_.py](Blocks/Architectures/__init__.py) can be accessed directly without need for entering their full paths, as in ```Eyes=ViT``` works just as well as ```Eyes=Blocks.Architectures.ViT```.


<details>
<summary><i>See more examples :open_book: </i></summary>
<br>

CIFAR-10 with ViT:

```console
python Run.py Eyes=ViT task=classify/cifar10 ema=true weight_decay=0.01 +eyes.depth=6 +eyes.out_channels=512 +eyes.mlp_hidden_dim=512 transform="{RandomCrop:{size:32,padding:4},RandomHorizontalFlip:{}}" Aug=Identity
```

Here is a more complex example, disabling the Encoder's flattening of the feature map, and instead giving the Actor and Critic unique Attention Pooling operations on their trunks to pool the unflattened features. The ```Identity``` architecture disables that flattening component.

```console
python Run.py task=classify/mnist Q_trunk=Transformer Pi_trunk=Transformer Pool=Identity
```

Here is a nice example of the critic using a small CNN for downsampling features:

```console
python Run.py task=dmc/cheetah_run Q_trunk=CNN +q_trunk.depth=1 pool=Identity
```

A CNN Actor and Critic:
```console
python Run.py Q_trunk=CNN Pi_trunk=CNN +q_trunk.depth=1 +pi_trunk.depth=1 Pool=Identity
```

[comment]: <> (<details>)

[comment]: <> (<summary><i>See even more examples :open_book: </i></summary>)

[comment]: <> (<br>)

[comment]: <> (Here's how you can load another saved agent's encoder from a pre-configured agent checkpoint ```<path>```:)

[comment]: <> (```)

[comment]: <> (python Run.py Eyes=Utils.load +eyes.path=<path> +eyes.attr=encoder.Eyes)

[comment]: <> (```)

[comment]: <> (You can imagine training a GAN CNN and then seamlessly using it for RL.)

[comment]: <> (<br>)

[comment]: <> (</details>)

*A little secret*, but pytorch code can be passed directly too via quotes:

```console
python Run.py "eyes='CNN(kwargs.input_shape,32,depth=3)'"
```
```console
python Run.py "eyes='torch.nn.Conv2d(kwargs.input_shape[0],32,kernel_size=3)'"
```

Some blocks have default args which can be accessed with the ```kwargs.``` interpolation shown above.

An intricate example of the expressiveness of this syntax:
```console
python Run.py Optim=SGD 'Pi_trunk="nn.Sequential(MLP(input_shape=kwargs.input_shape, output_dim=kwargs.output_dim),nn.ReLU(inplace=True))"' lr=0.01
```

Both the uppercase and lowercase syntax support direct function calls in place of usual syntax, with function calls distinguished by the syntactical quotes and parentheticals.

The parser automatically registers the imports/class paths in ```Utils.``` in both the uppercase and lowercase syntax, including modules/classes ```torch```, ```torch.nn```, and architectures/paths in ```./Blocks/Architectures/``` like ```CNN``` for direct access and no need to type ```Utils.```.

</details>

Of course, it's always possible to just modify the library code itself, which may be easier depending on your use case. The code is designed to be clear for educational and innovational purposes alike.

To make your own architecture mix-and-matchable, just put it in a pytorch module with initialization options for ```input_shape``` and ```output_dim```, as in the architectures in [```./Blocks/Architectures```](Blocks/Architectures).

</details>

### Custom Optimizers

<details>
<summary>
:mag: <i>Click to search/explore</i>
</summary>
<br>

Optimization parts can be accessed *e.g.* 

```console
python Run.py Optim=Utils.torch.optim.SGD lr=0.1
```

or via the expressive recipe interface described in [Custom Architectures](#custom-architectures):

```console
python Run.py Optim=SGD lr=0.1
```
or
```console
python Run.py "optim='torch.optim.SGD(kwargs.params, lr=0.1)'"
```

Learning rate schedulers can also be customized as well with ```scheduler=``` analogously, or via the ```lr_decay_epochs=``` shorthand for cosine annealing.


</details>

### Custom Dataset

<details>
<summary>
:mag: <i>Click to read/parse</i>
</summary>
<br>

You can pass in a Pytorch Dataset class as follows:

```console
python Run.py task=classify/custom Dataset=torchvision.datasets.MNIST
```

Another example:

```console
python Run.py task=classify/custom Dataset=Datasets.Suites._TinyImageNet.TinyImageNet
```

This will initiate a classify task on the custom-defined [```TinyImageNet```](Datasets/Suites/_TinyImageNet.py#L48) Dataset located in [./Datasets/Suites/_TinyImageNet.py](Datasets/Suites/_TinyImageNet.py). 

By default, the task name will appear as the Dataset class name (in the above examples, ```MNIST``` and ```TinyImageNet```). You can change the task name as it's saved for benchmarking and plotting, with ```task_name=```.

:exclamation: UnifiedML is compatible with datasets & domains beyond Vision.

<details>
<summary>
<i>More details :open_book:</i>
</summary>
<br>

For a non-Vision tutorial, see our full [end-to-end example](https://www.github.com/agi-init/XRD) of Crystalographic-Structure-And-Space-Group classification, in which we fully reproduce the [paper on classifying crystal structures and space groups from X-ray diffraction patterns]() in a single succinct file with some UnifiedML commands. The custom Crystal & Space Groups dataset will be downloaded automatically in the example.

> &#9432; Note that this dataset consists of *1-dimensional* data that is read into a 1D CNN and MLPs. UnifiedML architectures like CNN and MLP are **dimensionality-adaptive**! See [paper]() Section 3.6 for details about architecture adaptivity.

</details>

</details>

### Experiment naming, plotting

<details>
<summary>
:mag: <i>Click to see</i>
</summary>
<br>

Plots automatically save to ```./Benchmarking/<experiment>/```; the default experiment is ```experiment=Exp```.

```console
python Run.py
```

:chart_with_upwards_trend: :bar_chart: --> ```./Benchmarking/Exp/```

Optionally plot multiple experiments in a unified figure with ```plotting.plot_experiments=```.

```console
python Run.py experiment=Exp2 plotting.plot_experiments="['Exp', 'Exp2']"
```

Alternatively, you can call ```Plot.py``` directly

```console
python Plot.py plot_experiments="['Exp', 'Exp2']"
```

to generate plots. Here, the ```<experiment>``` directory name will be the underscore_concatenated union of all experiment names ("```Exp_Exp2```").

Plotting also accepts regex expressions. For example, to plot all experiments with ```Exp``` in the name:

```console
python Plot.py plot_experiments="['.*Exp.*']"
```

Another option is to use [WandB](https://wandb.ai/), which is supported by UnifiedML:

```console
python Run.py logger.wandb=true
```

You can connect UnifiedML to your WandB account by first running ```wandb login``` in your Conda environment.

To do a hyperparameter sweep, just use the ```-m``` flag.
```console
python Run.py -m task=atari/pong,classify/mnist seed=1,2,3 
```

</details>

[comment]: <> (The above will sweep over random seeds 1, 2, and 3, and whether to use EMA.)

# :bar_chart: Agents & Performances

In progress...

[comment]: <> (# :interrobang: How is this possible)

[comment]: <> (We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper]&#40;https://arxiv.com&#41;.)

[comment]: <> (Then we frame actions as "predictions" in supervised learning. We can even augment supervised learning with an RL phase, treating reward as negative error.)

[comment]: <> (For generative modeling, well, it turns out that the difference between a Generator-Discriminator and Actor-Critic is rather nominal.)

[comment]: <> ([comment]: <> &#40;![alt text]&#40;https://i.imgur.com/Yf8ltyI.png&#41;&#41;)

[comment]: <> (<img width="80%" alt="flowchart" src="https://i.imgur.com/nMUR9Ue.png">)

[comment]: <> (</p>)

[comment]: <> (<img width="80%" alt="flowchart" src="https://i.imgur.com/RM52cfJ.png?1">)

# :mortar_board: Pedagogy and Research

All files are designed for pedagogical clarity and extendability for research, to be useful for educational and innovational purposes, with simplicity at heart.

[comment]: <> (# :people_holding_hands: Contributing)

[comment]: <> (Please support financially: <br>)

[comment]: <> ([![Donate]&#40;https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat-square&#41;]&#40;https://www.paypal.com/cgi-bin/&#41; <br>)

[comment]: <> (We are a nonprofit, single-PhD student team. If possible, compute resources appreciated.)

[comment]: <> ([comment]: <> &#40;Our work will go towards helping nature through AI, making academic-level research accessible to all, and simplifying, bridging, and unifying the vast array of problem domains in our field.&#41;)

[comment]: <> (Feel free to [contact **agi.\_\_init\_\_**]&#40;mailto:agi.init@gmail.com&#41;.)

[comment]: <> (I am always looking for collaborators. Don't hesitate to volunteer in any way to help realize the full potential of this library.)

# Note

[comment]: <> (*While UnifiedML V.0 is a fully-realized, self-complete library, we note that it is also a ground for expansion beyond what is presented, with more performative breadth and depth on the way.*)

### If you are only interested in the RL portion,

Check out our [**UnifiedRL**](https:github.com/agi-init/UnifiedRL) library.

It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.~

<hr class="solid">

[MIT license Included.](MIT_LICENSE)



  