![UnifiedMLBanner](https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/dfdfba35-064f-4749-a8f8-081dda770e79)

### Quick Links

- [Setup](#wrench-setting-up)

- [Tutorial](#mag-full-tutorials)

- [Agents and performances](#bar_chart-agents--performances)

# :runner: Running The Code

To start a train session, once [installed](#wrench-setting-up):

```console
python Run.py
```

Defaults:

```Agent=Agents.AC2Agent```

```task=atari/pong```

Plots, logs, generated images, and videos are automatically stored in: ```./Benchmarking```.

![ApeMan](https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/761aaa53-3c90-46ee-a12e-b2e7a6302a04)

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, à la carte!

Drink up! :beers:

# :pen: Paper & Citing

For detailed documentation, [see our :scroll:](https://docs.google.com/presentation/d/1JpT09GMN0xa81J1h88urRklcOZkJ704s58LHHikmUG8/edit?usp=sharing).

```bibtex
@article{UnifiedML,
  title   = {UnifiedML: A Unified Framework For Intelligence Training},
  author  = {Sam Lerman, Chenliang Xu},
  howpublished = {https://github.com/AGI-init/UnifiedML},
  year    = {2023}
}
```

If you use this work, please give us a star :star: and be sure to cite the above!

An acknowledgment to [Denis Yarats](https://github.com/denisyarats), whose excellent [DrQV2 repo](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

# :open_umbrella: Unified Learning?

Yes.

Our ```AC2Agent``` supports discrete and continuous control, classification, generative modeling, and more.

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

## 3. Activate Your Conda Env.

```console
conda activate ML
```

#

> > &#9432; Depending on your CUDA version, you may need to redundantly install Pytorch with CUDA from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) after activating your Conda environment.
>
> For example, for CUDA 11.6:
> ```console
> pip uninstall torch torchvision torchaudio
> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
> ```

# :joystick: Installing The Suites

## 1. Atari Arcade

<p align="left">
<img src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/21c4643f-70c9-4f1a-a724-6404f93c1508" width="320">
<br><i>A collection of retro Atari games.</i>
</p>

You can install via ```AutoROM``` if you accept the license. First install ```AutoROM```.

```console
pip install autorom
```

Then accept the license.

```console
AutoROM --accept-license
```

## 2. DeepMind Control

Comes pre-installed! For any issues, consult the [DMC repo](https://github.com/deepmind/dm_control).

<p align="left">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rAai4QzcYbs" target="_blank"><i>:arrow_forward: Click to play</i></a><br>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rAai4QzcYbs" target="_blank">
<img src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/04781784-9f38-4efa-a9c8-466827a74ea7" alt="Play video" width="310" />
</a>
<br><i>Video of different tasks in action.</i>
</p>

## 3. Classify

<p align="left">

<img src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/32648461-e8a7-4b55-a2d0-6b7eb23b23cb" width="320">
<br><i>Eight different ladybug species in the iNaturalist dataset.</i>

</p>

[All datasets](Hyperparams/task/classify) come ready-to-use :white_check_mark:

That's it.

> :bulb: Train Atari example: ```python Run.py task=atari/mspacman```
>
> :bulb: Train DMC example: ```python Run.py task=dmc/cheetah_run```
>
> :bulb: Train Classify example: ```python Run.py task=classify/mnist```

# :file_cabinet: Key files

```Run.py``` handles learning and evaluation loops, saving, distributed training, logging, plotting.

```Environment.py``` handles rollouts.

```./Agents``` contains self-contained agents.

#

# :mag: Full Tutorials

### RL

<details>
<summary>
:mag: <i>Click to interact</i>
</summary>
<br>

**Train** [```DQN Agent```](Agents/DQN.py) **to play Ms. Pac-Man**:

```console

python Run.py task=atari/mspacman Agent=Agents.DQNAgent

```


* Our implementation expands on [ensemble Q-learning](https://arxiv.org/abs/1802.09477v3) with [data regularization](https://arxiv.org/pdf/2004.13649.pdf) and [Soft-DQN](https://arxiv.org/pdf/2007.14430.pdf).
* [Original Nature DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).

——❖——

**Humanoid from pixels** with [```DrQV2 Agent```](Agents/DrQV2.py), [a state of the art algorithm for continuous control from images](https://arxiv.org/abs/2107.09645):
```console
python Run.py task=dmc/humanoid_walk Agent=Agents.DrQV2Agent
```

⋆⋅☆⋅⋆

**Play Super Mario Bros.** with [```Dueling DQN Agent```](Agents/DuelingDQN.py), an extension of DQN that uses [dueling Q networks](https://arxiv.org/abs/1511.06581):
```console
python Run.py task=mario Agent=Agents.DuelingDQNAgent
```

•⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼✧ ☼ 𖥸 ☽ ✧⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽•

**The library's default Agent** is our [```AC2 Agent```](Agents/Lermanbots/AC2.py) (```Agent=Agents.AC2Agent```).

```console
python Run.py
```

* ```+agent.depth=5``` can activate a self-supervisor to predict temporal dynamics for a number of timesteps ahead, similar to [Dreamer](https://arxiv.org/pdf/2301.04104v1.pdf) and [SPR](https://arxiv.org/abs/2007.05929).
* ```+agent.num_actors=5 +agent.num_critics=5``` can activate actor-critic ensembling.

**In addition to RL**, this agent supports classification, generative modeling, and various modes.  Therefore we refer to it [as a framework](https://docs.google.com/presentation/d/1JpT09GMN0xa81J1h88urRklcOZkJ704s58LHHikmUG8/edit?usp=sharing), not just an agent. The full array of the library's features and cross-domain compatibilities are supported by this agent.

⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽⎽⎼⎻⎺⎺⎻⎼⎽

Save videos with ```vlog=true```.

:clapper: :movie_camera: -> ```Benchmarking/<experiment>/<agent>/<suite>/<task>_<seed>_Video_Image/```

<p>
<img src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/b7f64382-606d-4585-8f03-04c363cf09d6" width="64" height="84"/>
</p>

Check out [args.yaml](Hyperparams/args.yaml) for the full array of configurable options available, including
* N-step rewards (```nstep=```)
* Frame stack (```frame_stack=```)
* Action repeat (```action_repeat=```)
* & more, with [per-task](Hyperparams/task) defaults in ```/Hyperparams/task``` — please [share your hyperparams](https://github.com/agi-init/UnifiedML/discussions/11) if you discover new or better ones!


&#9432; If you'd like to **discretize** a continuous domain, pass in ```discrete=true``` and specify the number of discrete bins per action dimension via ```num_actions=```. If you'd like to **continuous-ize** a discrete domain, pass in ```discrete=false```. *Action space conversions are experimental*.

#

> :bulb: *The below sections describe many features in other domains, but chances are those features will work in RL as well. For example, a cosine annealing learning rate schedule can be toggled with: ```lr_decay_epochs=100```. It will anneal per-episode rather than per-epoch. Different model architectures, image transforms, EMAs, and more are all supported across domains!*
> 
> The vast majority of this hasn't been tested outside of its respective domain (CV, RL, etc.), so the research opportunity is a lot!

</details>

### Classification 

<details>
<summary>
:mag: <i>Click to categorize</i>
</summary>
<br>

CNN on MNIST:

```console
python Run.py task=classify/mnist 
```

*Note:* ```RL=false``` is the default for ```classify``` tasks. Keeps training at **standard** supervised-only classification.

**Variations**

Since this is *Unified*ML, there are a couple noteworthy variations. You can ignore these if you are only interested in standard classification via cross-entropy supervision only.

1. With ```RL=true```, an **augmented RL** update joins the supervised learning update $\text{s.t. } reward = -error$ (**experimental**).

2. Alternatively, and interestingly, ```supervise=false RL=true``` will *only* supervise via RL $reward = -error$. This is **pure-RL** training and actually works!

Classify environments can actually be great testbeds for certain RL problems since they give near-instant and clear performance feedback.

*Ignore these variations for doing standard classification.*

**Important features** 

Many popular features are unified in this library and generalized across RL/CV/generative domains, with more being added: 

* Evaluation with [exponential moving average (EMA)](https://arxiv.org/pdf/1803.05407.pdf) of params can be toggled with the ```ema=true``` flag; customize the decay rate with ```ema_decay=```. 
  
* See [Custom Architectures](#custom-architectures) for mix-and-matching custom or pre-defined (*e.g.* ViT, ResNet50) architectures via the command line syntax. 
  
* Different optimizations [can be configured](#custom-optimization) too.
  
* As well as [Custom Datasets](#custom-dataset). 

* Ensembling is supported (e.g., ```+agent.num_actors=```)
  
* Training with [weight decay](https://arxiv.org/abs/1711.05101) can be toggled via ```weight_decay=```. 
  
* A [cosine annealing learning rate schedule](https://arxiv.org/abs/1608.03983) can be applied for $N$ epochs (or episodes in RL) with ```lr_decay_epochs=```. 

* And [TorchVision transforms](https://pytorch.org/vision/stable/transforms.html) can be passed in as dicts via ```transform=```. 
  
For example,

```console
python Run.py task=classify/cifar10 weight_decay=0.01 transform="{RandomHorizontalFlip:{p:0.5}}" Eyes=Blocks.Architectures.ResNet18
```

The above returns a $94$% on CIFAR-10 with a ResNet18, which is pretty good. Changing datasets/architectures is as easy as modifying the corresponding parts ```task=``` and ```Eyes=``` of the above script.

And if you set ```supervise=false RL=true```, we get about the same score... vis-à-vis pure-RL. 

This library is meant to be useful for academic research, and out of the box supports [many datasets](Hyperparams/task/classify), including 
* Tiny-ImageNet (```task=classify/tinyimagenet```), 
* iNaturalist, (```task=classify/inaturalist```),
* CIFAR-100 (```task=classify/cifar100```), 
* & [more](Hyperparams/task/classify), normalized and no manual preparation needed

</details>

### Generative Modeling

<details>
<summary>
:mag: <i>Click to synth</i>
</summary>
<br>

Via the ```generate=true``` flag:
```console
python Run.py task=classify/mnist generate=true
```

<p align="left">
<img src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/adbba097-b118-4091-a8b6-54fd1d4a0735" width="180">
<br><i>Synthesized MNIST images, conjured up and imagined by a simple MLP.</i>
</p>

Saves to ```./Benchmarking/<experiment>/<Agent name>/<task>_<seed>_Video_Image/```.

Defaults can be easily modified with custom architectures or even datasets as elaborated in [Custom Architectures](#custom-architectures) and [Custom Datasets](#custom-dataset). Let's try the above with a CNN Discriminator:

```console
python Run.py task=classify/mnist generate=true Discriminator=CNN +agent.num_critics=1
```

```+agent.num_critics=1``` uses only a single Discriminator rather than ensembling as is done in RL. See [How Is This Possible?](#interrobang-how-is-this-possible) for more details on the unification. 

Or a ResNet18:

```console
python Run.py task=classify/mnist generate=true Discriminator=ResNet18
```

Let's speed up training by turning off the default image augmentation, which is overkill anyway for this simple case:

```console
python Run.py task=classify/mnist generate=true Aug=Identity +agent.num_critics=1
```

```Aug=Identity``` substitutes the default random cropping image-augmentation with the Identity function, thereby disabling it.

Generative mode implicitly treats training as [offline](#offline-rl), and assumes a replay [is saved](#saving) that can be loaded. As long as a dataset is available or a replay has [been saved](#saving), ```generate=true``` will work for any defined visual task, making it a powerful hyper-parameter that can just work. For now, only visual (image) tasks are compatible. 

Can even work with RL tasks (due to frame stack, the generated images are technically multi-frame videos).

```console
python Run.py task=atari/breakout generate=true
```

Make sure you have [saved a replay](#saving) that can be loaded before doing this.

</details>

### Saving
<details>
<summary>
:mag: <i>Click to remember</i>
</summary>
<br>

**Agents** are automatically saved at the end of training:

```console
python Run.py train_steps=2
```

**Agents** can be saved periodically and/or loaded with the ```save_per_steps=``` or ```load=true``` flags respectively:

```console
# Saves periodically
python Run.py save_per_steps=100000

# Load
python Run.py load=true
```

**Agents** may be trained without saving by adding the ```save=false``` flag.

An **experience replay** can be saved and/or loaded with the ```replay.save=true``` or ```replay.load=true``` flags.

```console
# Save
python Run.py replay.save=true

# Load
python Run.py replay.load=true
```

Online tasks, such as online RL, will create a new replay if ```replay.load=false```, or — careful — potentially delete the current replay at the end of training if ```replay.save=false```.

By default, classify tasks are offline, meaning you don't have to worry about loading or saving replays. Since the dataset is static, creating/loading is handled automatically.

<details>
<summary>
Click here to learn more about replays
</summary>
<br>

<img width="25%" alt="flowchart" src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/815d59ab-63e8-4c5e-9203-ebfbb87ef9a5"><br><br>

**In UnifiedML, replays are an efficient accelerated storage format for data that support both static and dynamic (changing/growing) datasets.**

You can disable the use of replays with ```stream=true```, which just sends data to the Agent directly from the environment. In RL, this is equivalent to on-policy training. In classification, it means you'll just directly use the Pytorch Dataset, without all the fancy replay features and accelerations.

Replays are recommended for RL because on-policy algorithmic support is currently limited.

~

</details>

Agents and replays save to ```./Checkpoints``` and ```./Datasets/ReplayBuffer``` respectively per *a unique experiment*, otherwise overwriting.

*A unique experiment* is distinguished by the flags: ```experiment=```, ```Agent=```, ```suite=```, ```task_name=```, and ```seed=```.

You can change the Agent load/save path with ```load_path=```/```save_path=```, and ```replay.path=``` for replays. All three accept string paths e.g. ```load_path='./Checkpoints/Exp/AC2Agent/classify/MNIST_1.pt'```.


</details>

### Offline RL

<details>
<summary>
:mag: <i>Click to play retroactively</i>
</summary>
<br>

Offline means the dataset size doesn't grow.

From a saved experience replay, sans additional rollouts:

```console
python Run.py task=atari/breakout offline=true
```

Assumes a replay [is saved](#saving).

Implicitly treats ```replay.load=true``` and ```replay.save=true```, and only does learning updates and evaluation rollouts.

```offline=true``` is the default for classification, where datasets are automatically downloaded and created into offline replays.

</details>

### Distributed

<details>
<summary>
:mag: <i>Click to de-centralize</i>
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

</details>

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

Atari with ViT:

```console
python Run.py Eyes=ViT +eyes.patch_size=7
```

Shorthands like ```Aug```, ```Eyes```, and ```Pool``` make it easy to plug and play custom architectures. All of an agent's architectural parts can be accessed, mixed, and matched with their [corresponding recipe shorthand](Hyperparams/args.yaml#L182-L233) names.

Generally, the rule of thumb is Capital names for paths to classes (such as ```Eyes=Blocks.Architectures.MLP```) and lowercase names for shortcuts to tinker with model args (such as ```+eyes.depth=1```).

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
python Run.py Optim=SGD 'Pi_trunk="nn.Sequential(MLP(input_shape=kwargs.input_shape, output_shape=kwargs.output_shape),nn.ReLU(inplace=True))"' lr=0.01
```

Both the uppercase and lowercase syntax support direct function calls in place of usual syntax, with function calls distinguished by the syntactical quotes and parentheticals.

The parser automatically registers the imports/class paths in ```Utils.``` in both the uppercase and lowercase syntax, including modules/classes ```torch```, ```torch.nn```, and architectures/paths in ```./Blocks/Architectures/``` like ```CNN``` for direct access and no need to type ```Utils.```.

</details>

To ***make*** a custom architecture, you can use any Pytorch module which outputs a tensor. Woohoo, done.

To make it mix-and-matchable throughout UnfiedML for arbitrary dimensionalities and domains, to generalize as much as possible, you can add:
1. ```input_shape``` and ```output_shape``` arguments to the \_\_init\_\_ method, such that your architecture can have a defined adaptation scheme for different possible shapes.
2. Support arbitrary many inputs (such as by concatenating them) of weird shapes (broadcasting them).
3. A ```repr_shape(*_)``` method that pre-computes the output shape given a varying-number of input shape dimensions as arguments.

None of these add-ons are *necessary*, but if you include all of them, then your architecture can adapt to everything. There are lazy ways to hack all of these features into any architecture, or you can follow the pretty basic templates used in our existing array of architectures. Most of our architectures can probably be used to build whatever architecture you’re trying to build, honestly, or at least something similar enough that you could have a good jumping-off point.

In short: To make your own architecture mix-and-matchable, just put it in a pytorch module with initialization options for ```input_shape``` and ```output_shape```, as in the architectures in [```./Blocks/Architectures```](Blocks/Architectures).

The Encoder Eyes automatically adapt 2d conv to 1d conv by the way (if data is 1d).

</details>

### Custom Optimizers

<details>
<summary>
:mag: <i>Click to search/explore</i>
</summary>
<br>

You can pass in a path to the ```Optim=``` flag or select a built-in Pytorch optimizer like ```SGD```, or both as below:

```console
python Run.py Optim=Utils.torch.optim.SGD lr=0.1
```

Equivalently via the expressive recipe interface:

```console
python Run.py Optim=SGD lr=0.1
```

or

```console
python Run.py "optim='torch.optim.SGD(kwargs.params, lr=0.1)'"
```

In the first two examples, the ```lr=``` flag was optional. The default learning rate is ```1e-4``` and we could have writen ```+optim.lr=```.

**Per-block optimizers** For example, just the Encoder:

```console
python Run.py encoder.Optim=SGD
```

**Learning rate schedulers.** ```Scheduler=``` works analogously to ```Optim=```, or just use the ```lr_decay_epochs=``` shorthand for cosine annealing *e.g.*

```console
python Run.py task=classify/mnist lr_decay_epochs=100
```

</details>

### Custom Env

<details>
<summary>
:mag: <i>Click to let there be light</i>
</summary>
<br>

As an example of custom environments, we provide the [Super Mario Bros.](https://github.com/Kautenja/gym-super-mario-bros) game environment in [./Datasets/Suites/SuperMario.py](Datasets/Suites/SuperMario.py).

To use it, you can just pass in the path to ```Env=``` and specify the ```suite``` and the ```task_name``` to your choosing:

```console
python Run.py Env=Datasets.Suites.SuperMario.SuperMario suite=SuperMario task_name=Mario
```

<p align="left">
<img src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/a9554164-4460-4a10-ad7b-fdae242e116d" width="180"/>
<br><i>Mario trained via DQN.</i>
</p>

Any Hyperparams you don't specify will be inherited from the default task, ```atari/pong``` in [```./Hyperparams/task/atari/pong.yaml```](Hyperparams/task/atari/pong.yaml), or whichever task is selected.

&#9432; If you want to save Hyperparams and formally define a task, you can create files like [```./Hyperparams/task/mario.yaml```](Hyperparams/task/mario.yaml) in the [./Hyperparams/task/](Hyperparams/task) directory:

```ruby
# ./Hyperparams/task/mario.yaml
defaults:
  - _self_

Env: Datasets.Suites.SuperMario.SuperMario
suite: SuperMario
task_name: Mario
discrete: true
action_repeat: 4
truncate_episode_steps: 250
nstep: 3
frame_stack: 4
train_steps: 3000000
stddev_schedule: 'linear(1.0,0.1,800000)'
```

Now you can launch Mario with:

```console
python Run.py task=mario
```

You can also customize params and worlds and stages with the ```+env.``` syntax:

```console
python Run.py task=mario +env.stage=2
```

</details>

### Custom Dataset

<details>
<summary>
:mag: <i>Click to read, parse, & boot up</i>
</summary>
<br>

You can pass in any Dataset as follows:

```console
python Run.py task=classify/custom Dataset=torchvision.datasets.MNIST
```

That will launch MNIST. Another example, with a custom class and path,

```console
python Run.py task=classify/custom Dataset=Datasets.Suites._TinyImageNet.TinyImageNet
```

This will initiate a classify task on the custom-defined [```TinyImageNet```](Datasets/Suites/_TinyImageNet.py#L48) Dataset.

You can change the task name as it's saved for benchmarking and plotting, with ```task_name=```. The default is the class name such as ```TinyImageNet```.

**UnifiedML is compatible with datasets & domains besides Vision.**

Thanks to dimensionality adaptivity ([slide 12](https://docs.google.com/presentation/d/1JpT09GMN0xa81J1h88urRklcOZkJ704s58LHHikmUG8/edit?usp=sharing)) for example, train the default CNN architecture on raw 1D Audio:

```console
python Run.py task=classify/custom Dataset=Datasets.Suites._SpeechCommands.SpeechCommands Aug=Identity
```

Gets a perfect score on speech command classification from raw 1D audio with the default CNN setting.  

<details>
<summary>
<i>More details and examples :open_book:</i>
</summary>
<br>

For a non-Vision/Audio tutorial, we provide a full [end-to-end example](https://www.github.com/agi-init/XRDs) in Crystal classification, reproducing [classifying crystal structures and space groups from X-ray diffraction patterns]().

</details>

---

Note: You can also specify an **independent test dataset** explicitly with ```TestDataset=```. 

</details>

### Recipes

<details>
<summary>
:mag: <i>Learn to cook</i>
</summary>
<br>

**Save hyperparams** to ```.yaml``` files by defining them in the [./Hyperparams/task/](Hyperparams/task) directory. There are many saved examples already.

If you've defined a ```.yaml``` file called ```my_recipe.yaml``` for example, you can use it via 

```console
python Run.py task=my_recipe
```

Please [share your recipes](https://github.com/agi-init/UnifiedML/discussions/11) in our Discussions page if you discover new or better hyperparams for a problem.

**Recipes can also be defined temporarily via command line without saving them to .yaml files.**

Below is a running list of some out-of-the-ordinary or interesting ones:

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

```console
python Run.py Aug=Sequential +aug._targets_="[IntensityAug, RandomShiftsAug]" +aug.scale=0.05 aug.pad=4
```

These are also useful for testing whether I've broken things.

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

Optionally plot multiple experiments

```console
python Run.py experiment=Exp2 plotting.plot_experiments="['Exp', 'Exp2']"
```

Alternatively, you can call [```Plot.py```](Plot.py) directly

```console
python Plot.py plot_experiments="['Exp', 'Exp2']"
```

to generate plots. Here, the ```<experiment>``` directory name will be the underscore_concatenated union of all experiment names ("```Exp_Exp2```").

Plotting also accepts regex expressions. For example, to plot all experiments with ```Exp``` in the name:

```console
python Plot.py plot_experiments="['Exp.*']"
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

Log video during evaluations with ```log_media=true```.

</details>

### Publishing

<details>
<summary>
:mag: <i>Click to write your own paper</i>
</summary>
<br>

We have released our slide deck!

[Templates available here](https://docs.google.com/presentation/d/1JpT09GMN0xa81J1h88urRklcOZkJ704s58LHHikmUG8/edit?usp=sharing)

Feel free to use our UnifiedML templates and figures in your work, citing [us](#pen-paper--citing) of course.

**Open-source research for minimal redundancy and optimal standardization is the way to go, balancing privacy and de-centrality, and streamlining successive works that depend on ours in good faith. Post your own designs and assets [here](https://github.com/agi-init/UnifiedML/discussions/12) in the discussion board. Read the rules to keep citations and credit attribution fair.**

</details>

# :bar_chart: Agents & Performances

<details>
<summary>
Atari
</summary>
<br>

We can attain 100% mean human-normalized score across the Atari-26 benchmark suite in about 1m environment steps. 

The below example script shows how to launch training for just Pong and Breakout with ```AC2Agent```:

```console
python Run.py task=atari/pong,atari/breakout -m
```

The results are reported for all 26 games and 3 different agents:

<img width="40%" alt="flowchart" src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/ac50932a-a25c-46af-9d6b-c8cc13d1a0e7">

<details>
<summary>
Click here to see per-task results.
</summary>
<br>

<img width="80%" alt="flowchart" src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/75688e0b-ef99-4909-afe7-a592ad030e3b">

</details>

We found these results to be pretty stable across a range of exploration rates as well:

<img width="60%" alt="flowchart" src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/517a6aff-7a44-4f6d-af21-fdbee8a27684">

Each time point averages over 10 evaluation episodes (and 26 games).

</details>

<details>
<summary>
DCGAN
</summary>
<br>

The simplest way to do DCGAN is to use the DCGAN architecture:

```console
python Run.py task=classify/celeba generate=true Discriminator=DCGAN.Discriminator Generator=DCGAN.Generator train_steps=50000
```

<img width="40%" alt="flowchart" src="https://github.com/AGI-init/Logos/assets/92597756/16a6eca0-10b1-4081-8903-952e1a5680d2">

We can then improve the results, and speed up training tenfold, by modifying the hyperparameters:

```console
python Run.py task=classify/celeba generate=true Discriminator=DCGAN.Discriminator Generator=DCGAN.Generator z_dim=100 Aug=Identity Optim=Adam '+optim.betas=[0.5, 0.999]' lr=2e-4 +agent.num_critics=1 train_steps=5000
```

<img width="60%" alt="flowchart" src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/66a28936-9234-4b76-8ea2-90cef0bd60e6">

</details>

# :interrobang: How is this possible

We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper](https://docs.google.com/presentation/d/1JpT09GMN0xa81J1h88urRklcOZkJ704s58LHHikmUG8/edit?usp=sharing).

Then we frame actions as "predictions" in supervised learning. We can even augment supervised learning with an RL phase, treating reward as negative error.

For generative modeling, well, it turns out that the difference between a Generator-Discriminator and Actor-Critic is rather nominal.

<img width="80%" alt="flowchart" src="https://github.com/AGI-init/UnifiedML-legacy/assets/92597756/fe80c2ab-4629-48d8-a445-a18424cf3268">

# :mortar_board: Pedagogy and Research

All files are designed for pedagogical clarity and extendability for research, to be useful for educational and innovational purposes, with simplicity at heart.

# :people_holding_hands: Contributing

Please support financially by Sponsoring. <br>

We are a nonprofit, single-PhD student team. If possible, compute resources appreciated.

Feel free to [contact **agi.\_\_init\_\_**](mailto:agi.init@gmail.com).

I am always looking for collaborators. Don't hesitate to volunteer in any way to help realize the full potential of this library.

<hr class="solid">

[MIT license Included.](MIT_LICENSE)

Non-legacy version: [here](https://github.com/AGI-init/UnifiedML).



  
