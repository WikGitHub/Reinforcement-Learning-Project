# Reinforcement-Learning-Project

DQN and its extensions

Worked with 5 other peers.

Code

Report

Video

as well as a presentation


# Reinforcemnet learning coursework2
# Overview
RL for pong
Thanks to the book "Deep-Reinforcement-Learning-Hands-On-Second-Edition" (https://learning.oreilly.com/library/view/deep-reinforcement-learning/9781838826994/), which encouraged us and gave us the initial idea of dealing with Atari games.
In this repository, we implemented five algorithms.
We began with DQN for the game pong, and we tried harder game 'breakout' with a more efficient algorithm 'multi-environment DQN'. Although 'multi-env' did well in the breakout, it was too time-consuming for the learning process and experiments with the hyperparameters seem like a huge burden for our computers. Thus we decided to stick to pong. Based on basic DQN we did three extensions, they are n_step DQN, dueling DQN, and prioritized DQN. Finally, we tried combining all the three improvements, which can be seen in the code 'combinedDqn.py', while it is just an experiment, not included in the paper.

# Requirements
     python 3
     tensorboardX
     ale_py==0.7.5
     gym == 0.23.1
     torch==1.11.0
     pytorch-ignite==0.4.9
     opencv_python==4.5.5.64
    
# Installation
    pip install -r requirements.txt

# Test
    python3 DQNs/testDqn.py

We recommend using Pycharm as IDE for running the code, or you would see:


Typical error:

gym.error.Error: We're Unable to find the game "Pong"

Solution:

We provided the 'roms' folder under the main directory, please move it to the '/venv/lib/python3.9/site-packages/ale_py'. Then try it again.
For more information see: https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management

# The libs
1.Pytorch-ignite is a helpful tool for training and NN works.(https://pytorch.org/ignite/index.html)

2.TensorboardX helps with visualization and figure tracking during the training process, how to use please see https://github.com/lanpa/tensorboardX

3.PTAN stands for PyTorch AgentNet -- reimplementation of AgentNet library for PyTorch. Please see https://github.com/Shmuma/ptan.


# Acknowledgement
We mainly used the 'experience' class and 'wrapper' class from the PTAN lib. 

The experience takes care of agents' interactions with the environment.

And the 'wrapper' class, which is mainly copied from OpenAi (https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py), makes it flexible executing actions in different games.

Compared to using a wrapper written by ourselves, the 'LazyFrames' function from OpenAi released much burden of the computation device, which helped during the training process. 

# Future work
Despite the computation power of computers, there are some alterations we could apply.

Combining several algorithms as experiments, or replacing the open-cv with pillow-simd may help wrappers calculate the frames.  
