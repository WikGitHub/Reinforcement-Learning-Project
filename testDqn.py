import gym
import ptan
import argparse
import random

import torch
import torch.optim as optim
from ignite.engine import Engine

# import utils
import common
import model_dqn
# import models
# from ptan import baseAgent
# from ptan import agent
import lossCalculator
from epsilonReducer import EpsilonReducer
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import warnings

METHOD_NAME = "basic_dqn"

if __name__ == "__main__":
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    result_list = []
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    game_parameters = common.HYPERPARAMS["pong"]
    # create the environment and apply a set of standard wrappers
    # render_mode = "human" would show the game screen
    # env = gym.make(game_parameters.environment_name, render_mode = "human")
    env = gym.make(game_parameters.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    # create the NN (double nets)
    device = torch.device("cuda" if args.cuda else "cpu")
    # print(env.action_space.n)
    # print(env.unwrapped.get_action_meanings())
    net = model_dqn.DQN(env.observation_space.shape, env.action_space.n).to(device)

    target_net = ptan.agent.TargetNet(net)

    # we create the agent, using an epsilon-greedy action selector as default.
    # During the training, epsilon will be decreased by the EpsilonReducer
    # This will decrease the amount of randomly selected actions and give more control to our NN
    action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=game_parameters.epsilon_start)
    epsilon_reducer = EpsilonReducer(selector=action_selector, params=game_parameters)
    agent = ptan.agent.DQNAgent(net, device=device, action_selector=action_selector)

    # The next two very important objects are ExperienceSource and ExperienceReplayBuffer.
    # The first one takes the agent and environment and provides transitions over game episodes.
    # Those transitions will be kept in the experience 'replay buffer'.
    experience_source = ptan.experience.ExperienceSourceFirstLast(env=env, agent=agent, gamma=game_parameters.gamma)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(experience_source=experience_source, buffer_size=game_parameters.replay_size)

    # Then we create an optimizer and define the processing function,
    # which will be called for every batch of transitions to train the model.
    # To do this, we call function loss_func of utils and then backpropagate on the result.
    opt = optim.Adam(net.parameters(), lr=game_parameters.learning_rate)

    # scheduler for learning rate decay(gamma is the decay rate), could be used in th future
    # see https://pytorch.org/docs/stable/optim.html
    # sched = scheduler.StepLR(opt, step_size=1, gamma=0.1)

    def process_batch(engine, batch):
        opt.zero_grad()
        loss_value = lossCalculator.mse_loss_func(
            batch, net, target_net.target_model,
            gamma=game_parameters.gamma, device=device)
        loss_value.backward()
        opt.step()
        epsilon_reducer.reduce_by_frames(engine.state.iteration)
        if engine.state.iteration % game_parameters.target_net_sync == 0:
            # sync the net
            target_net.sync()
        return {
            "loss": loss_value.item(),
            "epsilon": action_selector.epsilon,
        }

    # finally, we create the Ignite Engine object
    engine = Engine(process_batch)
    common.setup_ignite(engine, game_parameters, experience_source, METHOD_NAME)
    engine.run(common.batch_generator(replay_buffer, game_parameters.replay_initial,
                                      game_parameters.batch_size))