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
# from ptan import baseAgent
import lossCalculator
# from utils import PARA_SHORTCUT
from epsilonReducer import EpsilonReducer
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import warnings
METHOD_NAME = "prioritised_dqn"
REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 10**5
PROB_ALPHA = 0.6

class BetaClass:
    def __init__(self, beta):
        self.beta = beta


if __name__ == "__main__":
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    betaClass = BetaClass(BETA_START)
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
    replay_buffer = ptan.experience.PrioReplayBufferNaive(exp_source=experience_source, buf_size=game_parameters.replay_size, prob_alpha=PROB_ALPHA)

    # Then we create an optimizer and define the processing function,
    # which will be called for every batch of transitions to train the model.
    # To do this, we call function loss_func of utils and then backpropagate on the result.
    opt = optim.Adam(net.parameters(), lr=game_parameters.learning_rate)

    # scheduler for learning rate decay(gamma is the decay rate), could be used in th future
    # see https://pytorch.org/docs/stable/optim.html
    # sched = scheduler.StepLR(opt, step_size=1, gamma=0.1, verbose=True)

    # update_beta method of the buffer to change the beta parameter according to schedule.
    def update_beta(idx):
        value = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        beta = min(1.0, value)
        betaClass.beta = beta


    def create_batch_with_beta(buffer: ptan.experience.PrioReplayBufferNaive):
        step_size = 1
        buffer.populate(game_parameters.replay_initial)
        while 1:
            buffer.populate(step_size)
            # print(betaClass.beta)
            yield buffer.sample(game_parameters.batch_size, beta=betaClass.beta)

    def process_batch(engine, batch):
        batch, batch_indices, batch_weights = batch
        opt.zero_grad()
        loss_value, priorities = lossCalculator.pri_loss_func(
            batch, batch_weights, net, target_net.target_model,
            gamma=game_parameters.gamma, device=device)
        loss_value.backward()
        opt.step()
        replay_buffer.update_priorities(batch_indices, priorities)
        epsilon_reducer.reduce_by_frames(engine.state.iteration)
        if engine.state.iteration % game_parameters.target_net_sync == 0:
            # sync the net
            target_net.sync()
        return {
            "beta": update_beta(engine.state.iteration),
            "loss": loss_value.item(),
            "epsilon": action_selector.epsilon,
        }

    # finally, we create the Ignite Engine object
    engine = Engine(process_batch)
    common.setup_ignite(engine, game_parameters, experience_source, METHOD_NAME)
    engine.run(common.batch_generator(replay_buffer, game_parameters.replay_initial,
                                      game_parameters.batch_size))
    # engine = Engine(process_batch)
    # ptan_ignite.EndOfEpisodeHandler(experience_source, bound_avg_reward=game_parameters.stop_reward).attach(engine)
    # ptan_ignite.EpisodeFPSHandler().attach(engine)


    # @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    # def episode_completed(trainer: Engine):
    #     print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s, loss=%lf" % (
    #         trainer.state.episode, trainer.state.episode_reward,
    #         trainer.state.episode_steps, trainer.state.metrics.get('fps', 0),
    #         timedelta(seconds=trainer.state.metrics.get('time_passed', 0)),
    #         trainer.state.output["loss"]
    #     ))
    #     # if trainer.state.episode % 2 == 0:
    #     #     sched.step()
    #     #     print("LR decrease to", sched.get_last_lr()[0])
    #     result_list.append((trainer.state.episode,trainer.state.episode_reward))
    #
    #
    #
    # @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    # def game_solved(trainer: Engine):
    #     print("Game solved in %s, after %d episodes and %d iterations!" % (
    #         timedelta(seconds=trainer.state.metrics['time_passed']),
    #         trainer.state.episode, trainer.state.iteration))
    #     trainer.should_terminate = True
    #     print("--------Finished---------")
    #     print(result_list)
    #     for obj in result_list:
    #         print(obj)
    #
    #
    # # track TensorBoard data
    # logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{game_parameters.run_name}-{METHOD_NAME}={METHOD_NAME}"
    # tb = tb_logger.TensorboardLogger(log_dir=logdir)
    # RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")
    #
    # episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    # tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    #
    # # write to tensorboard every 100 iterations
    # ptan_ignite.PeriodicEvents().attach(engine)
    # handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'],
    #                                   output_transform=lambda a: a)
    # tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)
    #
    # engine.run(create_batch_with_beta(replay_buffer))


