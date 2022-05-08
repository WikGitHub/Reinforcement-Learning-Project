import ptan
# from utils import PARA_SHORTCUT
from types import SimpleNamespace

# implements epsilon decay during the training
# Epsilon defines the probability of taking the random action by the agent.
# It should be decayed from 1.0 in the beginning (fully random agent) to some small number,
# like 0.02 or 0.01. The code is trivial but needed in almost any DQN
class EpsilonReducer:
    # def __init__(self, selector: ptan.actions.GreedySelector,
                 # params: SimpleNamespace):
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.action_selector = selector
        self.parameters = params
        self.reduce_by_frames(0)

    def reduce_by_frames(self, current_frame:int):
        epsilon = self.parameters.epsilon_start - current_frame/self.parameters.epsilon_frames
        if epsilon >= self.parameters.epsilon_final:
            self.action_selector.epsilon = epsilon
        else:
            self.action_selector.epsilon = self.parameters.epsilon_final
