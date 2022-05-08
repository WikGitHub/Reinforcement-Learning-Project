import torch
import torch.nn as nn
import utils

# Calculation of the DQN loss function, based on nn's loss funcs
def huber_loss_func(batch, target_net, net, gamma, device="cpu"):
    states, actions, rewards, done, next_state = utils.get_batch(batch)

    states_value = torch.tensor(states).to(device)
    actions_value = torch.tensor(actions).to(device)
    rewards_value = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(done).to(device)
    next_states_value = torch.tensor(next_state).to(device)

    actions_value = actions_value.unsqueeze(-1)
    state_action_values = net(states_value).gather(1, actions_value)
    state_action_values = state_action_values.squeeze(-1)
    with torch.no_grad():
        next_state_values = target_net(next_states_value).max(1)[0]
        next_state_values[done_mask] = 0.0

    bellman_values = next_state_values.detach() * gamma + rewards_value
    theDelta = 1
    criterion = nn.HuberLoss(delta=theDelta)
    return criterion(state_action_values, bellman_values)


# def mse_loss_func(batch, target_net, net, gamma, device="cpu"):
#     states, actions, rewards, done, next_state = utils.get_batch(batch)
#
#     states_value = torch.tensor(states).to(device)
#     actions_value = torch.tensor(actions).to(device)
#     rewards_value = torch.tensor(rewards).to(device)
#     done_mask = torch.BoolTensor(done).to(device)
#     next_states_value = torch.tensor(next_state).to(device)
#
#     actions_value = actions_value.unsqueeze(-1)
#     state_action_values = net(states_value).gather(1, actions_value)
#     state_action_values = state_action_values.squeeze(-1)
#     with torch.no_grad():
#         next_state_values = target_net(next_states_value).max(1)[0]
#         next_state_values[done_mask] = 0.0
#
#     bellman_values = next_state_values.detach() * gamma + rewards_value
#     return nn.MSELoss()(state_action_values, bellman_values)


def mse_loss_func(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        utils.get_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)

# loss func for prioritised dqn
# new loss function, which accepts weights and returns the additional items' priorities
def pri_loss_func(batch, batch_weights, net, target_net, gamma, device="cpu"):
    states, actions, rewards, done, next_state = utils.get_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(done).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_state).to(device)
        next_s_vals = target_net(next_states_v).max(1)[0]
        next_s_vals[done_mask] = 0.0
        exp_sa_vals = next_s_vals.detach() * gamma + rewards_v
    l = (state_action_vals - exp_sa_vals) ** 2
    losses_v = batch_weights_v * l
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()