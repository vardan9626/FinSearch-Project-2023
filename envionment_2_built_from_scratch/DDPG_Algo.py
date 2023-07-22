import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy


# from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *random.sample(self.buffer, batch_size))
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, action_dimension, sigma=0.099, scale=0.01, mu=0, theta=0.15, decay_rate=0.999999):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        self.sigma *= self.decay_rate

    def decay(self):
        # decay the sigma value
        self.sigma *= self.decay_rate

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        self.decay()
        return np.array([self.state * self.scale], dtype='float').reshape(1, -1)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # action must be between -1 and 1
        return action


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dim + output_dim, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, state, action):
        state = torch.FloatTensor(state) if isinstance(state, np.ndarray) else state
        action = torch.FloatTensor(action) if isinstance(action, np.ndarray) else action
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


def check_action(action):

    if action[0, 0] > 1:
        action[0, 0] = 1
    elif action[0, 0] < -1:
        action[0, 0] = -1

    if action[0, 1] > 1:
        action[0, 1] = 1
    elif action[0, 1] < -1:
        action[0, 1] = -1
    return action


class DDPGAgent:
    def __init__(self, env, actor, critic, actor_lr, critic_lr, gamma, tau, buffer_maxlen,sigma=0.099):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.noise = OUNoise(2,sigma)
        self.replay_buffer = ReplayBuffer(max_size=buffer_maxlen)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.target_actor.state_dict(), filename + "_target_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.target_critic.state_dict(), filename + "_target_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.target_actor.load_state_dict(torch.load(filename + "_target_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.target_critic.load_state_dict(torch.load(filename + "_target_critic"))

    def get_action(self, state):
        state = torch.FloatTensor(np.array(state).flatten()).unsqueeze(0)  # Flattening the state
        action = self.actor(state)
        action = action.detach().numpy()[0]
        noise = self.noise.noise()
        action = action + noise
        action = check_action(action)
        action = action.reshape(-1)
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = np.array(states)
        states = torch.FloatTensor(states.reshape(batch_size, -1))  # Flattening the states
        # actions = torch.FloatTensor(actions).view(-1, 1)
        # rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_states = np.array(next_states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states.reshape(batch_size, -1))  # Flattening the next_states
        dones = torch.FloatTensor(dones).view(-1, 1)

        # Update critic
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions)
        rewards = torch.Tensor(rewards).view(-1, 1)

        # for debugging only
        # print(type(states),type(rewards), type(dones), type(self.gamma), type(next_q_values))
        # print(rewards.shape, dones.shape, next_q_values.shape,states.shape)

        # print("State shape: ", states.shape)
        # print("Action shape: ", actions.shape)
        # print(f"q_val shape: {q_values.shape}")
        # print(f"target_q_val shape: {target_q_values.shape}")
        # ends here
        target_q_values = torch.FloatTensor(rewards) + (1 - torch.Tensor(dones)) * self.gamma * next_q_values

        q_values = self.critic(states, actions)

        critic_loss = F.mse_loss(q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        policy_gradient = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        policy_gradient.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
