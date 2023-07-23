import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import copy
from Replay_Buffer import ReplayBuffer
from Noise_Class import OUNoise


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 300)
        self.fc3 = nn.Linear(300, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # action must be between -1 and 1
        return action


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dim + output_dim, 200)
        self.fc2 = nn.Linear(200, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state = torch.FloatTensor(state) if isinstance(state, np.ndarray) else state
        action = torch.FloatTensor(action) if isinstance(action, np.ndarray) else action
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class DDPGAgent:
    def __init__(self, env, actor, critic, actor_lr, critic_lr, gamma, tau, buffer_maxlen, sigma=0.2, epsilon=1):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.noise = OUNoise(2, sigma)
        self.replay_buffer = ReplayBuffer(max_size=buffer_maxlen)
        self.epsilon = epsilon

    def decay(self):
        self.epsilon *= 0.999

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
        if random.random() < self.epsilon:
            self.decay()
            return noise
        action = action.reshape(-1)
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = np.array(states)
        states = torch.FloatTensor(states.reshape(batch_size, -1))  # Flattening the states
        next_states = np.array(next_states)
        actions = np.array(actions)  # convert list of ndarrays to a single ndarray
        actions = torch.FloatTensor(actions)  # convert ndarray to tensor

        next_states = torch.FloatTensor(next_states.reshape(batch_size, -1))  # Flattening the next_states
        dones = torch.FloatTensor(dones).view(-1, 1)

        # Update critic
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions)
        rewards = torch.Tensor(rewards).view(-1, 1)

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
