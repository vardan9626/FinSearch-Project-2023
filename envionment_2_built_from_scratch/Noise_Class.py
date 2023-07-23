import numpy as np

class OUNoise:
    def __init__(self, action_dimension, sigma=0.2, scale=0.01, mu=0, theta=0.15, decay_rate=0.999):
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
        # self.decay()
        return np.clip(self.state * self.scale, -1.0, 1.0).reshape(1, -1)
