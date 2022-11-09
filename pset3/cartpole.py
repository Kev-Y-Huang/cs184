import gym
import numpy as np
from cartpole_controller import LocalLinearizationController, PIDController
from gym import wrappers

video_path = "./gym-results"
s_init = np.array([0, 0, 0, 0])
s_star = np.array([0, 0, 0, 0], dtype=np.double)
a_star = np.array([0], dtype=np.double)
T = 500

flag = 'LQR'

if flag == 'LQR':
    env = gym.make("env:CartPoleControlEnv-v0")
    controller = LocalLinearizationController(env)
    policies = controller.compute_local_policy(s_star, a_star, T)
else:
    controller = PIDController(20., 0., 1.)
    mask = np.array([0,0,1,0]) # PID only tracks angle error
    setpoint = 0


# For testing, we use a noisy environment which adds small Gaussian noise to
# state transition. Your controller only need to consider the env without noise.
env = gym.make("env:NoisyCartPoleControlEnv-v0")

env = wrappers.Monitor(env, video_path, force = True)
total_cost = 0
observation = env.reset(state = s_init)

for t in range(T):
    env.render()
    if flag == 'LQR':
        (K,k) = policies[t]
        action = (K @ observation + k)
    else:
        err = np.dot(observation, mask) - setpoint
        action = controller.get_action(err).reshape(1,)
    observation, cost, done, info = env.step(action)
    total_cost += cost
    if done: # When the state is out of the range, the cost is set to inf and done is set to True
        break
env.close()
print("cost = ", total_cost)
