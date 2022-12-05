import gym
import numpy as np
import utils
import matplotlib.pyplot as plt


def sample(theta, env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []

    for _ in range(N):
        total_rewards.append([])
        total_grads.append([])
        state = env.reset()

        for _ in range(200):
            phis = utils.extract_features(state, 2)
            policy = utils.compute_action_distribution(theta, phis)
            action = np.random.choice(2, 1, p=policy.flatten())
            state, reward, done, info = env.step(action.item())

            total_grads[-1].append(utils.compute_log_softmax_grad(theta, phis, action))
            total_rewards[-1].append(reward)

            if done:
                break

    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100,1)
    env = gym.make('CartPole-v0')
    env.seed(12345)

    episode_rewards = []

    for _ in range(T):
        grads, rewards = sample(theta, env, N)
        fisher = utils.compute_fisher_matrix(grads, lamb)
        v_grad = utils.compute_value_gradient(grads, rewards)
        eta = utils.compute_eta(delta, fisher, v_grad)

        theta += eta * np.linalg.inv(fisher) @ v_grad
        for i in range(len(rewards)):
            rewards[i] = sum(rewards[i])
        episode_rewards.append(np.average(rewards))

    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
