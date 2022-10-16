from MDP import build_mazeMDP
from DP import DynamicProgramming
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import numpy as np


class Visualize:
    def __init__(self, DP):
        self.DP = DP

        self.arrows = ['\u2191', '\u2193', '\u2190', '\u2192']
        self.iteration = 0

    def visualize_value_iteration(self, initialQ):
        self.Q = initialQ
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"Q-function values for iteration {self.iteration} of value iteration")

        Q_up = self.Q[:-1, 0].reshape(4, 4)
        Q_down = self.Q[:-1, 1].reshape(4, 4)
        Q_left = self.Q[:-1, 2].reshape(4, 4)
        Q_right = self.Q[:-1, 3].reshape(4, 4)

        a_img = {}
        a_img[0] = ax[0, 0].matshow(Q_up, cmap='seismic', vmin=np.min(Q_up) - 40, vmax=np.max(Q_up))
        ax[0, 0].set_title('Up-action (a=0)')
        a_img[1] = ax[0, 1].matshow(Q_down, cmap='seismic', vmin=np.min(Q_down) - 40, vmax=np.max(Q_down))
        ax[0, 1].set_title('Down-action (a=1)')
        a_img[2] = ax[1, 0].matshow(Q_left, cmap='seismic', vmin=np.min(Q_left) - 40, vmax=np.max(Q_left))
        ax[1, 0].set_title('Left-action (a=2)')
        a_img[3] = ax[1, 1].matshow(Q_right, cmap='seismic', vmin=np.min(Q_right) - 40, vmax=np.max(Q_right))
        ax[1, 1].set_title('Right-action (a=3)')

        text_dictionary = {0: {}, 1: {}, 2: {}, 3: {}}
        for i, j in np.ndindex(Q_up.shape):
            text_dictionary[0][(i, j)] = ax[0, 0].text(j, i, '{:0.1f}'.format(Q_up[i][j]), ha='center', va='center')
            text_dictionary[1][(i, j)] = ax[0, 1].text(j, i, '{:0.1f}'.format(Q_down[i][j]), ha='center', va='center')
            text_dictionary[2][(i, j)] = ax[1, 0].text(j, i, '{:0.1f}'.format(Q_left[i][j]), ha='center', va='center')
            text_dictionary[3][(i, j)] = ax[1, 1].text(j, i, '{:0.1f}'.format(Q_right[i][j]), ha='center', va='center')

        fig.canvas.mpl_connect('key_press_event', lambda event: self.VI_on_keyboard(event, fig, a_img, text_dictionary))

        plt.show()

    def VI_on_keyboard(self, event, fig, a_grid, text_dic):
        if event.key == 'right':
            self.iteration += 1
            self.Q = self.DP.valueIterationStep(self.Q)
            Q_up = self.Q[:-1, 0].reshape(4, 4)
            Q_down = self.Q[:-1, 1].reshape(4, 4)
            Q_left = self.Q[:-1, 2].reshape(4, 4)
            Q_right = self.Q[:-1, 3].reshape(4, 4)
            a_grid[0].set_array(Q_up)
            a_grid[1].set_array(Q_down)
            a_grid[2].set_array(Q_left)
            a_grid[3].set_array(Q_right)
            for i, j in np.ndindex(Q_up.shape):
                text_dic[0][(i, j)].set_text('{:0.1f}'.format(Q_up[i][j]))
                text_dic[1][(i, j)].set_text('{:0.1f}'.format(Q_down[i][j]))
                text_dic[2][(i, j)].set_text('{:0.1f}'.format(Q_left[i][j]))
                text_dic[3][(i, j)].set_text('{:0.1f}'.format(Q_right[i][j]))

            a_grid[0].set_clim(vmin=np.min(Q_up) - 40, vmax=np.max(Q_up))
            a_grid[1].set_clim(vmin=np.min(Q_down) - 40, vmax=np.max(Q_down))
            a_grid[2].set_clim(vmin=np.min(Q_left) - 40, vmax=np.max(Q_left))
            a_grid[3].set_clim(vmin=np.min(Q_right) - 40, vmax=np.max(Q_right))

        elif event.key == 'enter':
            pi = self.DP.extractMaxPifromQ(self.Q)
            V_pi = self.DP.computeVfromQ(self.Q, pi)
            self.visualize_V_and_Pi(V_pi, pi)
   
        fig.suptitle(f"Q-function values for iteration {self.iteration} of value iteration")

        fig.canvas.flush_events()
        fig.canvas.draw()

    def visualize_policy_iteration(self, initial_pi, exact):
        self.pi = initial_pi
        fig, ax = plt.subplots(2)
        self.V = self.DP.exactPolicyEvaluation(initial_pi) if exact else self.DP.approxPolicyEvaluation(initial_pi)[0]
        V_data = np.reshape(self.V[:-1], (4, 4))
        policy_data = np.reshape(self.pi[:-1], (4, 4))
        a_img = ax[0].matshow(V_data, cmap='seismic', vmin=np.min(self.V) - 40, vmax=np.max(self.V))
        b_img = ax[1].matshow(np.zeros((4, 4)), cmap=ListedColormap(['w', 'w', 'w', 'w']))

        a_text_dictionary = {}
        b_text_dictionary = {}
        for (i, j), z in np.ndenumerate(V_data):
            a_text_dictionary[(i, j)] = ax[0].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        for (i, j), z in np.ndenumerate(policy_data):
            b_text_dictionary[(i, j)] = ax[1].text(j, i, self.arrows[z], ha='center', va='center')
        fig.canvas.mpl_connect('key_press_event',
                               lambda event: self.PI_on_keyboard(event, exact, fig, a_img, b_img, a_text_dictionary,
                                                                 b_text_dictionary))

        fig.suptitle(f"Iteration {self.iteration} of {'exact' if exact else 'approximate'} policy iteration")

        plt.show()

    def PI_on_keyboard(self, event, exact, fig, a_grid, b_grid, a_dic, b_dic):
        if event.key == 'right':
            self.pi = self.DP.policyIterationStep(self.pi, exact)
            self.V = self.DP.exactPolicyEvaluation(self.pi) if exact else self.DP.approxPolicyEvaluation(self.pi)[0]

            V_data = np.reshape(self.V[:-1], (4, 4))
            a_grid.set_array(V_data)
            for (i, j), z in np.ndenumerate(V_data):
                a_dic[(i, j)].set_text('{:0.1f}'.format(z))

            a_grid.set_clim(vmin=np.min(self.V) - 40)
            a_grid.set_clim(vmax=np.max(self.V))

            policy_data = np.reshape(self.pi[:-1], (4, 4))
            b_grid.set_array(policy_data)
            for (i, j), z in np.ndenumerate(policy_data):
                b_dic[(i, j)].set_text(self.arrows[z])
            self.iteration += 1
            fig.suptitle(f"Iteration {self.iteration} of {'exact' if exact else 'approximate'} policy iteration")
            fig.canvas.flush_events()
            fig.canvas.draw()

    def visualize_V_and_Pi(self, V, pi):
        fig, ax = plt.subplots(2)
        V_data = np.reshape(V[:-1], (4, 4))
        pi_data = np.reshape(pi[:-1], (4, 4))
        ax[0].matshow(V_data, cmap='seismic', vmin=np.min(V) - 40, vmax=np.max(V))
        ax[1].matshow(np.zeros((4, 4)), vmin=-1, vmax=1)

        a_text_dictionary = {}
        b_text_dictionary = {}
        for (i, j) in np.ndindex(V_data.shape):
            a_text_dictionary[(i, j)] = ax[0].text(j, i, '{:0.1f}'.format(V_data[i, j]), ha='center', va='center')
            b_text_dictionary[(i, j)] = ax[1].text(j, i, self.arrows[pi_data[i, j]], ha='center', va='center')

        plt.show()


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2 or args[1] not in ["VI", "PI_exact", "PI_approx"]:
        print("Incorrect args")
        sys.exit(0)
    else:
        mdp = build_mazeMDP()
        dp = DynamicProgramming(mdp)
        v = Visualize(dp)
        if args[1] == "VI":
            Q = np.zeros((dp.nStates, dp.nActions))
            v.visualize_value_iteration(Q)
        elif args[1] == "PI_exact":
            pi = np.zeros(dp.nStates, dtype='int')
            v.visualize_policy_iteration(pi, True)
        else:
            pi = np.zeros(dp.nStates, dtype='int')
            v.visualize_policy_iteration(pi, False)
