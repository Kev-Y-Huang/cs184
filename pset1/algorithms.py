from env_MAB import *
from functools import lru_cache
import math
from scipy.special import factorial


def random_argmax(a):
    '''
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    '''
    return np.random.choice(np.where(a == a.max())[0])


class Explore():
    '''
    Pull each arm an equal number of times. This
    can be accomplished in a number of ways. For this assignment, please implement the algorithm
    as follows: at each step, pull the arm that has been pulled the least number of times, and ties are
    broken at random. This also addresses the case when K does not divide T 
    '''

    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        # grab index with the least number of pulls
        i = random_argmax(-1 * np.sum(self.MAB.get_record(), axis=1))
        self.MAB.pull(i)


class Greedy():
    '''
    Pull each arm once (in any order). At each time
    step after that, commit to the greedy arm (i.e., the arm with the highest observe reward). Ties are
    broken at random.
    '''

    def __init__(self, MAB):
        self.MAB = MAB
        self.curr = 0

    def reset(self):
        self.curr = 0
        self.MAB.reset()

    def play_one_step(self):
        # Pull each arm once
        if self.curr < self.MAB.get_K():
            self.MAB.pull(self.curr)
            self.curr += 1
        # Pull the arm with the highest observed reward
        else:
            record = self.MAB.get_record()
            i = random_argmax(record[:, 1] / np.sum(record, axis=1))
            self.MAB.pull(i)


class ETC():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.count = 0
        self.record = MAB.get_record()

        # Calculate the number of times to pull each arm
        self.N = math.floor((self.MAB.get_T(
        ) * math.sqrt(math.log(2 * self.MAB.get_K() / delta) / 2) / self.MAB.get_K()) ** (2/3))

    def reset(self):
        self.MAB.reset()
        self.count = 0
        self.record = self.MAB.get_record()

    def play_one_step(self):
        # Exploration phase (pull each arm N times)
        if self.count < self.MAB.get_K() * self.N:
            # Grab random index where count < N
            i = random_argmax(-1 * (np.sum(self.record, axis=1) // self.N))
            self.MAB.pull(i)
            # Increase count and store record
            self.count += 1
            self.record = self.MAB.get_record()
        # Exploitation phase (pull the arm with the highest empirical mean during the exploration phase)
        else:
            i = random_argmax(self.record[:, 1] / np.sum(self.record, axis=1))
            self.MAB.pull(i)


class Epgreedy():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.t = 0
        self.K = self.MAB.get_K()

    def reset(self):
        self.t = 0
        self.MAB.reset()

    def play_one_step(self):
        # Pull each arm once
        if self.t < self.K:
            self.MAB.pull(self.t)
        else:
            # Randomly choose an arm to explore (explore)
            unif = np.random.uniform()
            if unif <= (self.K * math.log(self.t) / self.t) ** (1/3):
                i = np.random.randint(0, self.K)
            # Pull greedy arm (exploit)
            else:
                record = self.MAB.get_record()
                i = random_argmax(record[:, 1] / np.sum(record, axis=1))
            self.MAB.pull(i)
        self.t += 1


class UCB():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.ub_term = math.log(self.MAB.get_K() * self.MAB.get_T() / delta)
        self.t = 0

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.t = 0
        self.MAB.reset()

    def play_one_step(self):
        # Pull each of the arms once
        if self.t < self.MAB.get_K():
            self.MAB.pull(self.t)
            self.t += 1
        else:
            record = self.MAB.get_record()
            counts = np.sum(record, axis=1)
            # Empirical reward estimate array
            reward_est = record[:, 1] / counts

            # Find index with highest upper confidence bound
            i = random_argmax(reward_est + np.sqrt(self.ub_term / counts))
            self.MAB.pull(i)


class Thompson_sampling():
    def __init__(self, MAB):
        self.MAB = MAB
        self.beta_vars = np.ones((self.MAB.get_K(), 2)).astype(int)

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.beta_vars = np.ones((self.MAB.get_K(), 2)).astype(int)

    def play_one_step(self):
        '''
        Implement one step of the Thompson sampling algorithm. 
        '''
        alphas = self.beta_vars[:, 0]
        betas = self.beta_vars[:, 1]
        thetas = np.random.beta(alphas, betas)

        i = random_argmax(thetas)
        reward = self.MAB.pull(i)
        self.beta_vars[i] = [alphas[i] + reward, betas[i] + 1 - reward]


class Gittins_index():
    def __init__(self, MAB, gamma=0.90, epsilon=1e-4, N=100):
        self.MAB = MAB
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = N
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())

        self.curr = 0

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())

        self.curr = 0

    @lru_cache(maxsize=None)
    def calculate_value_oab(self, successes, total_num_samples, lambda_hat, stage_num=0):
        '''
        Helper function for calculating the OAB value. Recursive function
        '''
        succ_prop = successes / total_num_samples
        if stage_num == self.N:
            return self.gamma**self.N / (1 - self.gamma) * max(succ_prop - lambda_hat / (1 - self.gamma), 0)
        return max(succ_prop - lambda_hat / (1 - self.gamma) + self.gamma *
                   (succ_prop * self.calculate_value_oab(successes + 1, total_num_samples + 1, lambda_hat, stage_num + 1) +
                    (1 - succ_prop) * self.calculate_value_oab(successes, total_num_samples + 1, lambda_hat, stage_num + 1)), 0)

    def compute_gittins_index(self, arm_index):
        '''
        Calibration for Gittins Index (Algorithm 1)
        '''
        eta, xi = self.MAB.get_record()[arm_index]
        ub, lb = self.upper_bound, self.lower_bound
        while ub - lb > self.epsilon:
            lambda_hat = (lb + ub) / 2
            value_oab = self.calculate_value_oab(xi + 1, xi + eta + 2, lambda_hat)
            if value_oab > 0:
                lb = lambda_hat
            else:
                ub = lambda_hat

        return ub

    def play_one_step(self):
        '''
        Select the arm with the highest Gittins Index and update its Gittins Index based on the value return by pull 
        '''
        arm = self.curr

        # Pull each arm once
        if self.curr < self.MAB.get_K():
            self.curr += 1
        # Pull the arm with the highest Gittins Index
        else:
            arm = random_argmax(self.gittins_indices)
        
        # print(self.gittins_indices)
        self.MAB.pull(arm)
        self.gittins_indices[arm] = self.compute_gittins_index(arm)
