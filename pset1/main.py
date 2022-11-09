from env_MAB import *
from algorithms import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Number of runs
    R = 30

    # Create MAB instance
    mab = MAB()

    # The list of algorithms
    explore = Explore(mab)
    greedy = Greedy(mab)
    etc = ETC(mab)
    epgreedy = Epgreedy(mab)
    ucb = UCB(mab)
    thompson = Thompson_sampling(mab)
    gittins = Gittins_index(mab)
    alg_list = [explore, greedy, etc, epgreedy, ucb, thompson, gittins]

    fig, ax = plt.subplots(3,3,figsize=(12,12),tight_layout=True)
    for i,alg in enumerate(alg_list):
        print (alg)
        regret_list = np.zeros((R,mab.get_T()))
        # Run many times
        for r in range(R):
            # Run the algorithm for T steps
            for t in range(mab.get_T()):
                alg.play_one_step()

            # Collect regrets
            regret_list[r,:] = mab.get_regrets()

            # Reset
            alg.reset()

        plt.subplot(3,3,i+1)
        plt.plot(range(mab.get_T()),np.mean(regret_list,axis=0))
        plt.fill_between(range(1,1+mab.get_T()), np.percentile(regret_list,5,axis=0), np.percentile(regret_list,95,axis=0), color='b', alpha=.1)
        plt.xlabel('Step')
        plt.ylabel('Cumulative Regret')
        plt.title(type(alg).__name__)
    plt.savefig('performances.png')
    plt.show()