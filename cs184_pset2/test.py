from MDP import build_mazeMDP
from DP import DynamicProgramming
import numpy as np
from numpy.testing import assert_allclose


initial_pi = np.array([2, 3, 3, 2, 1, 1, 0, 3, 1, 2, 3, 0, 0, 0, 1, 2, 1])
initial_V_pi = np.array([-101.26098962,  -56.88928761,  -34.74192403,  -33.2796497,
                         -144.02993153, -253.48361078,  -54.199026,  -28.57074442,
                         -138.50808554, -216.64718397,  -31.19557429,  -28.37275726,
                         -136.53728281, -182.10627299,  -36.66706639,  100.,
                         0.])


initial_Q_pi = np.array([[-92.98262845, -125.4870243, -101.26098962,  -67.53849609],
                         [-57.15613538, -206.56782099, -
                             107.44377746,  -56.88928761],
                         [-35.9699113,  -50.75730881,  -
                             52.68524883,  -34.74192403],
                         [-32.75458328,  -29.17581526,  -33.2796497,  -32.16832121],
                         [-115.72213863, -144.02993153, -
                             133.2408101, -216.42560634],
                         [-132.06760955, -253.48361078, -
                             205.44871276, -137.17722456],
                         [-54.199026,  -51.50380021, -199.91160654,  -28.9778281],
                         [-34.15566196,  -30.42642371,  -
                             48.04823842,  -28.57074442],
                         [-144.20249857, -138.50808554, -
                             132.92003037, -192.30574518],
                         [-278.76939188, -224.52261516, -
                             216.64718397, -135.08967542],
                         [-65.46815418,  -52.14386487, -
                             174.28413859,  -31.19557429],
                         [-28.37275726,   69.3410085,  -
                             17.92285718,  -15.77751624],
                         [-136.53728281, -135.03947274, -
                             130.89764493, -165.53007747],
                         [-182.10627299, -155.85518065, -
                             142.64991335,  -66.74854887],
                         [-32.5087324,  -36.66706639, -
                             145.84771834,   68.55304913],
                         [100.,  100.,  100.,  100.],
                         [0.,    0.,    0.,    0.]])

# This is the policy pi(s) = argmax_a Q(s,a) run on initial_Q_pi
max_pi = np.array([3, 3, 3, 1, 0, 0, 3, 3, 2, 3, 3, 1, 2, 3, 3, 0, 0])

optimal_pi = np.array([3, 3, 3, 1, 1, 3, 3, 1, 1, 1, 3, 1, 3, 3, 3, 0, 0])


def test_value_function(dp):

    Q_next = np.array([[-64.14989648, -100.76946481,  -69.73901733,  -61.64561888],
                       [-53.95249849, -111.08802317,  -70.28016226,  -45.35476749],
                       [-35.58004703,  -31.19933413,  -50.28923503,  -29.22699605],
                       [-29.24580483,  -28.78595099,  -32.88978543,  -28.65954277],
                       [-75.86928311, -125.55924916, -107.99238538, -120.41494327],
                       [-126.98235542, -186.41465016, -
                           176.18682685, -110.26115084],
                       [-42.66450589,  -39.96928009, -107.6354456,  -28.9778281],
                       [-28.64073399,   46.23205207,  -19.207456,  -18.8980724],
                       [-114.40974741, -125.9431322, -125.44810252, -127.09703276],
                       [-185.9623657, -136.31987958, -189.90675813, -112.59617151],
                       [-29.26927271,   44.85419399,  -99.90850732,   55.45881246],
                       [-19.08994951,   78.62381625,  -17.92285718,   58.48494574],
                       [-120.7956115, -119.25859856, -125.5448893,  -76.7915763],
                       [-109.59088992,  -57.65163374, -
                           119.65684146,   31.92568603],
                       [-21.54974861,   54.2592052,  -48.17993703,   78.54896011],
                       [100.,  100.,  100.,  100.],
                       [0.,    0.,    0.,    0.]])
    VI_final_V = np.array([52.9822413,  58.65393575,  71.80545601,  77.09256985,
                           46.03711716,  -5.15326234,  77.8311864,  84.14137428,
                           56.7814831,   1.29818168,  84.86720677,  91.78162855,
                           68.76880765,  76.10752288,  91.78162855, 100.,
                           0.])
    assert_allclose(dp.computeVfromQ(initial_Q_pi, initial_pi), initial_V_pi)
    assert_allclose(dp.computeQfromV(initial_V_pi), initial_Q_pi)
    assert_allclose(dp.extractMaxPifromV(initial_V_pi), max_pi)
    assert_allclose(dp.extractMaxPifromQ(initial_Q_pi), max_pi)
    assert_allclose(dp.valueIterationStep(initial_Q_pi), Q_next)

    [student_pi, student_V, student_nIterations,
        student_epsilon] = dp.valueIteration(initial_Q_pi)
    assert_allclose(student_pi, optimal_pi, )
    assert_allclose(student_V, VI_final_V)
    assert_allclose(student_nIterations, 20)
    assert_allclose(student_epsilon, 0.00542463948090699)


def test_policy_evaluation(dp):
    exact_initial_V = np.array([-101.26098962,  -56.88928761,  -34.74192403,  -33.2796497,
                                -144.02993153, -253.48361078,  -54.199026,  -28.57074442,
                                -138.50808554, -216.64718397,  -31.19557429,  -28.37275726,
                                -136.53728281, -182.10627299,  -36.66706639,  100.,
                                0.])
    approx_initial_V = np.array([-101.09100199,  -56.78515927,  -34.6444619,  -33.18268852,
                                 -143.86493206, -253.32508411,  -54.09584839,  -28.47603758,
                                 -138.34287712, -216.48307279,  -31.10197514,  -28.27838316,
                                 -136.37254398, -181.95013164,  -36.58783542,  100.,
                                 0.])

    student_approx_V, approx_iters, approx_epsilon = dp.approxPolicyEvaluation(
        initial_pi)

    assert_allclose(student_approx_V, approx_initial_V)
    assert_allclose(approx_iters, 125)
    assert_allclose(approx_epsilon, 0.00947376485538598)
    assert_allclose(dp.exactPolicyEvaluation(initial_pi), exact_initial_V)

    exact_optimal_V = np.array([52.98550685,  58.65553358,  71.8062328,  77.09295576,
                                46.0387177,  -5.15241096,  77.83151901,  84.14149059,
                                56.78226127,   1.29851475,  84.86730581,  91.78165089,
                                68.76919414,  76.10763931,  91.78165089, 100.,
                                0.])
    approx_optimal_V = np.array([52.97610646,  58.65093276,  71.80399364,  77.09184455,
                                 46.03414993,  -5.15486471,  77.83056159,  84.14115524,
                                 56.78003818,   1.29756299,  84.86702011,  91.78158659,
                                 68.76809005,  76.10730541,  91.78158659, 100.,
                                 0.])

    student_approx_V, approx_iters, approx_epsilon = dp.approxPolicyEvaluation(
        optimal_pi)

    assert_allclose(dp.exactPolicyEvaluation(optimal_pi), exact_optimal_V)
    assert_allclose(student_approx_V, approx_optimal_V)
    assert_allclose(approx_iters, 20)
    assert_allclose(approx_epsilon, 0.00882642348712892)


def test_policy_iteration(dp):
    pi_next = np.array([3, 3, 3, 1, 0, 0, 3, 3, 2, 3, 3, 1, 2, 3, 3, 0, 0])

    assert_allclose(dp.policyIterationStep(initial_pi, True), pi_next)
    assert_allclose(dp.policyIterationStep(initial_pi, False), pi_next)

    exact_optimal_V = np.array([52.98550685,  58.65553358,  71.8062328,  77.09295576,
                                46.0387177,  -5.15241096,  77.83151901,  84.14149059,
                                56.78226127,   1.29851475,  84.86730581,  91.78165089,
                                68.76919414,  76.10763931,  91.78165089, 100.,
                                0.])
    approx_optimal_V = np.array([52.97610646,  58.65093276,  71.80399364,  77.09184455,
                                 46.03414993,  -5.15486471,  77.83056159,  84.14115524,
                                 56.78003818,   1.29756299,  84.86702011,  91.78158659,
                                 68.76809005,  76.10730541,  91.78158659, 100.,
                                 0.])

    student_final_exact = dp.policyIteration(initial_pi, True)
    student_final_approx = dp.policyIteration(initial_pi, False)

    assert_allclose(student_final_exact[0], optimal_pi)
    assert_allclose(student_final_exact[1], exact_optimal_V)
    assert_allclose(student_final_exact[2], 5)

    assert_allclose(student_final_approx[0], optimal_pi)
    assert_allclose(student_final_approx[1], approx_optimal_V)
    assert_allclose(student_final_approx[2], 5)


'''
Where we test all methods: value iteration, policy iteration, and policy evaluation.
'''


def main():
    mdp = build_mazeMDP()
    dp = DynamicProgramming(mdp)
    test_value_function(dp)
    test_policy_evaluation(dp)
    test_policy_iteration(dp)


if __name__ == '__main__':
    main()
