import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr


class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array) with shape (4,) 
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation

    def compute_approx_hessian(self, Q, M, R, delta=1e-7):
        """
        This function computes an approximation of Q, R, and M such that the Hessian
        matrix H is positive definite, and thus invertible.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD.
            M (2d numpy array): A numpy array with shape (n_s, n_a)
            R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
        Returns:
            Q_approx (2d numpy array): A numpy array with shape (n_s, n_s)
            M_approx (1d numpy array): A numpy array with shape (n_s, n_a)
            R_approx (2d numpy array): A numpy array with shape (n_a, n_a)
        """
        H = np.block([[Q, M], [M.T, R]])
        w, v = np.linalg.eig(H)
        H_approx = delta * np.identity(H.shape[0])

        for i, vec in enumerate(w):
            H_approx += vec * v[:, i:i + 1] @ v[:, i:i + 1].T

        (q_1, q_2), (m_1, m_2), (r_1, r_2) = Q.shape, M.shape, R.shape

        return H_approx[:q_1, :q_2], H_approx[:m_1, q_2:q_2 + m_2], H_approx[q_1:q_1 + r_1, q_2:q_2 + r_2]

    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylor expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimal policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        def f_s(s): return self.f(s, a_star)
        def f_a(a): return self.f(s_star, a)
        def c_s(s): return self.c(s, a_star)
        def c_a(a): return self.c(s_star, a)

        A = jacobian(f_s, s_star)
        B = jacobian(f_a, a_star)

        q = gradient(c_s, s_star)
        r = gradient(c_a, a_star)

        Q = hessian(c_s, s_star)
        R = hessian(c_a, a_star)
        M = hessian(lambda sa: self.c(
            sa[:-1], sa[-1:]), np.concatenate((s_star, a_star)))[:-1, -1:]

        Q_2 = Q / 2
        R_2 = R / 2

        q_2 = (q.T - s_star.T @ Q - a_star.T @ M.T).T
        q_2 = q_2.reshape(q_2.shape[0], 1)

        r_2 = (r.T - a_star.T @ R - s_star.T @ M).T
        r_2 = r_2.reshape(r_2.shape[0], 1)

        b = self.c(s_star, a_star) + s_star.T @ Q @ s_star / 2 + a_star.T @ R @ a_star / \
            2 + s_star.T @ M @ a_star - q.T @ s_star - r.T @ a_star
        b = b.flatten()

        m = self.f(s_star, a_star) - A @ s_star - B @ a_star
        m = m.reshape(m.shape[0], 1)

        Q_2, M, R_2 = self.compute_approx_hessian(Q_2, M, R_2)

        return lqr(A, B, m, Q_2, R_2, M, q_2, r_2, b, T)


class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + \
            self.D * (err - self.err_prev)
        self.err_prev = err
        return a
