import numpy as np
from scipy.optimize import minimize
from class_and_func.spectral_functions import *


class univariate_spectral_noised_estimator(object):
    def __init__(self, fixed_parameter, loss=uni_grad_ll_mask, grad=True, initial_guess="random", options=None):
        self.fixed_parameter = fixed_parameter
        self.loss = loss
        self.grad = True  # By default uses grad version of spectral ll
        self.initial_guess = initial_guess

        if options is None:
            self.options = {'disp': False,"maxls":40}
        else:
            self.options = options

    def fit(self, periodogram, max_time):

        #np.random.seed(0)

        K = int(periodogram.shape[0])
        self.dim = (periodogram[0]).shape[0]

        # Bounds
        bounds = [(1e-16, None), (1e-16, 1 - 1e-16), (1e-16, None), (1e-16, None)]

        # Initial point
        if isinstance(self.initial_guess, str) and self.initial_guess == "random":
            np.random.seed()
            init_a = np.random.uniform(0, 3, 3)
            init_alpha = np.random.uniform(0, 1, (self.dim, self.dim))

            init = np.concatenate((init_a[0].ravel(), init_alpha.ravel(), init_a[1:].ravel()))

        # Mask of non-fixed parameters
        indices = [i != self.fixed_parameter[0] for i in range(4)]
        bounds = np.array(bounds)[indices]
        init = init[indices]

        # else:
        #    param_mask = np.array([True]*(self.dim * (2 + self.dim) + 1))
        # Estimation
        self.res = minimize(self.loss,
                            init, tol=1e-8,
                            method="L-BFGS-B", jac=self.grad,
                            args=(periodogram, K, max_time, self.fixed_parameter),
                            bounds=bounds, options=self.options)

        # theta_estim = np.zeros((self.dim * (2+self.dim) + 1))

        # true_indices = np.where(param_mask)[0]
        # theta_estim[true_indices] = self.res.x[:len(true_indices)]

        # self.mu_estim = theta_estim[0: self.dim].reshape((self.dim, 1))
        # self.alpha_estim = theta_estim[self.dim: -self.dim-1].reshape((self.dim, self.dim))
        # self.beta_estim = theta_estim[-self.dim-1:-1].reshape((self.dim, 1))
        # self.noise_estim = theta_estim[-1]

        # return self.mu_estim, self.alpha_estim, self.beta_estim, self.noise_estim
        return self.res


class multivariate_spectral_unnoised_estimator(object):
    def __init__(self, loss=grad_ll_unnoised_mask, grad=True, initial_guess="random", mask=None, options=None):
        self.loss = loss
        self.grad = grad  # By default uses grad version of spectral ll
        self.initial_guess = initial_guess
        self.mask = mask

        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, periodogram, max_time):

        K = int(periodogram.shape[0])
        self.dim = (periodogram[0]).shape[0]

        # Bounds
        bounds = [(1e-16, None)] * self.dim
        bounds += ([(1e-16, 1 - 1e-16)] + [(1e-16, None)] * self.dim) * (self.dim - 1) + [(1e-16, 1 - 1e-16)]
        bounds += [(1e-16, None)] * (self.dim)

        # Initial point
        if isinstance(self.initial_guess, str) and self.initial_guess == "random":
            init_a = np.random.uniform(0, 3, self.dim * 2)

            a = np.random.uniform(0, 3, (self.dim, self.dim))
            radius = np.max(np.abs(np.linalg.eig(a)[0]))
            div = np.random.uniform(1e-16, 1 - 1e-16)
            init_alpha = a * div / (radius)

            self.init = np.concatenate((init_a[0:2].ravel(), init_alpha.ravel(), init_a[2:].ravel()))

        # Mask of parameters
        if self.mask is not None:
            param_mask = np.concatenate(([True] * self.dim, self.mask.ravel(), self.mask.any(axis=1)))
            bounds = np.array(bounds)[param_mask]
            self.init = self.init[param_mask]

        else:
            param_mask = np.array([True]*(self.dim * (2 + self.dim)))

        # Estimation
        self.res = minimize(self.loss,
                            self.init, tol=1e-8,
                            method="L-BFGS-B", jac=self.grad,
                            args=(periodogram, K, max_time, self.mask),
                            bounds=bounds, options=self.options)

        theta_estim = np.zeros((self.dim * (2 + self.dim)))

        true_indices = np.where(param_mask)[0]
        theta_estim[true_indices] = self.res.x[:len(true_indices)]

        self.mu_estim = theta_estim[0: self.dim].reshape((self.dim, 1))
        self.alpha_estim = theta_estim[self.dim: -self.dim].reshape((self.dim, self.dim))
        self.beta_estim = theta_estim[-self.dim:].reshape((self.dim, 1))

        #return self.mu_estim, self.alpha_estim, self.beta_estim, self.noise_estim
        return self.res


class multivariate_spectral_noised_estimator(object):
    def __init__(self, loss=grad_ll_mask, grad=True, initial_guess="random", mask=None, options=None):
        self.loss = loss
        self.grad = grad  # By default uses grad version of spectral ll
        self.initial_guess = initial_guess
        self.mask = mask

        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, periodogram, max_time):

        K = int(periodogram.shape[0])
        self.dim = (periodogram[0]).shape[0]

        # Bounds
        bounds = [(1e-16, None)] * self.dim
        bounds += ([(1e-16, 1 - 1e-16)] + [(1e-16, None)] * self.dim) * (self.dim - 1) + [(1e-16, 1 - 1e-16)]
        bounds += [(1e-16, None)] * (self.dim + 1)

        # Initial point
        if isinstance(self.initial_guess, str) and self.initial_guess == "random":
            init_a = np.random.uniform(0, 3, self.dim * 2 + 1)

            a = np.random.uniform(0, 3, (self.dim, self.dim))
            radius = np.max(np.abs(np.linalg.eig(a)[0]))
            div = np.random.uniform(1e-16, 1 - 1e-16)
            init_alpha = a * div / (radius)

            self.init = np.concatenate((init_a[0:2].ravel(), init_alpha.ravel(), init_a[2:].ravel()))

        # Mask of parameters
        if self.mask is not None:
            param_mask = np.concatenate(([True] * self.dim, self.mask.ravel(), self.mask.any(axis=1), [True]))
            bounds = np.array(bounds)[param_mask]
            self.init = self.init[param_mask]

        else:
            param_mask = np.array([True]*(self.dim * (2 + self.dim) + 1))

        # Estimation
        self.res = minimize(self.loss,
                            self.init, tol=1e-16,
                            method="L-BFGS-B", jac=self.grad,
                            args=(periodogram, K, max_time, self.mask),
                            bounds=bounds, options=self.options)

        theta_estim = np.zeros((self.dim * (2 + self.dim) + 1))

        true_indices = np.where(param_mask)[0]
        theta_estim[true_indices] = self.res.x[:len(true_indices)]

        self.mu_estim = theta_estim[0: self.dim].reshape((self.dim, 1))
        self.alpha_estim = theta_estim[self.dim: -self.dim - 1].reshape((self.dim, self.dim))
        self.beta_estim = theta_estim[-self.dim - 1:-1].reshape((self.dim, 1))
        self.noise_estim = theta_estim[-1]

        #return self.mu_estim, self.alpha_estim, self.beta_estim, self.noise_estim
        return self.res