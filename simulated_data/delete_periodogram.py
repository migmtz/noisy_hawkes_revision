import numpy as np
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import univariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram, spectral_f_exp_noised
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time


def job(it, periodo, max_time, fixed_parameter):
    np.random.seed(it)

    estimator = univariate_spectral_noised_estimator(fixed_parameter)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


if __name__ == "__main__":

    alpha = np.array([[0.5]])
    beta = np.array([[1.0]])

    noise_list = [0.4 * i for i in range(1, 11)]
    avg_total_intensity = 6.0

    max_time = 8000
    burn_in = -100
    repetitions = 1

    K_func = lambda x: int(x)

    fixed_list = ["mu", "alpha", "beta", "noise"]

    for noise in noise_list:
        mu = (1-alpha) * (avg_total_intensity - noise)

        theta = np.concatenate((mu, alpha, beta, np.array([[noise]])))
        fixed_parameter_list = [(i, theta[i]) for i in range(4)]

        # Simulations and periodograms

        x = np.linspace(-5.0, 5.0, 1000)
        y = np.array([spectral_f_exp_noised(x0, (mu, alpha, beta, noise)) for x0 in x]).squeeze()

        fig, ax = plt.subplots()

        ax.plot(x, y)
        aux_mean = (mu/(1-alpha) + noise).squeeze()
        ax.plot([aux_mean, aux_mean], [0.0, 10.0])

        plt.show()