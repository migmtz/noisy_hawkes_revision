from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import spectral_w_mask, fast_multi_periodogram2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return (a**2)/(b**2 + (2*np.pi*x)**2) + c**2

def der(x, a, b, c):
    return -a * (8 *x * np.pi**2)/((b + (2*np.pi*x)**2)**2)


if __name__ == "__main__":
    sns.set_theme()

    np.random.seed(2)
    mu = np.array([[1.0]])
    alpha = np.array([[0.5]])
    beta = np.array([[1.0]])

    print("average intensity:", mu/(1-alpha))
    print("bump at 0:", mu/((1-alpha)**3))

    noise = 1.0

    avg_intensity = noise + mu / (1 - alpha)
    print("avg_intensity", avg_intensity)

    max_time = 8000.0
    burn_in = -100

    nb_freq = 5000 # K
    max_freq = 10.0 # W
    x = np.arange(-nb_freq, nb_freq+1) * max_freq / nb_freq
    #print(x, x.shape, x[nb_freq], x[nb_freq+1], 1/nb_freq)

    f_x = np.array([spectral_w_mask((mu, alpha, beta, noise), x_0)[0] for x_0 in x])
    IT_x = np.zeros(x.shape)#, dtype=np.complex128)
    debiaised_IT_x = np.zeros(x.shape)#, dtype=np.complex128)
    debiaised_empirical_IT_x = np.zeros(x.shape)#, dtype=np.complex128)

    repet = 1
    for i in range(repet):
        np.random.seed(i)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, burn_in=burn_in, max_time=max_time)
        hp.simulate()
        hp_times = hp.timestamps

        #times = np.array(hp.timestamps + [hp.timestamps[-1]])
        pp = multivariate_exponential_hawkes(np.array([[noise]]), 0 * alpha, beta, burn_in=burn_in, max_time=max_time)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

        #times = np.array([0.0] + [t for t in hp.timestamps if t > 0] + [max_time])

        #IT_x += np.array([bartlett_periodogram(x_0, parasited_times) for x_0 in x])
        IT_x[nb_freq+1:] += fast_multi_periodogram2(nb_freq, parasited_times, max_time, max_freq).squeeze().real
        #IT_x[nb_freq] += ((len(parasited_times) - 2)**2)/max_time
        #debiaised_IT_x += np.array([debiaised_bartlett_periodogram(x_0, parasited_times, avg_intensity) for x_0 in x])
        #empirical_avg = (len(parasited_times) - 2)/max_time
        #debiaised_empirical_IT_x += np.array([debiaised_bartlett_periodogram(x_0, parasited_times, empirical_avg) for x_0 in x])
    # conv_IT_x = np.convolve((1 / 50) * np.ones(50), IT_x / repet)[:-49]
    # cum_IT_x = [np.mean((IT_x/repet)[:i+1]) for i in range(len(IT_x))]

    IT_x[np.arange(nb_freq - 1, -1, -1)] = (IT_x[nb_freq + 1:])
    IT_x /= repet

    print(IT_x[nb_freq])

    largo = nb_freq // 25
    print("largo", largo)
    # debiaised_IT_x /= repet
    # debiaised_empirical_IT_x /= repet
    # print(IT_x[0])
    #
    # smooth_rolling = []
    smooth_rolling_centered = []
    general_weights = np.array([stats.binom.pmf(i, 2 * largo, 0.5) for i in range(2 * largo+1)])
    for i in range(0, len(IT_x)):
    #     smooth_rolling += [np.mean(IT_x[max(i - largo, 0): min(i + largo, len(IT_x))+1])]
    #
        aux = IT_x[max(i - largo, 0): min(i + largo, len(IT_x))+1]
        if i < largo:
            weights = general_weights[largo-i:] / stats.binom.cdf(largo + i, 2*largo, 0.5)
        elif i >= len(IT_x) - largo-1:
            weights = general_weights[(largo +i-len(IT_x))+1:] / stats.binom.cdf((largo -i+len(IT_x)-1), 2*largo, 0.5)
            weights = np.flip(weights)
        else:
            weights = general_weights

        smooth_rolling_centered += [np.average(aux, weights=weights)]
    #
    # double_rolling = []
    # double_rolling_binomial = []
    # for i in range(0, len(IT_x)):
    #     double_rolling += [np.mean(smooth_rolling[max(i - largo, 0): min(i + largo, len(smooth_rolling))+1])]
    #     double_rolling_binomial += [np.mean(smooth_rolling_centered[max(i - largo, 0): min(i + largo, len(smooth_rolling_centered))+1])]
    #
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(15,6))
    #
    ax[0].plot(x, f_x, label="Spectral density")
    ax[0].plot(x, IT_x, c="r", alpha=0.5, label="Periodogram")
    #
    # ax[1].plot(x, f_x, label="Spectral density")
    # ax[1].plot(x, smooth_rolling, label="Rolling average smoothing")
    #
    ax[2].plot(x, f_x, label="Spectral density")
    ax[2].plot(x, smooth_rolling_centered, label="Rolling weighted average smoothing", alpha=0.5, zorder=0)

    # ax[1].legend()
    # ax[2].legend()
    #
    # #plt.savefig("non_parametric_estimation.pdf", format="pdf", bbox_inches="tight")

    popt, _ = curve_fit(func, x[nb_freq+1:], smooth_rolling_centered[nb_freq+1:])
    a,b,c = popt

    xaux = x[nb_freq:]
    yaux = func(x[nb_freq:], *popt)

    ax[2].plot(xaux, yaux, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    #
    ax[2].legend()

    epsilon = 0.001 * a / (b**2)
    aux_freq = np.abs(func(x[nb_freq:], *popt) - c)
    idx_max_freq = np.argmax(aux_freq < epsilon)
    est_max_freq = x[nb_freq + 1 + idx_max_freq]
    ax[2].plot([est_max_freq, est_max_freq], ax[2].get_ylim())
    print("Cut", epsilon)
    print("Estimated max freq:", idx_max_freq, est_max_freq, der(x[nb_freq + idx_max_freq], *popt), der(x[nb_freq + 1+idx_max_freq], *popt))
    print("Explicit estimated:", np.sqrt(a / epsilon - b**2) /(2*np.pi))
    #print(np.sqrt((1/0.1 - 1) * b) /(2*np.pi))

    print("*"*200)
    mu_est = (a*np.sqrt(b**2) / beta)/(beta**2 - b**2)
    alpha_est = 1 - np.sqrt(b**2) / beta
    noise_est = c - a/(beta**2 - b**2)
    print(popt)
    print(mu_est, alpha_est, noise_est)

    plt.show()
