import pandas as pd
import numpy as np
import finufft
from scipy.linalg import inv, det
from scipy.optimize import minimize
import time


def fast_multi_periodogram(K, tList, max_time, precision=1e-9):
    dim = int(np.max(np.array(tList)[:, 1]))
    dimensional_times = [[t for t, i in tList if i == j] for j in range(1, dim + 1)]

    # put K for w=0
    aux = np.array([finufft.nufft1d1(2 * np.pi * np.array(x) / max_time, np.ones(len(x)) + 0j, n_modes=2 * K + 1,
                                     isign=-1, eps=1e-9)[K + 1:] for x in dimensional_times])
    aux = (aux.T)[:, :, np.newaxis]
    aux = (aux @ np.transpose(np.conj(aux), axes=(0, 2, 1))) / max_time

    return aux


def ei(size, index):
    e = np.zeros((size))
    e[index] = 1.0
    return e


def grad_unnoised_completw(theta, w, periodogramw):  # better version
    mu0, alpha0, beta0 = theta
    alpha0 = np.nan_to_num(alpha0)

    dim = mu0.shape[0]
    a = inv(np.identity(dim) - np.nan_to_num(alpha0))
    #print("jjj", alpha0, np.linalg.eig(alpha0)[0], a)

    mean_matrix = np.identity(dim) * (a @ mu0)
    #print(mean_matrix)

    fourier_matrix = alpha0 * beta0 / (beta0 + 2j * np.pi * w)
    spectral_matrix = inv(np.identity(dim) - fourier_matrix)

    f_theta = (spectral_matrix) @ mean_matrix @ np.conj(spectral_matrix.T)
    f_inv = inv(f_theta)
    #f_inv = np.conj(fourier_matrix.T) @ (1/mean_matrix)
    #try:
    #    f_inv = inv(f_theta + np.random.normal(0, 1e-12,(dim,dim)) * np.identity(dim))
    #except:
    #    print(a, f_theta, det(f_theta))

    ll = np.log(det(f_theta)) + np.trace(f_inv @ periodogramw)

    dmu = np.zeros((dim, dim, dim), dtype=np.complex128)
    dalpha = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    dbeta = np.zeros((dim, dim, dim), dtype=np.complex128)
    aux_dbeta = alpha0 * (2j * np.pi * w) * np.repeat(1 / (beta0 + 2j * np.pi * w) ** 2, dim, axis=1)

    dmu = a @ np.array([ei(((dim, dim)), i) for i in range(dim)]) * np.array([np.identity(dim)] * dim)
    # print(dmu)
    dbeta = aux_dbeta * np.array([ei((dim, dim), i) for i in range(dim)])

    dij = np.array([[ei(dim, i)[:, np.newaxis] * ei(dim, j)[np.newaxis, :] for j in range(dim)] for i in range(dim)])

    # dij = np.array([[ei(dim,i)[:,np.newaxis,np.newaxis,np.newaxis] * ei(dim,j)[np.newaxis,:,:,:] for j in range(dim)] for i in range(dim)])
    # print(dij.shape)
    dalpha_cent = a @ dij @ a @ mu0
    dalpha_cent = dalpha_cent * np.array([[np.identity(dim)] * dim] * dim)
    dalpha_bord = dij * beta0 / (beta0 + 2j * np.pi * w)

    dmu = spectral_matrix @ dmu @ np.conj(spectral_matrix.T)
    dalpha = (spectral_matrix @ dalpha_bord @ f_theta) + (
                f_theta @ np.transpose(np.conj(dalpha_bord), axes=(0, 1, 3, 2)) @ np.conj(spectral_matrix.T))
    dalpha += spectral_matrix @ dalpha_cent @ np.conj(spectral_matrix.T)
    dbeta = (spectral_matrix @ dbeta @ f_theta) + (
                f_theta @ np.transpose(np.conj(dbeta), axes=(0, 2, 1)) @ np.conj(spectral_matrix.T))

    aux_det = f_inv.T

    aux_trace_mu = (aux_det.T) @ dmu @ (aux_det.T)
    aux_trace_alpha = (aux_det.T) @ dalpha @ (aux_det.T)
    aux_trace_beta = (aux_det.T) @ dbeta @ (aux_det.T)

    dmu = np.sum(aux_det * dmu, axis=(1,2)) - np.sum((periodogramw.T) * aux_trace_mu,
                                                                        axis=(1,2))
    dalpha = np.sum(aux_det * dalpha, axis=(2, 3)) - np.sum((periodogramw.T) * aux_trace_alpha, axis=(2, 3))
    dbeta = np.sum(aux_det * dbeta, axis=(1,2)) - np.sum((periodogramw.T) * aux_trace_beta,
                                                                            axis=(1,2))
    # print(dnoise)
    grad_final = np.concatenate((dmu.real.ravel(), dalpha.real.ravel(), dbeta.real.ravel()))

    return np.concatenate((np.array([ll.real]), grad_final))


# Grad of loglikelihood
def grad_unnoised_ll(theta, periodogram, K, max_time):
    dim = int(np.sqrt(theta.shape[0] + 1) - 1)
    theta_aux = (theta[:dim].reshape((dim, 1)), theta[dim:dim + dim ** 2].reshape((dim, dim)), theta[-dim:].reshape((dim, 1)))

    ll = np.sum([grad_unnoised_completw(theta_aux, j / max_time, periodogram[j - 1]) for j in range(1, K + 1)], axis=0)
    ll /= max_time

    return (ll[0], ll[1:])


def job(it, periodo):
    np.random.seed(it)
    K = int(periodo.shape[0])
    dim = (periodo[0]).shape[0]

    bounds = [(1e-16, None)] * dim
    #bounds += ([(1e-16, 1 - 1e-16)] + [(1e-16, None)] * dim) * (dim-1) + [(1e-16, 1 - 1e-16)]
    bounds += [(1e-16, 0.5 - 1e-16)] * (dim**2)
    bounds += [(1e-16, None)] * dim

    init_a = np.random.chisquare(3, dim * 2)

    a = np.random.chisquare(3, (dim, dim))
    radius = np.max(np.abs(np.linalg.eig(a)[0]))
    div = np.random.uniform(1e-16, 1 - 1e-16)
    init_alpha = a * div / (radius)

    init = np.concatenate((init_a[0:2].ravel(), init_alpha.ravel(), init_a[2:].ravel()))

    res = minimize(grad_unnoised_ll,
                   init, tol=1e-16,
                   method="L-BFGS-B", jac=True,
                   args=(periodo, K, max_time),
                   bounds=bounds, options={"disp": False})

    print('-', end='')
    # print(res.x)

    return res.x


if __name__ == "__main__":
    df = pd.read_csv('../real_data_application/spk_mouse22.csv', sep=',', header=0, index_col=0)
    print(df.values)

    max_time = 725
    dim = 3
    repetitions = 5
    K_func = lambda x: int(x)
    #K_func = lambda x : int(x * np.log(x))

    data = [df.values[df.values[:, 2] == i, 0:2] for i in range(1,repetitions+1)]

    variates_idx = {4.0:1, 5.0:2, 6.0:3}

    data = [[(t - i * max_time, variates_idx[m]) for (t,m) in data[i]] for i in range(0,repetitions)]

    print("# of points per repetition:", [len(u) for u in data])

    for u in data:
        aux = np.array(u)
        nb_1, nb_2, nb_3 = np.sum(aux[:,1] == 1), np.sum(aux[:,1] == 2), np.sum(aux[:,1] == 3)
        print("# of points per dimension:", nb_1, nb_2, nb_3)

    periodogram_list = [fast_multi_periodogram(K_func(len(u)), u, max_time) for u in data]

    print("K:", [K_func(len(u)) for u in periodogram_list])

    res = np.zeros((repetitions, 2*dim + dim*dim))
    print("|" + "-"*repetitions + "|")
    print("|", end="")
    for it, periodo in zip(range(repetitions), periodogram_list):
        start_time = time.time()
        res[it, :] = job(it, periodo)
        end_time = time.time()
    print("|")
    print(res)

    #np.savetxt("realdata_estimation_unnoised_N.csv", res, delimiter=",")