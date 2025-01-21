import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()


if __name__ == "__main__":

    parameters_full = [r"$\mu_1$", r"$\mu_2$", r"$\alpha_{11}$", r"$\alpha_{12}$", r"$\alpha_{21}$", r"$\alpha_{22}$",
                       r"$\beta_1$", r"$\beta_2$", r"$\lambda_0$"]


    # Column interactions boxplots

    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.5, 0.0],
                      [0.4, 0.4]])
    beta = np.array([[1.0],
                     [1.3]])
    noise = 0.5

    theta = np.concatenate((mu.ravel(), alpha.ravel(), beta.ravel(), np.array([noise])))

    column_estimations = pd.read_csv('saved_estimations/multivariate_triangle_3000.csv', sep=',', header=None, index_col=None)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    bplot = ax.boxplot(column_estimations)
    _ = ax.scatter(range(1, 10), theta, marker="*", s=100)
    #ax.set_xticklabels(parameters_full)

    print("Proportion of null:", np.mean(column_estimations < 2e-16, axis=0))
    _ = ax.set_title("Full model")
    ax.set_ylim((-0.1, 2.0))
    plt.show()