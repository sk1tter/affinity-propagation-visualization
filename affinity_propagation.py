from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from time import time
from typing import Any, Tuple
from stqdm import stqdm
import streamlit as st


def compute_responsibility(
    S: np.ndarray, R: np.ndarray, A: np.ndarray, damping_factor: float = 0.5
) -> np.ndarray:
    """
    S: n by n matrix of similarities
    R: n by n matrix of current responsibilities
    A: n by n matrix of current availabilities
    damping_factor: Damping factor used to calculate new responsibilities. Acts as weight
                for the weighted sum of the old and new responsibilities.
    Returns matrix containing the new responsibilities.
    """
    n = S.shape[0]

    AS = S + A
    rows = np.arange(n)
    np.fill_diagonal(AS, -np.inf)

    idx_max = np.argmax(AS, axis=1)
    first_max = AS[rows, idx_max]

    AS[rows, idx_max] = -np.inf
    second_max = AS[rows, np.argmax(AS, axis=1)]

    max_matrix = np.zeros_like(R) + first_max[:, None]
    max_matrix[rows, idx_max] = second_max

    new_R = R * damping_factor + (1 - damping_factor) * (S - max_matrix)
    return new_R


def compute_availability(
    R: np.ndarray, A: np.ndarray, damping_factor: float = 0.5
) -> np.ndarray:
    """
    S: n by n matrix of similarities
    R: n by n matrix of current responsibilities
    A: n by n matrix of current availabilities
    damping_factor: Damping factor used to calculate new availabilities. Acts as weight
                for the weighted sum of the old and new availabilities.
    Returns matrix containing the new availabilities.
    """
    n = R.shape[0]
    k_idx = np.arange(n)
    a = np.array(R)
    a[a < 0] = 0
    np.fill_diagonal(a, 0)
    a = a.sum(axis=0)
    a = a + R[k_idx, k_idx]

    a = np.ones(A.shape) * a

    a -= np.clip(R, 0, np.inf)
    a[a > 0] = 0

    a_ = np.array(R)
    np.fill_diagonal(a_, 0)

    a_[a_ < 0] = 0

    a[k_idx, k_idx] = a_.sum(axis=0)

    new_A = A * damping_factor + (1 - damping_factor) * a

    return new_A


def similarity(u: np.ndarray, v: np.ndarray):
    return -((u - v) ** 2).sum()


def create_matrices(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = np.zeros((X.shape[0], X.shape[0]))
    R = np.array(S)
    A = np.array(S)

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            S[i, j] = similarity(X[i], X[j])

    return A, R, S


def give_preferences(
    S: np.ndarray, preference: Any = "median"
) -> Tuple[np.ndarray, int]:
    if preference == "median":
        indices = np.where(~np.eye(S.shape[0], dtype=bool))
        m = np.median(S[indices])
    elif type(preference) == np.ndarray:
        m = preference
    else:
        try:
            m = float(preference)
        except ValueError:
            raise ValueError(
                "Parameter 'preference' must either be 'median', a np.ndarray or a scalar."
            )

    np.fill_diagonal(S, m)
    return S, m


def affinity_prop(
    X: np.ndarray,
    maxiter: int = 100,
    preference: Any = "median",
    damping_factor: float = 0.5,
    local_thresh: int = 10,
    data_plot=None,
    fig=None,
    ax=None,
    c: st.empty = None,
):
    """
    Params:
        X: Input matrix with data to cluster.
        maxiter: Maximum iterations after which to stop the clustering if it
                 does not converge before.
        preference: Either 'median' (default), a vector of the same size as the input data,
                    or a fixed scalar value. Determines the initial "preferences", i.e., self-similarities:
                    values on the diagonal of the similarity matrix. Details in the README.
        damping_factor: Damping factor used to calculate new availabilities and
                        responsibilities. Acts as weight for the weighted sum of
                        the old and new availabilities or responsibilities.
        local_thresh: Number of iterations without any change in the outcome labels before the algorithm stops.
        data_plot: If not None, a function that takes a figure and axes as input and plots the data.
        fig: If not None, the figure to plot the data on.
        ax: If not None, the axes to plot the data on.
        c: If not None, the container to print logs on.
    """

    X = np.asarray(X)
    A, R, S = create_matrices(X)
    S, p = give_preferences(S, preference=preference)
    log = ""
    log += f"Initial preferences: {p}  \n  "

    count_equal = 0
    i = 0
    converged = False

    for i in stqdm(range(maxiter), st_container=st.sidebar):
        c.write(log)
        E_old = R + A
        labels_old = np.argmax(E_old, axis=1)
        R = compute_responsibility(S, R, A, damping_factor=damping_factor)
        A = compute_availability(R, A, damping_factor=damping_factor)
        E_new = R + A
        labels_cur = np.argmax(E_new, axis=1)

        if i % 5 == 0:
            exemplars = plot_iteration_n_get_exemplars(
                X, labels_cur, data_plot, fig, ax, c
            )
            log += f"Iteration {i}: found {len(exemplars)} exemplars  \n  "

        if np.all(labels_cur == labels_old):
            count_equal += 1
        else:
            count_equal = 0

        if count_equal > local_thresh:
            converged = True
            break

    if converged:
        log += f"Converged after {i} iterations.  \n  "
    else:
        log += f"Did not converge after {i} iterations.  \n  "
    c.write(log)

    E = R + A
    plot_iteration_n_get_exemplars(X, labels_cur, data_plot, fig, ax, c)
    labels = np.argmax(E, axis=1)
    exemplars = np.unique(labels)
    centers = X[exemplars]
    c.write(log + f"{len(exemplars)} exemplars found.")

    return exemplars, labels, centers


def plot_iteration_n_get_exemplars(
    data: np.ndarray, labels, data_plot=None, fig=None, ax: plt.Axes = None, c=None
) -> np.ndarray:
    exemplars = np.unique(labels)
    colors = dict(zip(exemplars, cycle("bgrcmyk")))
    ax.clear()
    for i in range(len(labels)):
        if i in exemplars:
            exemplar = i
            edge = "k"
            ms = 10
        else:
            exemplar = labels[i]
            ms = 3
            edge = None
            ax.plot(
                [data[i][0], data[exemplar][0]],
                [data[i][1], data[exemplar][1]],
                c=colors[exemplar],
            )
        ax.plot(
            data[i][0],
            data[i][1],
            "o",
            c=colors[exemplar],
            markersize=ms,
            markeredgecolor=edge,
        )
    data_plot.pyplot(fig)
    return exemplars
