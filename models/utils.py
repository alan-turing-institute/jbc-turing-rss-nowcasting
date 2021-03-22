import torch as t
import numpy as np
import math


def betaln(a, b):
    return t.lgamma(a) + t.lgamma(b) - t.lgamma(a + b)


def binomln(a, b):
    return t.lgamma(a + 1) - t.lgamma(b + 1) - t.lgamma(a - b + 1)


def kappa_marginal_moments(
    prior_kappa_loc, prior_kappa_scale, random_walk_scale, lambda_particles
):
    loc = prior_kappa_loc * lambda_particles
    scale = t.sqrt(random_walk_scale ** 2 + (lambda_particles * prior_kappa_scale) ** 2)
    return loc, scale

def random_walk_scale_estimator(observations, max_lag=4):
    my_obs = [x[-1].item() for x in observations]
    windowed_mean = np.array([(my_obs[i] + obs + my_obs[i+2])/3 for i,obs in enumerate(my_obs[1:-max_lag - 1])])
    deltas = np.diff(windowed_mean)

    return ((deltas - np.mean(deltas)) ** 2).mean() ** 0.25
    #return np.float32((np.diff([x[-1] for x in observations[:-max_lag]]) ** 2).mean()**0.125    )

def empirical_credint(samples, alpha=0.11):

    n_samples = len(samples)
    idx = math.floor(0.5 * alpha * n_samples)
    ss = sorted(samples)

    lower = ss[idx]
    upper = ss[::-1][idx]
    return lower, upper


def floor_median(my_tensor):
    n_elements = len(my_tensor)
    floor_middle = n_elements // 2
    return t.sort(my_tensor).values[floor_middle]


def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
