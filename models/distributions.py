import numpy as np
import scipy.stats
import torch as t
from torch.distributions import Normal, Poisson
import math

from .utils import betaln, binomln, kappa_marginal_moments


# flat gaussian mixture
class GaussianMixture:
    def __init__(self, mus, sigs):
        """
        Args:
            - mus: (t.tensor) vector of component means [shape (n_particles,)]
            - sigs: (t.tensor) vector of component variances [shape (n_particles,)]
        """
        self._kernel = Normal(loc=mus, scale=sigs)
        self._n_particles = len(mus)

    def log_prob(self, lam, keepdims=False):
        """
        Log probability of scalar poisson rate under
        the Gaussian Mixture.

        Args:
            - lam: (scalar/size 0 tensor) Poisson rate
        """
        if type(lam) == float:
            lam = t.tensor([lam])
        elif len(lam.size()) == 0:
            lam = lam.view(1)

        log_p = t.logsumexp(
            self._kernel.log_prob(lam[:, None]), dim=1, keepdims=keepdims
        )
        normalize = math.log(self._n_particles)
        return log_p - normalize

    def sample(self, n_samples):
        return self._kernel.sample(n_samples)


class PoissonMixture:
    def __init__(self, rates):
        """
        Args:
            - rates: (t.tensor) vector of component means [shape (n_particles,)]
        """
        self._kernel = Poisson(rates)
        self._n_particles = len(rates)
        self.support = (0, np.inf)

    def log_prob(self, x, keepdims=False):
        """
        Log probability of scalar count under
        the Poisson Mixture.

        Args:
            - x: (scalar/size 0 tensor) count
        """
        if type(x) == float:
            x = t.tensor([x])
        elif len(x.size()) == 0:
            x = x.view(1)

        log_p = t.logsumexp(self._kernel.log_prob(x), dim=1, keepdims=keepdims)
        normalize = math.log(self._n_particles)
        return log_p - normalize

    def sample(self, n_samples):
        return self._kernel.sample(n_samples)


class LaggedBetaBinomial:
    def __init__(self, ys, alphas, betas, prior):
        """
        beta-binomial emission distribution

        Args:
            - ys: (t.tensor) [shape (n_lag_steps,)]
            - alphas (t.tensor) first beta shape parameters for thinning prior [shape (n_lag_steps,)]
            - alphas (t.tensor) second beta shape parameters for thinning prior [shape (n_lag_steps,)]
            - prior_x (distribution) prior distribution on the true count. Must implement log_prob
        """
        # support of the true count for latent marginalization
        try:
            lower = prior.support[0]
        except:
            lower = prior.support.lower_bound

        self._support_lower = max(lower, int(ys.max()))

        self._ys = ys
        self._alphas = alphas
        self._betas = betas
        self._prior = prior

        self._left = self._ys + self._alphas  # (n_lag_steps,)
        self._right = betas - ys  # (n_lag_steps,)
        self._log_denom = betaln(alphas, betas)  # (n_lag_steps,)

    def log_prob(self, x, support_check=True):
        if support_check:
            if x < self._support_lower:
                return t.tensor([-np.inf])

        if type(x) == float:
            x = t.tensor([x])
        elif len(x.size()) == 0:
            x = x.view(1)

        right = x[None, :] + self._right[:, None]  # (n_lag_steps, x_dim)
        log_num = betaln(self._left[:, None], right)  # (n_lag_steps, x_dim)
        log_binom = binomln(x[None, :], self._ys[:, None])  # (n_lag_steps, x_dim)
        log_prior = self._prior.log_prob(x[:, None])  # [:, None]) # (x_dim, prior_dim)

        log_beta_binom = (
            log_binom[:, :, None]
            + log_num[:, :, None]
            - self._log_denom[:, None, None]
            + log_prior[None, :]
        )  # (n_lag_steps, x_dim, prior_dim)

        return log_beta_binom.sum(dim=0)  # (x_dim, prior_dim)

    def log_marginal(self, support_max=100):

        try:
            upper = prior.support[1]
        except:
            upper = support_max

        alpha_f, beta_f = self._alphas[-1], self._betas[-1]
        mu = alpha_f / (alpha_f + beta_f)  # 0.5, var = 0.5
        sig = np.sqrt(
            alpha_f * beta_f / ((alpha_f + beta_f) ** 2 * (alpha_f + beta_f + 1))
        )

        xs = t.arange(self._support_lower, support_max).float()

        return self.log_prob(xs, support_check=False).logsumexp(dim=0)  # (prior_dim,)


class PoissonBetaBinomial:
    def __init__(self, ys, alphas, betas, prior_x, prior_lam):
        """
        Args:
            - ys: (t.tensor) vector of reported counts [shape (n_lag_steps,)]
            - alphas: (t.tensor) vector of beta prior shape parameters [shape (n_lag_steps,)]
            - betas: (t.tensor) vector of beta prior shape parameters [shape (n_lag_steps,)]
            - prior_x: (distribution) prior distribution on true count [must implement log_prob]
            - prior_lam: (distribution) prior distribution on poisson rate [must implement log_prob]
        """

        # support of the true count for latent marginalization
        support_lower = max(prior_x.support[0], int(ys.max()))
        self._xs = t.arange(support_lower, prior_x.support[1]).float()

        # set prior
        self._prior_lam = prior_lam

        self._log_beta_binom = self._beta_binom(ys, alphas, betas, prior_x)
        self._poisson_log_norm = t.lgamma(self._xs + 1)

        # utils for sampler
        self.is_continuous = True
        self.support = (0, np.inf)

    def log_prob(self, lam, support_check=True):
        """
        log-probability of scalar poisson rate under
        the Poisson beta-binomial emission model.

        Args:
            - lam: (float or 0-dim tensor) poisson rate
        """
        support_lower, support_upper = self.support
        if support_check:
            if not self._support_check(lam):
                return -np.inf

        if type(lam) == float:
            lam = t.tensor([lam])
        elif len(lam.size()) == 0:
            lam = lam.view(1)

        log_poisson = (
            self._xs[:, None] * np.log(lam)[None, :]
            - lam[None, :]
            - self._poisson_log_norm[:, None]
        )  # (x_dim, lam_dim)
        log_series = (log_poisson[None, :] + self._log_beta_binom[:, :, None]).sum(
            axis=0
        )  # (x_dim, lam_dim)
        if self._prior_lam is None:
            log_prior_lam = 0.
        else:
            log_prior_lam = self._prior_lam.log_prob(lam)  # (lam_dim)
        log_prob_lam = t.logsumexp(log_series, dim=0) + log_prior_lam  # (lam_dim)

        return log_prob_lam

    def _beta_binom(self, ys, alphas, betas, prior_x):
        """
        beta-binomial emission distribution

        Args:
            - ys: (t.tensor) [shape (n_lag_steps,)]
            - alphas (t.tensor) first beta shape parameters for thinning prior [shape (n_lag_steps,)]
            - alphas (t.tensor) second beta shape parameters for thinning prior [shape (n_lag_steps,)]
            - prior_x (distribution) prior distribution on the true count. Must implement log_prob
        """
        xs = self._xs

        left = (ys + alphas)[:, None]  # (n_lag_steps, 1)
        right = xs - ys[:, None] + betas[:, None]  # (n_lag_steps, x_dim)
        log_num = betaln(left, right)  # (n_lag_steps, x_dim)
        log_binom = binomln(xs[None, :], ys[:, None])  # (n_lag_steps, x_dim)
        log_denom = betaln(alphas, betas)[:, None]  # (n_lag_steps, 1)
        log_prior_x = prior_x.log_prob(xs)  # (x_dim)

        log_beta_binom = (
            log_binom + log_num - log_denom + log_prior_x
        )  # (n_lag_steps, x_dim)

        return log_beta_binom

    def _support_check(self, lam):
        return self.support[0] <= lam <= self.support[1]


class CountSmoothingDistribution:
    def __init__(self, ys, a, b, lambda_smoothing_particles):

        prior = Poisson(lambda_smoothing_particles)
        self._emission = LaggedBetaBinomial(ys, a, b, prior)
        _n_particles = len(lambda_smoothing_particles)
        self._log_normalizer = math.log(_n_particles)

    def log_prob(self, x):
        weights = self._emission.log_prob(
            x, support_check=False
        ) - self._emission.log_marginal(support_max=x.max())
        lp = t.logsumexp(weights, dim=1) - self._log_normalizer
        return lp


class AdditiveDriftDistribution:
    def __init__(self, kappa_sigma, prior_particles, emission_dist):

        self._prior_lambdas = prior_particles[:, 0].squeeze().numpy()
        self._prior_kappas = prior_particles[:, 1].squeeze().numpy()
        self._kappa_sigma = kappa_sigma
        self.y_likelihood = emission_dist

    def sample(self, length_scale, burn_in=100, thin=100, n_samples=500):

        n_steps = n_samples * thin + burn_in

        kappa_proposal_dist = Normal(0, scale=length_scale)
        deltas = kappa_proposal_dist.sample([n_steps]).squeeze()

        # init
        kap = np.mean(self._prior_kappas)
        lam_idx = np.random.choice(np.arange(len(self._prior_lambdas)))
        lam = (t.tensor([self._prior_lambdas[lam_idx]]) + kap).abs()

        ll = self.y_likelihood.log_prob(lam) + Normal(
            self._prior_kappas[lam_idx], self._kappa_sigma
        ).log_prob(kap)
        samples = []

        for i, delta in enumerate(deltas):
            # sampler_pbar.update()
            kap_p = kap + delta
            lam_idx = np.random.choice(np.arange(len(self._prior_lambdas)))
            lam_p = kap_p + self._prior_lambdas[lam_idx]

            weight = sum(self._prior_lambdas == self._prior_lambdas[lam_idx]).item()

            # component likelihood
            lam_p_ll = self.y_likelihood.log_prob(lam_p)
            kap_p_ll = Normal(self._prior_kappas[lam_idx], self._kappa_sigma).log_prob(
                kap_p
            )
            p_ll = lam_p_ll + kap_p_ll + np.log(weight)

            log_prob_accept = p_ll - ll

            if log_prob_accept > 0:
                accept = True
            else:
                p = t.exp(log_prob_accept).item()
                accept = np.random.choice([True, False], p=[p, 1 - p])

            if accept:
                kap = kap_p
                lam = lam_p
                ll = p_ll

            samples.append(t.tensor([lam, kap]))

        return t.stack(samples[burn_in:][::thin])


class DriftSmoothingDistribution:
    def __init__(
        self,
        lambda_filtering_particles,
        lambda_smoothing_particles,
        prior_kappa_loc,
        prior_kappa_scale,
        random_walk_scale,
    ):

        # required for kappa log probability
        self._filtering = lambda_filtering_particles
        self._smoothing = lambda_smoothing_particles
        self._rw_scale = random_walk_scale
        self._prior_kappa = Normal(loc=prior_kappa_loc, scale=prior_kappa_scale)

        # Marginal normalizer is a gaussian mixture
        mixture_locs, mixture_scales = kappa_marginal_moments(
            prior_kappa_loc,
            prior_kappa_scale,
            random_walk_scale,
            lambda_filtering_particles,
        )
        normalizer = GaussianMixture(mixture_locs, mixture_scales)
        self._row_norm = normalizer.log_prob(lambda_smoothing_particles, keepdims=True)

    def log_prob(self, kappa):

        # prior probability of kappa
        log_prior = self._prior_kappa.log_prob(kappa)

        # likelihood function for kappa marginalized over the filtering and smoothing distributions
        transition_log_proba = self.particle_transition_matrix(kappa)
        marginal_log_likelihood = t.logsumexp(
            transition_log_proba - self._row_norm, dim=(0, 1)
        )

        # smoothing probability for kappa
        particle_norm = math.log(self._smoothing.shape[0]) + math.log(
            self._filtering.shape[0]
        )
        lp = log_prior + marginal_log_likelihood - particle_norm

        return lp

    def particle_transition_matrix(self, kappa):
        tm_loc = kappa * self._filtering
        tm_scale = self._rw_scale
        transition_dist = Normal(loc=tm_loc, scale=tm_scale)
        transition_log_prob_matrix = transition_dist.log_prob(self._smoothing[:, None])
        return transition_log_prob_matrix


# This lets me sample lam
class CorrectedPoissonBetaBinomial:
    def __init__(
        self, ys, alphas, betas, prior_x, prior_lam, prior_correction, multidim=False
    ):
        self._pbb = PoissonBetaBinomial(ys, alphas, betas, prior_x, prior_lam=None)
        self._prior_lam = prior_lam
        self._prior_correction = prior_correction
        self._multidim = multidim

    def log_prob(self, lam, support_check=True):

        if not self._multidim:
            if support_check:
                if lam < 0:
                    return -np.inf

            # LAM MUST BE SCALAR HERE
            effective_lam = lam * self._prior_correction.values

            if self._prior_lam is None:
                prior_lam_term = 0.
            else:
                prior_lam_term = self._prior_lam.log_prob(lam)  # (lam_dim)

            lp = t.logsumexp(
                self._pbb.log_prob(effective_lam, support_check=False)
                + self._prior_correction.log_probas,
                axis=0,
            )
            lp = lp + prior_lam_term

        else:
            effective_lam = (
                lam[:, None] * self._prior_correction.values[None, :]
            )  # (lam_dim, z_dim)
            if self._prior_lam is None:
                prior_lam_term = 0.
            else:
                prior_lam_term = self._prior_lam.log_prob(lam)  # (lam_dim)

            pbb_proba = self._pbb.log_prob(effective_lam.view(-1), support_check=False)
            lp = t.logsumexp(
                pbb_proba.view(effective_lam.shape) + self._prior_correction.log_probas,
                axis=1,
            )
            lp = lp + prior_lam_term

        return lp


class DiscreteDistribution:
    def __init__(self, values, log_probas):
        self.values = values
        self.log_probas = log_probas


class EmpiricalDistribution:
    def __init__(self, support, probas):
        """
        Args:
        - support: (tuple) edges of support. Support assumed to exist on all integers between. [shape (2,)]
        - probas: (t.tensor) probabilities for each element of support. [shape (support[1] - support[0],)]
        """
        self.support = support
        self.probas = probas

        self._xs = t.arange(support[0], support[1]).float()

    def log_prob(self, x):
        return self.probas[x.int() - self.support[0]]

    def sample(self, size):
        idxs = np.arange(0, len(self._xs))
        sample_idxs = np.random.choice(idxs, p=self.probas, size=size)
        samples = self._xs[sample_idxs]

        return samples
