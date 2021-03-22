import math
import itertools
import torch as t
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .distributions import (
    PoissonBetaBinomial,
    CorrectedPoissonBetaBinomial,
    DiscreteDistribution,
    GaussianMixture,
    DriftSmoothingDistribution,
    PoissonMixture,
    LaggedBetaBinomial,
    CountSmoothingDistribution,
    EmpiricalDistribution,
    AdditiveDriftDistribution,
)
from torch.distributions import Normal, Gamma, Bernoulli, Poisson

from scipy.stats import betabinom
from scipy.special import logsumexp

from .utils import (
    _isnotebook,
    kappa_marginal_moments,
    floor_median,
    empirical_credint,
    random_walk_scale_estimator,
)
from .plotting import plot_posterior
import tqdm

tq = tqdm.tqdm_notebook if _isnotebook() else tqdm.tqdm

sampler_pbar = None
EVIDENCE_N_SAMPLES = 50

## Use self.max_x_support() to set max support from data
# MAXIMUM_X_SUPPORT = 2000
MINIMUM_X_SUPPORT = 0.0


class Nowcaster:
    """
    Parameters
    ----------
    prior_theta_shape_a: List, each value is the a shape parameter of the prior
        on the under reporting proportion (Beta distribution).
        Lag 1 is parameter is last i.e. (prior_theta_shape_a[-1]).
        List must have same length as there are observations.
    prior_theta_shape_b: List, each value is the b shape parameter of the prior
        on the under reporting proportion (Beta distribution).
        Lag 1 parameter is last i.e. (prior_theta_shape_b[-1]).
        List must have same length as there are observations.
    random_walk_scale: Float, the scale of the AR1 prior
    prior_lam0_shape: optional Float, the shape parameter of the prior on the
        initial lambda (Gamma distribution)
    prior_lam0_rate: optional Float, the rate parameter of the prior on the
        initial lambda (Gamma distribution)
    sampler_lenght_scale: optional Float, controls step size of the MCMC sampler.
        If None, default is set (based on the random_walk_scale value).
    prior_weekend_shape_a: optional Float, the a shape parameter of the weekend
        effect prior (Beta distribution)
    prior_weekend_shape_b: optional Float, the b shape parameter of the weekend
        effect prior (Beta distribution)
    """

    def __init__(
        self,
        prior_theta_shape_a,
        prior_theta_shape_b,
        random_walk_scale,
        prior_lam0_shape=None,
        prior_lam0_rate=None,
        sampler_length_scale=None,
        prior_weekend_shape_a=6,
        prior_weekend_shape_b=2,
    ):
        # ===============================================
        # Hyperparameters for priors
        # ===============================================
        self._prior_theta_shape_a = t.FloatTensor(prior_theta_shape_a)
        self._prior_theta_shape_b = t.FloatTensor(prior_theta_shape_b)
        self._prior_lam0_shape = prior_lam0_shape
        self._prior_lam0_rate = prior_lam0_rate
        self._random_walk_scale = random_walk_scale
        self._prior_weekend_shape_a = prior_weekend_shape_a
        self._prior_weekend_shape_b = prior_weekend_shape_b

        # ===============================================
        # Params for marginal filtering MCMC
        # ===============================================
        self._sampler_length_scale = sampler_length_scale

        # ===============================================
        # Results
        # ===============================================
        self._lambda_filtering_samples = None
        self._lambda_smoothing_samples = None
        self._x_smoothing_samples = None
        self._drift_samples = None
        self._log_evidence = None

    @property
    def lambda_filtering_samples(self):
        return self._lambda_filtering_samples

    @property
    def lambda_smoothing_samples(self):
        return self._lambda_smoothing_samples

    @property
    def x_smoothing_samples(self):
        return self._x_smoothing_samples

    @property
    def kappa_smoothing_samples(self):
        return self._drift_smoothing_samples

    @property
    def log_evidence(self):
        return self._log_evidence

    def max_x_support(self):
        return int(max(self._observations)) * 3

    def set_prior_x(self):
        """
        Flat prior on Xs in range [0,self.max_x_support()].
        """
        x_support = (int(MINIMUM_X_SUPPORT), self.max_x_support())
        len_support = x_support[1] - x_support[0]
        x_prior_probas = np.ones(len_support) / len_support
        self._prior_x = EmpiricalDistribution(support=x_support, probas=x_prior_probas)

    def lam_0_init(self):
        """
        If no parameters are given for prior on the initial lambda, choose prior
        to have a mean equal to the first observation.
        """
        epsilon = 0.1
        if self._prior_lam0_shape is None or self._prior_lam0_rate is None:
            x_0 = self._observations[0]
            self._prior_lam0_shape = 0.3 * x_0 ** 2. + (epsilon if x_0 == 0 else 0)
            self._prior_lam0_rate = 0.3 * x_0 + (epsilon if x_0 == 0 else 0)

    def get_theta_priors(self, i):
        """
        Get shape parameters for prior on the under reportiong proportion theta
        at i-th step. Return as 1D tensors.
        """
        return (
            self._prior_theta_shape_a[i : i + 1],
            self._prior_theta_shape_b[i : i + 1],
        )

    def check_dates(self, dates):
        """
        Ensures `self._dates` is None or a Pandas DatetimeIndex object.
        """
        if dates is None or isinstance(dates, pd.DatetimeIndex):
            self._dates = dates
        else:
            self._dates = pd.DatetimeIndex(dates)

    def check_weekend(self, i):
        """
        Return True if i-th date is a weekend.
        """
        if self._dates is None:
            return False
        else:
            days = self._dates.day_name()
            day = days[i]
            if day in ["Saturday", "Sunday"]:
                return True
            else:
                return False

    def set_weekend_correction_prior(self):
        """
        Prior for weekend correction (what proportion of usual/weekday count expect to observe on weekends. Have the same prior for each weekend day.
        """
        self._grid_weekend = t.arange(0.01, 1, 0.01)
        self._weekend_prior_correction = DiscreteDistribution(
            self._grid_weekend,
            t.distributions.Beta(
                self._prior_weekend_shape_a, self._prior_weekend_shape_b
            ).log_prob(self._grid_weekend),
        )  # t.log(t.ones_like(grid) / len(grid)))
        self._weekend_corrections = []

    def get_weekend_correction_posterior(self, predictive_samples, y, a, b):
        """
        Save posterior for weekend effect (appends a distribution to list) at this step.

        Example use (plotting):
            plt.plot(
                Nowcaster._weekend_corrections[0].values, Nowcaster._weekend_corrections[0].log_probas.exp()
            )

        Parameters
        ----------
        predictive_samples: t.FloatTensor, the lambda predictive samples
        y: Integer, the reported count
        a: Float: the a shape parameter for the prior on the under reporting proportion
        b: Float: the b shape parameter for the prior on the under reporting proportion
        """
        lps = []

        for s in self._grid_weekend:
            xprior = PoissonMixture(predictive_samples * s)
            lp = LaggedBetaBinomial(y, a, b, xprior).log_marginal(
                support_max=self.max_x_support()
            )
            lps.append(t.logsumexp(lp, axis=0))

        lps = t.tensor(lps)
        lps = lps + self._weekend_prior_correction.log_probas
        lps = lps - t.logsumexp(lps, axis=0)

        posterior_correction = DiscreteDistribution(self._grid_weekend, lps)
        self._weekend_corrections.append(posterior_correction)

    def marginal_filtering_drift(
        self, observations, dates=None, thin=100, burn_in=100, n_samples=500
    ):
        """
        Lambda and kappa filtering distributions.

        Parameters
        ----------
        observations: List of integers, the observed counts (most recent are last)
        dates: List of dates (can be strings), same length and order as observations
        thin: Integer, MCMC parameter (marginal filtering for lambda and kappa)
        burn_in: Integer, MCMC parameter (marginal filtering for lambda and kappa)
        n_samples: Integer, the number of MCMC samples (marginal filtering for lambda and kappa)
        """
        self._observations = t.from_numpy(observations).float()
        if self._sampler_length_scale is None:
            self._sampler_length_scale = self._random_walk_scale / 2

        self.set_prior_x()
        self.set_weekend_correction_prior()
        self.check_dates(dates)

        self.lam_0_init()
        prior_lam = Gamma(self._prior_lam0_shape, self._prior_lam0_rate)
        init_lam = prior_lam.sample()
        particle_container = []

        outer_pbar = tq(enumerate(self._observations), total=len(self._observations))
        outer_pbar.set_description(f"Running marginal MCMC")

        self._lambda_filtering_samples = []
        self._drift_filtering_samples = []

        prior_lam_particles = Gamma(
            self._prior_lam0_shape, self._prior_lam0_rate
        ).sample([100])
        prior_kap_particles = Normal(0, self._random_walk_scale).sample([100])
        prior_particles = t.stack([prior_lam_particles, prior_kap_particles]).T

        for i, ys in outer_pbar:

            outer_pbar.set_description(f"Particle MCMC: Time-step {i}")

            a, b = self.get_theta_priors(i)
            y = ys[None]

            weekend = self.check_weekend(i)
            if weekend:
                emission_dist = CorrectedPoissonBetaBinomial(
                    y, a, b, self._prior_x, None, self._weekend_prior_correction
                )
            else:
                emission_dist = PoissonBetaBinomial(y, a, b, self._prior_x, None)

            dist = AdditiveDriftDistribution(
                self._random_walk_scale, prior_particles, emission_dist
            )
            samples = dist.sample(
                length_scale=self._sampler_length_scale,
                thin=thin,
                burn_in=burn_in,
                n_samples=n_samples,
            )

            prior_particles = samples
            self._lambda_filtering_samples.append(samples[:, 0])
            self._drift_filtering_samples.append(samples[:, 1])

            if i == 0:
                predictive_samples = prior_lam.sample([10]).abs().flatten()
            else:
                kap_samples = Normal(kap_filt, self._random_walk_scale).sample([1])
                predictive_samples = (kap_samples + lam_filt).abs()

            if weekend:
                self.get_weekend_correction_posterior(predictive_samples, y, a, b)

            lam_filt = self._lambda_filtering_samples[i]
            kap_filt = self._drift_filtering_samples[i]

        # reset the progress bar
        global sampler_pbar
        sampler_pbar = None

    def marginal_filtering(
        self, observations, dates, thin=100, burn_in=100, n_samples=500
    ):
        """
        Lambda filtering distribution for original model without additive drift.

        Parameters
        ----------
        observations: List of integers, the observed counts (most recent are last)
        dates: List of dates (can be strings), same length and order as observations
        thin: Integer, MCMC parameter (marginal filtering for lambda)
        burn_in: Integer, MCMC parameter (marginal filtering for lambda)
        n_samples: Integer, the number of MCMC samples (marginal filtering for lambda)
        """
        self._observations = t.from_numpy(observations).float()
        if self._random_walk_scale is None:

            rws = random_walk_scale_estimator(self._observations)

            if max(self._observations) < 30:
                self._random_walk_scale = max(0.05, min(0.5, rws))
            else:
                self._random_walk_scale = rws

        self._sampler_length_scale = max(1., self._random_walk_scale)

        self.set_prior_x()
        self.set_weekend_correction_prior()
        self.check_dates(dates)

        self.lam_0_init()
        prior_lam = Gamma(self._prior_lam0_shape, self._prior_lam0_rate)
        init_lam = prior_lam.sample()
        particle_container = []

        outer_pbar = tq(enumerate(self._observations), total=len(self._observations))
        outer_pbar.set_description(f"Running marginal MCMC")

        self._lambda_filtering_samples = []

        for i, ys in outer_pbar:

            outer_pbar.set_description(f"Particle MCMC: Time-step {i}")

            a, b = self.get_theta_priors(i)
            y = ys[None]

            weekend = self.check_weekend(i)
            if weekend:
                emission_dist = CorrectedPoissonBetaBinomial(
                    y, a, b, self._prior_x, prior_lam, self._weekend_prior_correction
                )
            else:
                emission_dist = PoissonBetaBinomial(y, a, b, self._prior_x, prior_lam)

            # Filtering for underlying rate
            sampler = MetropolisSampler(
                emission_dist, length_scale=self._sampler_length_scale
            )
            particles = sampler.run_chain(
                x_init=init_lam, thin=thin, burn_in=burn_in, n_samples=n_samples
            )
            particle_container.append(particles)
            # sampler._pbar.reset()
            self._lambda_filtering_samples.append(particles)

            predictive_samples = prior_lam.sample([10]).abs().flatten()
            prior_lam = GaussianMixture(particles, self._random_walk_scale)

            # prior correction part
            if weekend:
                self.get_weekend_correction_posterior(predictive_samples, y, a, b)

            # set sampler starting point to present mean
            init_lam = particles.mean()

        # reset the progress bar
        global sampler_pbar
        sampler_pbar = None

        self._lambda_filtering_samples = particle_container

    def marginal_smoothing(self):
        """
        Lambda smoothing distribution (for original model without additive drift).
        """
        assert (
            self.lambda_filtering_samples is not None
        ), "Run marginal_filtering before calling the marginal_smoothing method"

        filtering_particles = self.lambda_filtering_samples
        init_smoothing = filtering_particles[-1]
        smoothing_particles = init_smoothing
        particle_container = [init_smoothing]

        n_particles = len(init_smoothing)
        idx_range = np.arange(0, n_particles)

        for filtering_particles in filtering_particles[::-1][1:]:
            particle_transition_matrix = Normal(
                filtering_particles, self._random_walk_scale
            ).log_prob(smoothing_particles[:, None])
            row_norm = t.logsumexp(particle_transition_matrix, axis=1, keepdims=True)
            resampling_probas = (
                t.logsumexp(particle_transition_matrix - row_norm, dim=0)
                - math.log(n_particles)
            ).exp()
            idxs = np.random.choice(
                idx_range, p=resampling_probas.numpy(), size=n_particles
            )
            smoothing_particles = filtering_particles[idxs]
            particle_container.append(smoothing_particles)

        self._lambda_smoothing_samples = particle_container[::-1]

    def marginal_smoothing_drift(self):
        """
        Lambda and kappa smoothing distributions.
        """
        lambda_filtering_particles = self._lambda_filtering_samples
        kappa_filtering_particles = self._drift_filtering_samples

        init_smoothing_lam = lambda_filtering_particles[-1]
        init_smoothing_kap = kappa_filtering_particles[-1]

        n_particles = len(init_smoothing_lam)
        idx_range = np.arange(0, n_particles)

        lam_smoothing_container = [init_smoothing_lam]
        kap_smoothing_container = [init_smoothing_kap]

        for i, (lam, kap) in enumerate(
            zip(
                lambda_filtering_particles[::-1][1:],
                kappa_filtering_particles[::-1][1:],
            )
        ):

            lam_prev = lambda_filtering_particles[::-1][i]
            kap_prev = kappa_filtering_particles[::-1][i]
            transitions_mask = lam_prev == kap_prev[None, :] + lam[:, None]

            log_weights = []

            for j in idx_range:
                mu = kap_prev[transitions_mask[:, j].nonzero().squeeze()]
                resample_weight = (
                    Normal(mu, self._random_walk_scale)
                    .log_prob(kap[j])
                    .logsumexp(dim=0)
                    .squeeze()
                )
                log_weights.append(resample_weight)

            log_weights = t.tensor(log_weights)
            log_probas = log_weights - log_weights.logsumexp(dim=0)
            resampling_probas = log_probas.exp().numpy()

            smoothed_idxs = np.random.choice(
                idx_range, p=resampling_probas, size=n_particles
            )

            smoothed_lams = lam[smoothed_idxs]
            smoothed_kaps = kap[smoothed_idxs]

            lam_smoothing_container.append(smoothed_lams)
            kap_smoothing_container.append(smoothed_kaps)

        self._lambda_smoothing_samples = lam_smoothing_container[::-1]
        self._drift_smoothing_samples = kap_smoothing_container[::-1]

    def x_smoothing(self):
        """
        X smoothing distribution (the now-cast).
        """
        assert (
            self.lambda_smoothing_samples is not None
        ), "Run marginal_smoothing before calling the x_smoothing method"
        smoothing_xs = []

        for i, vals in enumerate(
            zip(self._observations, self.lambda_smoothing_samples)
        ):

            ys, ss = vals

            grid_xs = t.from_numpy(np.arange(MINIMUM_X_SUPPORT, self.max_x_support()))

            a, b = self.get_theta_priors(i)
            y = ys[None]

            all_probas = []

            log_bb = betabinom.logpmf(y, grid_xs.numpy(), a, b)

            weekend = self.check_weekend(i)
            if weekend:
                pm = Poisson(ss[:, None] * self._grid_weekend[None, :])
                log_poisson = pm.log_prob(grid_xs[:, None, None])

                log_joint = (
                    t.from_numpy(log_bb)[:, None, None]
                    + self._weekend_prior_correction.log_probas[None, None, :]
                    + log_poisson
                )
                normalizer = log_joint.logsumexp(dim=[0, 2])
                log_posterior = (
                    (log_joint - normalizer[:, None]).exp().sum(dim=-1).mean(dim=-1)
                )
            else:
                pm = Poisson(ss)
                log_poisson = pm.log_prob(grid_xs[:, None])

                log_joint = t.from_numpy(log_bb)[:, None] + log_poisson
                normalizer = log_joint.logsumexp(dim=0)
                log_posterior = (log_joint - normalizer).exp().mean(dim=1)

            samples = np.random.choice(grid_xs, p=log_posterior, size=1000)
            smoothing_xs.append(samples)

            # self._all_probas = all_probas
            self._x_smoothing_samples = smoothing_xs

        self._x_smoothing_samples = smoothing_xs

    def compute_evidence(self):
        """
        Log evidence for original model without additive drift.
        """
        assert (
            self.lambda_filtering_samples is not None
        ), "Run marginal_filtering before calling the compute_evidence method"

        evolved_evidence_container = []
        evidence_container = []
        prior_lam = Gamma(self._prior_lam0_shape, self._prior_lam0_rate)

        for i, ys in enumerate(self._observations):

            a, b = self.get_theta_priors(i)
            y = ys[None]

            weekend = self.check_weekend(i)
            if weekend:
                emission_dist = CorrectedPoissonBetaBinomial(
                    y,
                    a,
                    b,
                    self._prior_x,
                    None,
                    self._weekend_prior_correction,
                    multidim=True,
                )
            else:
                emission_dist = PoissonBetaBinomial(y, a, b, self._prior_x, None)

            # note abs for folded normal approximation
            if i == 0:
                evidence_n_samples = 100
            else:
                evidence_n_samples = 2

            evidence_samples = (
                prior_lam.sample([evidence_n_samples]).abs().flatten()
            )  # (EVIDENCE_N_SAMPLES, n_particles)
            # evidence_samples = t.arange(1,1000)

            norm = evidence_samples.shape[0]

            conditional_ev = emission_dist.log_prob(
                evidence_samples, support_check=False
            ).logsumexp(axis=0) - np.log(norm)
            evidence_container.append(conditional_ev.item())

            prior_lam = GaussianMixture(
                self._lambda_filtering_samples[i], self._random_walk_scale
            )

        self._log_evidence = evidence_container

    def compute_evidence_drift(self):
        """
        Log evidence for model.
        """
        assert (
            self.lambda_filtering_samples is not None
        ), "Run marginal_filtering before calling the compute_evidence method"

        evidence_container = []
        prior_lam = Gamma(self._prior_lam0_shape, self._prior_lam0_rate)

        n_particles = len(self._lambda_filtering_samples[0])
        particle_idxs = np.arange(0, n_particles)

        for i, ys in enumerate(self._observations):

            a, b = self.get_theta_priors(i)
            y = ys[None]

            weekend = self.check_weekend(i)
            if weekend:
                emission_dist = CorrectedPoissonBetaBinomial(
                    y,
                    a,
                    b,
                    self._prior_x,
                    None,
                    self._weekend_prior_correction,
                    multidim=True,
                )
            else:
                emission_dist = PoissonBetaBinomial(y, a, b, self._prior_x, None)

            evidence_n_samples = 5

            if i == 0:
                lam_samples = prior_lam.sample([evidence_n_samples]).abs().flatten()
            else:
                kap_samples = Normal(kap_filt, self._random_walk_scale).sample(
                    [evidence_n_samples]
                )
                lam_samples = (kap_samples + lam_filt).abs()

            conditional_ev = emission_dist.log_prob(
                lam_samples.flatten(), support_check=False
            ).logsumexp(axis=0) - np.log(lam_samples.shape[0])

            evidence_container.append(conditional_ev.item())

            lam_filt = self._lambda_filtering_samples[i]
            kap_filt = self._drift_filtering_samples[i]

        self._log_evidence = evidence_container

    def fit(
        self,
        observations,
        dates=None,
        drift=True,
        evidence=True,
        xs=True,
        thin=20,
        burn_in=750,
        n_samples=250,
    ):
        """
        Fit model to data:
            - compute filtering and smoothing for lambda and kappa
            - get smoothing distributions for Xs (the now-cast)
            - get model evidence

        Note: original version of the model did not include an additive drift term.
        Setting `drift=False` runs this original version of the model (in which case the random walk is directly on the lambda).

        Parameters
        ----------
        observations: List of integers, the observed counts (most recent are last)
        dates: List of dates (can be strings), same length and order as observations
        drift: Boolean, whether to include additive drift term in model (default)
        evidence: Boolean, indicates whether to compute model evidence
        xs: Boolean, indicates whether to compute the x smoothing distribution
        thin: Integer, MCMC parameter (marginal filtering for lambda and kappa)
        burn_in: Integer, MCMC parameter (marginal filtering for lambda and kappa)
        n_samples: Integer, the number of MCMC samples (marginal filtering for lambda and kappa)
        """

        if drift is False:
            self.marginal_filtering(observations, dates, thin, burn_in, n_samples)
            self.marginal_smoothing()
            if evidence:
                self.compute_evidence()
        else:
            self.marginal_filtering_drift(observations, dates, thin, burn_in, n_samples)
            self.marginal_smoothing_drift()
            if evidence:
                self.compute_evidence_drift()
        if xs:
            self.x_smoothing()

    def plot_lambda_filtering(self, alpha=0.11):
        # title_str = (
        #    r"Marginal filtering $p(\lambda_{t} \mid \mathbf{y}_{0:t})$ for $\lambda$"
        # )
        lambda_filtering_means = np.array(
            [s.mean() for s in self.lambda_filtering_samples]
        )
        lambda_filtering_cis = np.array(
            [empirical_credint(s, alpha) for s in self.lambda_filtering_samples]
        )
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_posterior(
            ax=ax,
            obs=self._observations,
            posterior_means=lambda_filtering_means,
            posterior_confidence_intervals=lambda_filtering_cis,
            # title=title_str,
            legend=True,
            posterior_label="lambda Filtering",
            confidence_interval_label=f"{1-alpha} Credible Interval",
            obs_label="Reported Count",
            tex_plot=False,
        )
        if self._dates is not None:
            plt.xticks(
                np.arange(0, len(self._observations)),
                labels=self._dates.strftime("%Y-%m-%d"),
            )

    def plot_lambda_smoothing(self, alpha=0.11):
        # title_str = (
        #     r"Marginal smoothing $p(\lambda_{t} \mid \mathbf{y}_{0:T})$ for $\lambda$"
        # )
        lambda_smoothing_means = np.array(
            [s.mean() for s in self.lambda_smoothing_samples]
        )
        lambda_smoothing_cis = np.array(
            [empirical_credint(s, alpha) for s in self.lambda_smoothing_samples]
        )
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_posterior(
            ax=ax,
            obs=self._observations,
            posterior_means=lambda_smoothing_means,
            posterior_confidence_intervals=lambda_smoothing_cis,
            # title=title_str,
            legend=True,
            posterior_label=r"$\lambda$ Smoothing",
            confidence_interval_label=f"{1-alpha} Credible Interval",
            obs_label="Reported Count",
        )
        if self._dates is not None:
            plt.xticks(
                np.arange(0, len(self._observations)),
                labels=self._dates.strftime("%Y-%m-%d"),
            )

    def plot_x_smoothing(self, alpha=0.11):
        # title_str = r"Marginal smoothing $p(x_{t} \mid \mathbf{y}_{0:T})$ for $x$"
        x_smoothing_means = np.array([s.mean() for s in self.x_smoothing_samples])
        x_smoothing_cis = np.array(
            [empirical_credint(s, alpha) for s in self.x_smoothing_samples]
        )
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_posterior(
            ax=ax,
            obs=self._observations,
            posterior_means=x_smoothing_means,
            posterior_confidence_intervals=x_smoothing_cis,
            # title=title_str,
            legend=True,
            posterior_label=r"$x$ Smoothing",
            confidence_interval_label=f"{1-alpha} Credible Interval",
            obs_label="Reported Count",
        )
        if self._dates is not None:
            plt.xticks(
                np.arange(0, len(self._observations)),
                labels=self._dates.strftime("%Y-%m-%d"),
            )

    def plot_drift_smoothing(self, alpha=0.11):
        kappa_smoothing_means = np.array(
            [s.mean() for s in self.kappa_smoothing_samples]
        )
        kappa_smoothing_cis = np.array(
            [empirical_credint(s, alpha) for s in self.kappa_smoothing_samples]
        )
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_posterior(
            ax=ax,
            posterior_means=kappa_smoothing_means,
            posterior_confidence_intervals=kappa_smoothing_cis,
            legend=True,
            posterior_color="xkcd:denim blue",
            confidence_interval_color="xkcd:denim blue",
            posterior_label=r"$\kappa$ Smoothing",
            confidence_interval_label=f"{1-alpha} Credible Interval",
            alpha=0.11,
        )
        ax.plot([0, len(self._observations)], [0, 0], "r--")
        if self._dates is not None:
            plt.xticks(
                np.arange(0, len(self._observations)),
                labels=self._dates.strftime("%Y-%m-%d"),
            )


class MixedMetropolisProposal:
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def rsample(self, size):
        n_steps = size[0]
        dim = size[1]

        # TODO: Bernoulli has no rsample
        proposal_0 = Normal(loc=0, scale=self.length_scale[0]).rsample([n_steps])
        proposal_1 = (Bernoulli(probs=0.5).sample([n_steps]) - 0.5) * 2

        proposals = t.stack([proposal_0, proposal_1], axis=1)

        return proposals


class DiscreteProposal:
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def rsample(self, size):
        step_sizes = np.random.choice(t.arange(1, self.length_scale + 1), size=size)
        return (Bernoulli(probs=0.5).sample(size) - 0.5) * 2 * step_sizes


class MetropolisSampler:
    def __init__(self, dist, length_scale, mode="gaussian"):
        self.dist = dist
        self._L = length_scale

        if type(length_scale) in [float, int, np.float32, np.float64]:
            self._dim = 1
        else:
            self._dim = len(length_scale)

        if mode == "gaussian":
            self.proposal_dist = Normal(loc=0, scale=length_scale)
        elif mode == "discrete":
            self.proposal_dist = DiscreteProposal(length_scale=length_scale)
        elif mode == "mixed":
            self.proposal_dist = MixedMetropolisProposal(length_scale=length_scale)
        else:
            raise ValueError("mode must be one of 'gaussian', 'discrete', 'mixed'")

    def run_chain(self, x_init, n_samples=500, burn_in=100, thin=100):
        n_steps = n_samples * thin + burn_in
        samples = []

        deltas = self.proposal_dist.rsample([n_steps, self._dim]).squeeze()
        x = x_init

        global sampler_pbar
        if sampler_pbar is None:
            sampler_pbar = tq(total=n_steps)
        else:
            sampler_pbar.reset(n_steps)
            sampler_pbar.refresh()

        sampler_pbar.set_description(f"Chain progress")

        for i, delta in enumerate(deltas):
            sampler_pbar.update()
            x_p = x + delta
            accept = self._accept(x, x_p)
            if accept:
                x = x_p

            samples.append(x.float())

        return t.stack(samples[burn_in:][::thin])

    def _init_x(self):
        pass

    def _accept(self, x, x_p):

        # log likelihood ratio for current position vs. proposal
        ll = self.dist.log_prob(x)
        ll_p = self.dist.log_prob(x_p)
        log_prob_accept = ll_p - ll

        # accept/reject step
        if log_prob_accept > 0:
            accept = True
        else:
            p = t.exp(log_prob_accept).item()

            accept = np.random.choice([True, False], p=[p, 1 - p])

        return accept
