import multiprocess as mp

import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch as t

import os
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

SEED = 7451

from models.inference import Nowcaster
from data_processing.data import get_data, moment_match_theta_priors

thin = 20
burn_in = 750
n_samples = 250

prior_kappa_scale = 0.01

# adj_dict = pickle.load(open('../../ltla-covid-cases/data/pickles/ltla_adjacency_dict.pickle', 'rb'))
# all_ltlas = list(adj_dict.keys()) + ["Hackney and City of London"]
# all_ltlas.remove("Hackney")
# all_ltlas.remove("City of London")

df = pd.read_csv(f"../data/cases-2020-12-14.csv")
all_ltlas = df.query("`Area type` == 'ltla'")['Area name'].unique()

epsilon = 0.1


def fit(sigma):
    # for ltla in all_ltlas:
    for ltla in ['Oldham']:

        random.seed(SEED)
        np.random.seed(SEED)
        t.manual_seed(SEED)

        out_dir = f'../../ltla-covid-cases/data/pickles/drift_model_refactored_scale_{sigma}/2020-12-13'
        os.makedirs(out_dir, exist_ok=True)

        save_name = ltla.replace(" ", "_")
        rd = pd.date_range(start="2020-11-19", end="2020-12-13")
        sd = pd.date_range(start="2020-11-18", end="2020-12-12")

        if not os.path.exists(f"{out_dir}/{save_name}.pickle"):
            print(ltla)
            df = get_data(
                ltla,
                area_type="ltla",
                report_dates=rd,
                specimen_dates=sd,
            )

            alphas, betas = moment_match_theta_priors(df[df['Specimen date']>='2020-11-25'], n_lags=18)

            last_report = df[df['Report date']=='2020-12-13']
            x_0 = last_report['Daily lab-confirmed cases'].values[-25:-18].mean()

            prior_lam0_shape = 0.3 * x_0 ** 2. + (epsilon if x_0==0 else 0)
            prior_lam0_rate = 0.3 * x_0 + (epsilon if x_0==0 else 0)

            ys = last_report["Daily lab-confirmed cases"].values[-18:]

            model = Nowcaster(
                prior_lam0_shape=prior_lam0_shape,
                prior_lam0_rate=prior_lam0_rate,
                random_walk_scale=sigma,
                prior_theta_shape_a = alphas,
                prior_theta_shape_b = betas,
                sampler_length_scale=None,
            )

            model.marginal_filtering_drift(ys, sd[-18:], thin, burn_in, n_samples)
            model.marginal_smoothing_drift()
            model.x_smoothing()
            model.compute_evidence_drift()

            save_name = ltla.replace(" ", "_")
            pickle.dump(model, open(f'{out_dir}/{save_name}.pickle', 'wb'))


# def main():
#     with mp.Pool(processes=4) as pool:
#         r = pool.map(fit, np.arange(1,9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sigma", help="random walk scale")
    args = parser.parse_args()
    sigma = int(args.sigma)
    fit(sigma)
