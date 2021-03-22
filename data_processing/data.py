import pickle
import math

import torch as t
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, date
from os.path import dirname, abspath, isfile

from scipy.stats import norm
from scipy.special import logsumexp

import sys
sys.path.append('..')

# column names
an, ac, at = "Area name", "Area code", "Area type"
sd, dlcc, clcc = (
    "Specimen date",
    "Daily lab-confirmed cases",
    "Cumulative lab-confirmed cases",
)
rd = "Report date"

# field names
ltla, utla, region, nation = (
    "Lower tier local authority",
    "Upper tier local authority",
    "Region",
    "Nation",
)

# dates for the reports we load
# dates = pd.date_range(start="2020-07-02", end="2020-07-30")

# get root directory and data directory paths
root_dir = dirname(dirname(abspath(__file__)))
data_path = f"{root_dir}/data"

def get_data(area_name, area_type="ltla", path=data_path, report_dates=None,  specimen_dates=None):
    """
    Returns data for `area_name` and `area_type` as recorded in reports published on
    `report_dates` for test dates specified in `specimen_dates`
    # (we expect to find data in timestamped datafiles in `<path>/<file_prefix><timestamp>.csv`).

    NOTES on data processing steps:
        - Not all SDs are reported in each report file
            --> carry values forward from last report
        - Sometimes, LTLA never gives a report for a given SD
            --> assume 0 count here
        - Create additional columns (`lag` and `prop_reported`)
            --> the proportion reported is capped at 1.0 (adjust for over-reports)

    :param area_name: str, the area name
    :param area_type: str, the area type (ltla, utla, nation)
    :param path: str, path to directory with data files
    :param report_dates: pandas.DatetimeIndex, the dates for which to load reports
    :param sd_dates: pandas.DatetimeIndex, the test dates for which to get reports

    :returns: Pandas dataframe
    """
    if specimen_dates is None:
        # look at SDs starting 5 days before the first file that load
        # don't include the last report date in sd dates
        first_sd = date.strftime(report_dates[0] - timedelta(5), "%Y-%m-%d")
        dts = [first_sd] + report_dates.strftime("%Y-%m-%d").to_list()[:-1]
    else:
        dts = specimen_dates.strftime("%Y-%m-%d").to_list()

    # ================================================
    # Loop through report files and get dataframe for
    # each report day (for area name, type and SDs)
    # ================================================
    dataframes = []
    for dt in report_dates.strftime("%Y-%m-%d").to_list():
        file = f'{path}/cases-{dt}.csv'
        if isfile(file):
            df_day = pd.read_csv(file)
            df_day[rd] = dt
            dataframes.append(
                df_day[
                    (df_day[at] == area_type)
                    & (df_day[sd].isin(dts))
                    & (df_day[an] == area_name)
                ]
            )
    # combined data for all report dates
    df = pd.concat(dataframes)

    # =================================================
    # If don't have any data for an SD
    # --> assume report of 0 & create/append a row for it
    # =================================================
    row = df.tail(1)
    row[clcc] = np.nan
    for dt in dts:
        if dt not in df[sd].unique():
            # fill in values that care about
            row[sd] = dt
            row[dlcc] = 0
            row[rd] = date.strftime(datetime.strptime(dt, "%Y-%m-%d") + timedelta(1), "%Y-%m-%d")
            df = df.append(row)

    df[rd] = pd.to_datetime(df[rd])
    df[sd] = pd.to_datetime(df[sd])

    # =================================================
    # Counts are not always re-reported if not changed.
    # In these cases --> carry the last report forward.
    # =================================================
    all_reports = []
    df = df.sort_values(sd)
    for i, dt in enumerate(df[sd].unique()):
        profile = df[df[sd] == dt]
        profile = profile.sort_values(rd)
        profile.index = profile[rd]

        # reindex using report dates after dt
        valid_report_dates = report_dates[report_dates > dt]
        profile = profile.reindex(valid_report_dates, method="ffill")

        # sometimes a new row is created at start so stays empty after ffill
        # --> fill the NaNs
        profile[dlcc] = profile[dlcc].fillna(0)
        profile[sd] = dt
        profile[rd] = profile.index
        all_reports.append(profile)
    all_df = pd.concat(all_reports).reset_index(drop=True)

    # =================================================
    # ignore any lag 0 reports
    # -- these don't always exist (and shouldn't)
    # =================================================
    all_df = all_df.drop(all_df[all_df[rd] == all_df[sd]].index)

    # the final reported count by date and LTLA
    final_count = (
        all_df.rename(columns={dlcc: "final_count"})
        .sort_values(rd)
        .groupby(sd)
        .tail(1)
    )

    # combine
    data = all_df.merge(final_count[[sd, "final_count"]], on=sd, how="left")

    # add columns
    data["prop_reported"] = data[dlcc] / data["final_count"]
    # if final report is 0 --> reported proportion is 1
    data.loc[data['final_count']==0, 'prop_reported'] = 1.0
    # correct for over-reporting (don't allow report proportions > 1.0)
    data.loc[data['prop_reported']>1.0, 'prop_reported'] = 1.0
    data["lag"] = (data[rd] - data[sd]).dt.days

    data = data.sort_values(sd)

    return data


def create_profiles(df):
    """
    Format df into reporting profiles for each SD.

    :param df : Pandas dataframe, returned by `get_data()`
    :returns: list of tensors, a reporting profile for each SD in df
    """
    profiles = []
    df = df.sort_values(sd)
    for i, dt in enumerate(df[sd].unique()):
        profile = df[df[sd] == dt]
        profile = profile.sort_values(rd)
        profiles.append(profile[dlcc])

    return [t.from_numpy(np.array(profile)).float() for profile in profiles]


def get_beta_params(mu, var):
    """
    Return parameters of the Beta distribution from mean and variance.
    Includes sensibility checks:
        - if both mean and variance are 0, return a flat prior [1,1]
        - don't return params with values <= 0

    :param mu : float, mean of the distribution
    :param var : float, variance of the distribution
    :returns: list of floats, the [alpha, beta] parameters of the Beta distribution
    """
    if mu==0 and var==0:
        return [1., 1.]

    # avoid division by 0 error
    if var == 0:
        var = 10e-6

    # avoid returning params <= 0
    if mu >= 1.0:
        mu = 1-10e-5
    if var >= mu*(1-mu):
        var = mu*(1-mu)*0.9

    alpha = (mu ** 2 * (1 - mu)) / var - mu
    beta = (mu * (1 - mu) / var - 1) * (1 - mu)

    return [alpha, beta]


def moment_match_theta_priors(df, n_drop=4, lag_col='lag', prop_col='prop_reported', weighted=False, n_lags=None):
    """
    Get priors for theta given data in df using moment matching (for each lag in df or up to max_lag).

    :param df: Pandas dataframe with columns `lag_col` and `prop_col` (returned by `get_data()`)
    :param n_drop: int, number of recent days to drop (where expect true count not reported yet)
    :param weighted: bool, indicates whether to use linear weights in moment matching
    :param n_lags: optional int, indicates how many lags to return params for
    :returns: tuple (list, list), alpha and beta parameters of the Beta prior on theta
    """
    if n_drop > 0:
        dates = df[sd].sort_values().unique()[:-n_drop]
        df = df[df[sd].isin(dates)]

    # Sort ascending by Report date (most recent last)
    # --> if use weights, should be ascending too
    df = df.sort_values(rd)

    #====================================================================
    # for each lag, get mean and variance of the reporting proportion
    # as lag increases, number of data points decreases
    #====================================================================
    mus = []
    var = []
    if n_lags == None:
        n_lags = df["lag"].max()
    for lag in range(1, n_lags + 1):
        lag_data = df[df[lag_col] == lag][prop_col]

        if len(lag_data) <= 1:
            break
        if weighted:
            weights = np.array([i for i in range(1, len(lag_data)+1)])
            m = (sum(lag_data * weights))/sum(weights)
            v = sum(weights*(lag_data-m)**2)/sum(weights)
        else:
            m = lag_data.mean()
            v = lag_data.var()
        mus.append(m)
        var.append(v)

    # use means and vars to get alpha&beta params of the Beta prior over theta
    params = [get_beta_params(m, v) for m, v in zip(mus, var)]

    # if have fewer params than lags, use last value and extend
    fill =  n_lags - len(params)
    alphas = [param[0] for param in params] + [params[-1][0]] * fill
    betas = [param[1] for param in params] + [params[-1][1]] * fill

    return alphas[::-1], betas[::-1]
