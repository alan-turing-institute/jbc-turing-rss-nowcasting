
# JBC-Turing-RSS Nowcasting

A Bayesian model for time-series count data with weekend effects and a lagged reporting process.  

## Overview

The model was developed within the [JBC-Turing-RSS lab](https://www.turing.ac.uk/research/research-projects/providing-independent-research-leadership-joint-biosecurity-centre) to now-cast COVID-19 positive test counts. In the UK, Pillar 2 PCR positive tests usually take about 4 to 5 days to process. Over this period, partial results are reported daily until the final count is reached. We make use of stability in the under reporting behaviour to infer the final count given the partial information as it arrives.

For more information on the model and inference read the [pre-print](https://arxiv.org/abs/2103.12661).

<!-- EXAMPLE PLOT HERE -->

## Quick start

The model requires Python 3 to use.

1. Clone this repository

```{bash}
git clone https://github.com/alan-turing-institute/jbcc-rss-turing-nowcasting.git
cd jbcc-rss-turing-nowcasting
```

2. Install dependencies

```{bash}
pip install -r requirements.txt
```

3. Start demo (requires `jupyter lab`)

```{bash}
jupyter lab notebooks/quick_start.ipynb
```

## Stan implementation

To see an example implementation of the model in Stan with a hyperprior on the random walk scale:

```{bash}
jupyter lab notebooks/stan_version.ipynb
```

## Results

For code to reproduce results and plots in the paper see the [experiments/](experiments/) directory.
