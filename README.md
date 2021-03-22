# Now-casting counts under reporting lag

A Bayesian time-series model for now-casting counts under reporting lag.  

The model was developed to now-cast COVID-19 positive test counts. PCR positive tests usually take about 4 to 5 days to process. Over this period, partial results are reported daily until the final count is reached. We make use of stability in the under reporting behaviour to infer the final count given the partial information as it arrives.

For more information on the model implemented here, read the [pre-print](arxiv_link).

<!-- MAYBE EXAMPLE PLOT HERE -->

## Overview

This directory is structured as follows:
- `data`: directory for datasets
- `data_processing`: functions to load and format data for modelling
- `models`: code implementing the model and inference
- `notebooks`: demos
- `experiments`: scripts used to obtain results in the report

## Quick start

The model requires Python 3 to use.

1. Clone this repository

```{bash}
git clone https://github.com/alan-turing-institute/jbcc-rss-turing-nowcasting.git
cd jbcc-rss-turing-nowcasting
```

2. Install dependencies

```{bash}
<!-- pip install . -->
pip install -r requirements.txt
```

3. Start demo (requires `jupyter lab`)

```{bash}
jupyter lab notebooks/quick_start.ipynb
```

## Data

The data consists of daily snapshots of COVID-19 positive test counts as reported on the [coronavirus dashboard](https://coronavirus.data.gov.uk). The snapshots are from October to December 2020 and stored as timestamped csv files. A full archive from which we obtained the files is maintained [here](https://github.com/theosanderson/covid_uk_data_timestamped).
