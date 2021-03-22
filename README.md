
# JBC-RSS-Turing Nowcasting

A Bayesian model for time-series count data with weekend effects and a lagged reporting process.  

## Overview

The model was developed as part of the [JBC-Turing-RSS lab](https://www.turing.ac.uk/research/research-projects/new-partnership-between-alan-turing-institute-and-royal-statistical) to now-cast COVID-19 positive test counts. Pillar 2 PCR positive tests usually take about 4 to 5 days to process. Over this period, partial results are reported daily until the final count is reached. We make use of stability in the under reporting behaviour to infer the final count given the partial information as it arrives.

For more information on the model and inference read the [pre-print](arxiv_link).

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

## Data

The data consists of daily snapshots of COVID-19 positive test counts as reported on the [coronavirus dashboard](https://coronavirus.data.gov.uk). The snapshots are from October to December 2020 and stored as timestamped csv files. The files comes from an archive maintained [here](https://github.com/theosanderson/covid_uk_data_timestamped).

## Results

For code to reproduce results and plots in the report see the [experiments/](experiments/) directory.
