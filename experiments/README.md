# Experiments

The scripts here were used to produce results and plots in the report. Note that all scripts should be run from within this directory.

## Data

We fit the model to data as reported on the 14th of December 2020 capturing the period when England was transitioning out of the second lockdown. We learned the reporting lag priors from two weeks of most recent data where we believed the reports to have converged to the final count (26th of November-9th of December 2020).

## Results

The only free parameter in the model is the scale ($\sigma$) of the temporal smoothing applied by the random walk prior. To fit the model to all Lower Tier Local Authorities (LTLAs) using a given random walk scale:

```{bash}
python run.py -s <random_walk_scale>
```

We run the model with $\sigma$ values in a range 1-9. For each LTLA we then chose the model with the $\sigma$ that had the highest model evidence:

```{bash}
python select_model.py
```

The final optimised models are saved in the `results/` directory.

## Plots

For code to create plots of the results:

```{bash}
jupyter lab report_figs.ipynb
```
