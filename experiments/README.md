# Experiments

The scripts in this directory were used to produce results and plots in the report. Note that all scripts should be run from within this directory.

## Data and priors

We fit the model to data as reported on the 14th of December 2020 capturing the period when England was transitioning out of the second national lockdown. We learned priors on the reporting lag from the most recent two weeks of converged data where the final count is known (26th of November-9th of December 2020). We set the prior on the initial lambda to have the mean of the seven day moving average of positive test counts at the date of the first observation passed to the model.

## Results

The only free parameter in the model is the scale of the temporal smoothing applied by the random walk prior. To fit the model to all Lower Tier Local Authorities (LTLAs) using a given random walk scale:

```{bash}
python run.py -s <random_walk_scale>
```

We run the model with scale values of 1-9. For each LTLA we then chose the model with the scale that had the highest model evidence:

```{bash}
python select_model.py
```

The final optimised models are saved in the `results/` directory (one per LTLA).

## Plots

For code to create plots of the results:

```{bash}
jupyter lab report_figs.ipynb
```
