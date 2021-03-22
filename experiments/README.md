# Experiments

To create results in the report, we run the model for all Lower tier local authorities (LTLAs) using a range of values for the random walk scale in the model.

To recreate results for all LTLAs using a model with a random walk scale of 4, run:

```{bash}
python run.py -s 4
```

For the report we run the script with random walk scale values of 1 to 9. For each LTLA, we saved the model with the scale that had the highest model evidence. These are the models used to produce the final results. The optimised models (one per LTLA) are saved in the `results/` directory.

For code used to create plots in the report using the results:

```{bash}
jupyter lab reports_figs.ipynb
```
