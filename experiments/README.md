# Experiments

We run the model on data as reported on the 14th of December. To recreate results for all LTLAs using a model with a given random walk scale run the below command (replace `<random_walk_scale>` with a value):

```{bash}
python run.py -s <random_walk_scale>
```

We run the model with random walk scale values of 1 to 9. For each LTLA, we saved the model with the scale that had the highest model evidence in the `results/` directory:

```{bash}
python select_model.py
```

For code to create plots of the final optimised results in the report:

```{bash}
jupyter lab reports_figs.ipynb
```
