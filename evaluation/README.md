# Evaluation

To generate various plots based on training data, take a look at the respective jupyter notebooks:

- [baseline](./baseline_plots.ipynb)
- [drac](./simple_visuals_drac.ipynb)
- [augmentation-regularization](./simple_visuals_regularization.ipynb)

Addtionally, to run the game in a matplotlib visual window, run [test_standalone.py](./test_standalone.py) with the following arguments, changing the environment and checkpoint as necessary:

```bash
python -m evaluation.test_standalone --checkpoint .\path\to\my\checkpoint --env_name chaser --type impoola
```
