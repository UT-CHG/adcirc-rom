# ADCIRC ROM
ADCIRC ROM (Reduced-Order-Modeling) is suite of tools for developing surrogate machine learning models of storm surge.
The tools can be used both in an HPC and a single-threaded environment.

## Designsafe Quickstart

1. Start a Jupyter lab session.

2. Use Jupyter lab to launch a terminal, and in the terminal run the following:
```
pip install netCDF4 sklearn global-land-mask xgboost geopandas scipy h5py fire
git clone https://github.com/UT-CHG/adcirc-rom.git
cd adcirc-rom
python3 dataset.py setup
```

This will create a `data` folder with the subdirectories `datasets`, `storms` and `models`.
These subdirectories are needed for the model development workflow. The `datasets` directory is used for storing
machine-learning ready datasets. The `storms` directory will contain the raw ADCIRC input (note when run within the DesignSafe
environment this directory will be prepopulated with a dataset of 446 synthetic ADCIRC simulations).
Finally, the `models` dataset is used for storing saved ML models and predictions.

3. To generate a dataset, run the command 
```
python3 dataset.py create default
```

This will create a dataset named 'default' in the directory `data/datasets`.
This dataset can be used to train machine learning models.
The `dataset.py` script takes a number of options that control the size and scope of the generated dataset,
as well as the included features.

Note: with the default settings, dataset
creation in the Designsafe environment will take a few hours due to the lack of MPI support and the
size of the data to be processed. The dataset generation script supports parallization with MPI - and is significantly
faster when run on HPC resources such as TACC.
 

4. To train and save a new model named 'xgb_base', using the dataset named default, run the command
```
python3 model.py train xgb_base --dataset=default
```

This will create a new model named 'xgb_base'. During, training, a portion of the dataset is set aside
for testing purposes - predictions are generated for the test dataset and saved alongside the model binary.
Additional model training parameters can be passed to the script.

To perform cross-validation, run the command

```
python3 model.py cv --dataset=default
```

This will also print out feature importances. Note that all model training parameters that are supported by the
`train` are supported by the `cv` command as well. This allows for testing out of parameters before training a model.

Finally, to generate predictions on a new dataset using a saved model, run
```
python3 model.py predict [modelname] --dataset=[datasetname]
```

All predictions can be accessed in the folder `data/datasets/[modelname]`.

Please reach out with any questions or bug reports to Benjamin Pachev <benjamin.pachev@gmail.com>.
