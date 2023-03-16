"""
Datasets

Setup and create datasets.
"""
import os

import click

from adcirc_rom.constants import SUPPORTED_MODELS


@click.group()
def cli():
    """
    ADCIRC Reduced Order Model CLI
    """
    pass


@cli.group()
def dataset():
    """
    Subgroup of commands for setting up and generating datasets for ML models.
    """
    pass


@dataset.command()
@click.option(
    "--datadir",
    default="data",
    help="The data directory to be created",
    show_default=True,
)
@click.option(
    "--projectdir",
    default=os.path.expandvars("$HOME/NHERI-Published/PRJ-2968"),
    help="The project directory",
    show_default=True,
)
def setup(datadir, projectdir):
    """
    Setup the folder structure for analysis to work

    Setup folder structure comprising of:
      - datasets: Directory is used for storing machine-learning ready datasets.
      - storms: Directory containing raw ADCIRC input (note when run within
                the DesignSafe environment this directory will be prepopulated
                with a dataset of 446 synthetic ADCIRC simulations).
      - models: Directory containing saved ML models and predictions.
    """
    from adcirc_rom.dataset import Dataset

    ds = Dataset()
    ds.setup(datadir=datadir, projectdir=projectdir)


@dataset.command()
@click.argument("name")
@click.option(
    "--datadir",
    default="data",
    help="The path to the directory to store dataset.",
    show_default=True,
)
@click.option(
    "--stormsdir",
    default="storms",
    help="The name of the directory that contains the storm data.",
    show_default=True,
)
@click.option(
    "-p/-np",
    "--parallel/--no-parallel",
    default=False,
    is_flag=True,
    help="The name of the directory that contains the storm data.",
    show_default=True,
)
@click.option(
    "--hours-before",
    default=6,
    help="Number of hours before a storm to consider.",
    show_default=True,
)
@click.option(
    "--hours-after",
    default=6,
    help="Number of hours after a storm to consider.",
    show_default=True,
)
@click.option(
    "--cutoff-coastal-dist",
    default=30,
    help="Cutoff coastal distance in kilometers.",
    show_default=True,
)
@click.option(
    "--max-depth",
    default=2,
    help="Maximum depth of the bathymetry data in meters.",
    show_default=True,
)
@click.option(
    "--min-depth",
    default=-4,
    help="Minimum depth of the bathymetry data in meters.",
    show_default=True,
)
@click.option(
    "-r", default=150, help="The radius of influence in kilometers.", show_default=True
)
@click.option(
    "--downsample-factor",
    default=100,
    help="The factor to downsample the data by.",
    show_default=True,
)
@click.option(
    "-b",
    "--bounds",
    multiple=True,
    type=float,
    default=(24, 32, -98, -88),
    help="The bounds of the domain in the form "
    + "(lat_min, lat_max, lon_min, lon_max).",
    show_default=True,
)
def create(
    name,
    datadir,
    stormsdir,
    parallel,
    hours_before,
    hours_after,
    cutoff_coastal_dist,
    max_depth,
    min_depth,
    r,
    downsample_factor,
    bounds,
):
    """
    Create a dataset for training ML models

    Creates a new dataset by loading and processing storm data stored in
    the directory specified by `datadir`. The processed data is then stored
    in a format that is suitable for use with the model.py library and
    entrypoints to train and validate models.

    Note: If mip4py is not installed, dataset creation in a serial environment
    (i.e. Designsafe jupyter) will take about 2 hours due to the lack of MPI
    and the size of the data to be processed. The dataset generation script
    supports parallization with MPI - and is significantly faster when run on
    HPC resources such as TACC.
    """
    from adcirc_rom.dataset import Dataset

    ds = Dataset()
    ds.create(
        name=name,
        datadir=datadir,
        stormsdir=stormsdir,
        parallel=parallel,
        hours_before=hours_before,
        hours_after=hours_after,
        cutoff_coastal_dist=cutoff_coastal_dist,
        max_depth=max_depth,
        min_depth=min_depth,
        r=r,
        downsample_factor=downsample_factor,
        bounds=bounds,
    )


@cli.group()
def model():
    """
    Subgroup of commands for training ML models and using them to generate
    predictions.
    """
    pass


@model.command()
@click.option(
    "--dataset",
    default="default",
    help="Name of feature data-set hdf5 file. (Created with `arom "
    + "dataset create`)",
    show_default=True,
)
@click.option(
    "--datadir",
    default="data",
    help="The path to the directory to store dataset.",
    show_default=True,
)
@click.option(
    "-l/-nl",
    "--latlon/--no-latlon",
    default=False,
    is_flag=True,
    help="Flag to include latlon in training model or not.",
    show_default=True,
)
@click.option(
    "-b/-nb",
    "--bathy/--no-bathy",
    default=False,
    is_flag=True,
    help="Flag to include bathymetry or not in training the model.",
    show_default=True,
)
@click.option(
    "-o/-no",
    "--order/--no-order",
    default=False,
    is_flag=True,
    help="TODO: document",
    show_default=True,
)
@click.option(
    "--classifier",
    default="nn1",
    help="Classifier model name",
    type=click.Choice(SUPPORTED_MODELS.keys(), case_sensitive=False),
    show_default=True,
)
@click.option(
    "--regressor",
    default="nn1",
    help="Regressor model name",
    type=click.Choice(SUPPORTED_MODELS.keys(), case_sensitive=False),
    show_default=True,
)
@click.option("--epochs", default=100, help="Number of epochs", show_default=True)
@click.option(
    "--preprocess",
    default=None,
    type=click.Choice(["pca", "importance", "correlation"], case_sensitive=False),
    help="Preprocessing method",
    show_default=True,
)
@click.option("--modelname", default=None, help="Name for the trained model")
@click.option(
    "--pca_components",
    default=50,
    help="Number of PCA components to keep",
    show_default=True,
)
@click.option(
    "--correlation_threshold",
    default=0.9,
    help="Correlation threshold for feature selection",
    show_default=True,
)
def train(
    dataset,
    datadir,
    latlon,
    bathy,
    order,
    classifier,
    regressor,
    epochs,
    preprocess,
    modelname,
    pca_components,
    correlation_threshold,
):
    """Train the stacked model"""
    from adcirc_rom.stacked_model import StackedModel

    # Call the train function with the provided arguments
    model = StackedModel(
        dataset=dataset,
        datadir=datadir,
        include_latlon=latlon,
        exclude_bathy=bathy,
        ordered_split=order,
    )

    res = model.train(
        classifier=classifier,
        regressor=regressor,
        epochs=epochs,
        preprocess=preprocess,
        modelname=modelname,
        pca_components=pca_components,
        correlation_threshold=correlation_threshold,
    )
    print(f"Results: {res}")


@model.command()
@click.option(
    "-m" "--modelname",
    type=str,
    required=True,
    help="Name of the trained model to use for prediction.",
)
@click.option(
    "-t",
    "--test_only",
    is_flag=True,
    help="Whether to generate predictions for the test set only.",
)
def predict(modelname, test_only):
    """
    Generate predictions for the given dataset.

    If test_only is True, assume this is the original dataset the model was trained with
    and regenerate predictions for the test set. Otherwise, generated named predictions
    for the entire datset.
    """

    """Train the stacked model"""
    from adcirc_rom.stacked_model import StackedModel

    # Call the train function with the provided arguments
    model = StackedModel()

    out_file = model.predict(modelname, test_only=test_only)

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    cli()
