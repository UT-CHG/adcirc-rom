
"""
Datasets

Setup and create datasets.
"""
import os
import click
from adcirc_rom.dataset import Dataset


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
@click.option('--datadir', default='data',
              help='The data directory to be created')
@click.option('--projectdir',
              default=os.path.expandvars("$HOME/NHERI-Published/PRJ-2968"),
              help='The project directory')
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
    ds = Dataset()
    ds.setup(datadir=datadir, projectdir=projectdir)


@dataset.command()
@click.argument('name')
@click.option('--datadir', default='data',
              help='The path to the directory to store dataset.')
@click.option('--stormsdir', default='storms',
              help='The name of the directory that contains the storm data.')
@click.option('--hours-before', default=6,
              help='Number of hours before a storm to consider.')
@click.option('--hours-after', default=6,
              help='Number of hours after a storm to consider.')
@click.option('--cutoff-coastal-dist', default=30,
              help='Cutoff coastal distance in kilometers.')
@click.option('--max-depth', default=2,
              help='Maximum depth of the bathymetry data in meters.')
@click.option('--min-depth', default=-4,
              help='Minimum depth of the bathymetry data in meters.')
@click.option('-r', default=150,
              help='The radius of influence in kilometers.')
@click.option('--downsample-factor', default=100,
              help='The factor to downsample the data by.')
@click.option('--bounds', default=(24, 32, -98, -88),
              help='The bounds of the domain in the form ' +
              '(lat_min, lat_max, lon_min, lon_max).')
def create(name,
           datadir,
           stormsdir,
           hours_before,
           hours_after,
           cutoff_coastal_dist,
           max_depth,
           min_depth,
           r,
           downsample_factor,
           bounds):
    """
    Create a dataset for training ML models

    Creates a new dataset by loading and processing storm data stored in
    the directory specified by `datadir`. The processed data is then stored
    in a format that is suitable for use with the model.py library and
    entrypoints to train and validate models.

    Note: If mip4py is not installed, dataset creation in a serial environment
    (i.e. Designsafe jupyter) will take a few hours due to the lack of MPI
    and the size of the data to be processed. The dataset generation script
    supports parallization with MPI - and is significantly faster when run on
    HPC resources such as TACC.
    """
    ds = Dataset()
    ds.create(name=name,
              datadir=datadir,
              stormsdir=stormsdir,
              hours_before=hours_before,
              hours_after=hours_after,
              cutoff_coastal_dist=cutoff_coastal_dist,
              max_depth=max_depth,
              min_depth=min_depth,
              r=r,
              downsample_factor=downsample_factor,
              bounds=bounds)


if __name__ == "__main__":
    cli()
