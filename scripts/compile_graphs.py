#!/usr/bin/env python
import click
import pandas as pd
from pathlib import Path

from cutie import __version__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
@click.option('-f', '--first_file', required=True,
              type=click.Path(exists=True),
              help='Path to the first input data file')
@click.option('-s', '--second_file', required=True,
              type=click.Path(exists=True),
              help='Path to the second input data file')
@click.option('-d', '--cutie_directory', required=True,
              type=click.Path(exists=True),
              help='Path to the cutie work directory')
@click.option('-t', '--threshold', default=0.9,
              type=float, help='Path the the cutie work directory')
def compile_graphs(first_file, second_file, cutie_directory, threshold):
    base_dir = Path(cutie_directory)
    correlation_df = pd.read_csv(base_dir / 'data_processing' / 'summary_df_resample_1.txt', sep='\t')
    first_df = pd.read_csv(first_file, sep='\t')
    second_df = pd.read_csv(second_file, sep='\t')

    # Get the cols for their index
    first_cols = first_df.axes[1]
    second_cols = second_df.axes[1]

    correlation_df.sort_values(by='correlations', axis=0, inplace=True, ascending=False)
    print(correlation_df)

    # graph_dir = base_dir /


if __name__ == "__main__":
    compile_graphs()
