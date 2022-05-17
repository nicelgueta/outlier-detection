import click
from pathlib import Path
from .algorithm import (
    OutlierDetectionModel,
    TrailingZScoreConfig,
    DEFAULT_LOOKBACK_WINDOW,
)
from .utils import (
    cli_error,
    cli_print_outlier_output,
    cli_read_csv,
    cli_saved_removed_outliers,
)


@click.command()
@click.argument("src-path", type=Path)
@click.argument("field", type=str)
@click.option(
    "--dest-path",
    default=None,
    help=(
        "Optionally add name of output file. "
        "Filename is suffixed with '-altered' if this is not provided"
    ),
)
@click.option(
    "--inc-current",
    default=False,
    type=bool,
    help=(
        "Include the current observation in the z-score calc. This determines "
        "the scoring strategy described in the readme"
    ),
)
@click.option(
    "--lookback",
    default=DEFAULT_LOOKBACK_WINDOW,
    help="Look back window used to calculate z scores",
    type=int,
)
def main(src_path: Path, field: str, dest_path: Path, inc_current: bool, lookback: int):
    """
    \b

    Remove outliers from a CSV.

    Required Arguments:

    - SRC-PATH: Path: Path to the csv to remove outliers from

    - FIELD: str: Column name of discrete variable evaluate
    """

    config = TrailingZScoreConfig(
        lookback_window=lookback, z_score_incl_current=inc_current
    )
    src_path = src_path.resolve()
    if not src_path.is_file():
        cli_error(f"File '{str(src_path)}' does not exist")

    data = cli_read_csv(src_path)
    model = OutlierDetectionModel(config)

    try:
        if field not in data.columns:
            raise KeyError(f"KeyError: column '{field}' not a valid column name")
        outliers = model.fit_predict(data[field])
    except Exception as e:
        cli_error(str(e))

    cli_print_outlier_output(data, outliers)

    cli_saved_removed_outliers(src_path, data, outliers, dest_path)


@click.group()
def cli():
    pass


cli.add_command(main)

if __name__ == "__main__":
    cli()
