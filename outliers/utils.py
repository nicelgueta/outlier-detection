import click
from pathlib import Path
import pandas as pd
import sys


def cli_error(message: str):
    """
    Print styled error for neater CLI error handling
    """
    click.secho(f"{click.style('Error: ', fg='red')}{message}")
    sys.exit(1)


def cli_read_csv(path: Path) -> pd.DataFrame:
    """
    print styled error if csv reading fails
    """
    try:
        df = pd.read_csv(path)
    except:
        cli_error(f"File {str(path)} cannot be read as csv")
    return df


def cli_print_outlier_output(data: pd.DataFrame, outliers: pd.Series):
    """
    Neatly print outlier results to CLI
    """
    click.secho("-" * 24, fg="cyan")
    click.secho("    Outliers Found")
    click.secho("-" * 24, fg="cyan")

    # ensure everything is always printed
    with pd.option_context(  # type: ignore
        "display.max_rows", len(data), "display.max_columns", len(data.columns)
    ):
        click.secho(data[outliers])

    click.secho("-" * 24, fg="cyan")
    click.secho(f"Total: {sum(outliers)}")
    click.secho("-" * 24, fg="cyan")


def cli_saved_removed_outliers(
    src_path: Path,
    data: pd.DataFrame,
    outliers: pd.Series,
    dest_path: Path = None,
):
    """
    Save data with outliers removed to disk with CLI styled logging
    """
    trimmed_df = data[~outliers]
    if not dest_path:
        dest_path = Path(src_path.parent, f"{src_path.stem}-altered{src_path.suffix}")
    trimmed_df.to_csv(dest_path, index=False)
    click.secho(
        f"Cleaned data saved to {click.style(dest_path, fg='cyan')}", fg="green"
    )
