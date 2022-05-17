from click.testing import CliRunner
import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from ..cli import main


@pytest.fixture(scope="module")
def tcli():
    return CliRunner()


def test_happy(tcli):
    # arrange
    df = pd.DataFrame(np.random.rand(100, 2), columns=["a", "b"])
    tf = tempfile.NamedTemporaryFile()
    df.to_csv(tf.name)
    result = tcli.invoke(main, [tf.name, "b", "--dest-path", tf.name])
    assert result.exit_code == 0


def test_wrong_field_name(tcli):
    # arrange
    df = pd.DataFrame(np.random.rand(100, 2), columns=["a", "b"])
    tf = tempfile.NamedTemporaryFile()
    df.to_csv(tf.name)

    # act
    result = tcli.invoke(main, [tf.name, "c", "--dest-path", tf.name])

    assert result.exit_code == 1
    assert result.stdout == "Error: \"KeyError: column 'c' not a valid column name\"\n"


def test_no_field_name(tcli):
    df = pd.DataFrame(np.random.rand(100, 2), columns=["a", "b"])
    tf = tempfile.NamedTemporaryFile()
    df.to_csv(tf.name)

    result = tcli.invoke(main, [tf.name, "--dest-path", tf.name])

    assert result.exit_code == 2


def test_invalid_file_name(tcli):
    result = tcli.invoke(main, ["gibberish", "c"])
    p = Path("gibberish").resolve()
    assert result.exit_code == 1
    assert result.stdout == f"Error: File '{p}' does not exist\n"


def test_invalid_csv(tcli):
    tf = tempfile.NamedTemporaryFile()
    with open(tf.name, "w") as f:
        f.write("")

    result = tcli.invoke(main, [tf.name, "c"])

    assert result.exit_code == 1
    assert result.stdout == f"Error: File {tf.name} cannot be read as csv\n"


def test_using_str_instead_of_float_vector(tcli):
    df = pd.DataFrame(
        [["hello", "world"], ["Hello again", "world"]], columns=["a", "b"]
    )
    tf = tempfile.NamedTemporaryFile()
    df.to_csv(tf.name)
    result = tcli.invoke(main, [tf.name, "b", "--dest-path", tf.name])

    assert result.exit_code == 1
    assert (
        result.stdout
        == "Error: Cannot convert series to float. Ensure vector contains valid numerical data\n"
    )
