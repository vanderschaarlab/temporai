"""A test script to run all notebooks in a directory."""

import pprint
import re
import time
from pathlib import Path
from typing import List

import click
import joblib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

workspace = Path(__file__).parents[0] / "workspace"
workspace.mkdir(parents=True, exist_ok=True)


# For fine-grained control of which notebooks to test, adjust this:
ENABLED_TESTS = [
    "tutorial",  # All tutorial notebooks include "tutorial" in file name.
]


def filter_notebooks(all_paths: List, enabled_tests: List) -> List:
    """Filter out non-notebooks and notebooks that are not enabled.

    Args:
        all_paths (List): List of paths to notebooks.
        enabled_tests (List): List of strings that must be in the notebook name.

    Returns:
        List: List of paths to notebooks that are enabled.
    """
    return [
        p
        for p in all_paths
        if any(val in p.name for val in enabled_tests) and p.suffix == ".ipynb" and "checkpoint" not in p.name
    ]


def run_notebook(notebook_path: Path, skip_pip_install: bool = True) -> None:
    """Run a notebook.

    Args:
        notebook_path (Path): Path to notebook.
        skip_pip_install (bool, optional): Comment out the pip install commands in cells if present. Defaults to True.
    """
    with open(notebook_path, encoding="utf8") as f:
        f_str = f.read()

    if skip_pip_install:
        # Comment out `pip install` commands, as do not need to run those in test.
        f_str = re.sub(r"[%!]\s*pip", "#pip", f_str, flags=re.DOTALL)

    nb = nbformat.reads(f_str, as_version=4)
    proc = ExecutePreprocessor(timeout=6000)

    # Will raise on cell error:
    proc.preprocess(nb, {"metadata": {"path": workspace}})


@joblib.delayed
def test_notebook(p: Path) -> None:
    """Test a notebook.

    Args:
        p (Path): Path to notebook.

    Raises:
        BaseException: Exception raised if notebook fails.
    """
    print("Testing notebook:", p.name)
    start = time.time()
    try:
        run_notebook(p)
    except BaseException as e:
        print("FAIL", p.name, e)
        raise e
    finally:
        print(f"Notebook {p.name} took {time.time() - start:.2f} sec")


@click.command()
@click.option("--nb_dir", type=str, default=".", help="Path to directory containing notebooks to test")
@click.option("-n", "--n_jobs", type=int, default=-1, help="Number of joblib Parallel jobs")
@click.option("-v", "--verbose", type=int, default=0, help="Verbosity value for joblib Parallel")
def main(nb_dir: Path, n_jobs: int, verbose: int) -> None:
    """Test all notebooks in a directory.

    Args:
        nb_dir (Path): Path to directory containing notebooks to test.
        n_jobs (int): Number of joblib Parallel jobs.
        verbose (int): Verbosity value for joblib Parallel.
    """

    nb_dir = Path(nb_dir)
    notebook_paths = filter_notebooks(list(nb_dir.rglob("*")), ENABLED_TESTS)
    print("Notebooks to be tested:")
    pprint.pprint([str(p) for p in notebook_paths])

    print(f"Running n_jobs={n_jobs} parallel jobs. Note: -1 means as many jobs as available CPUs.")
    start = time.time()
    joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(test_notebook(p) for p in notebook_paths)
    print(f"Testing all notebooks took {time.time() - start:.2f} sec")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
