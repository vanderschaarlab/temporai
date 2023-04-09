import re
from pathlib import Path
from time import time

import click
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

workspace = Path(__file__).parents[0] / "workspace"
workspace.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook_path: Path, skip_pip_install: bool = True) -> None:
    with open(notebook_path, encoding="utf8") as f:
        f_str = f.read()

    if skip_pip_install:
        # Comment out `pip install` commands, as do not need to run those in test.
        f_str = re.sub(r"[%!]\s*pip", "#pip", f_str, flags=re.DOTALL)

    nb = nbformat.reads(f_str, as_version=4)
    proc = ExecutePreprocessor(timeout=1800)

    # Will raise on cell error:
    proc.preprocess(nb, {"metadata": {"path": workspace}})


# For fine-grained control of which notebooks to test, adjust this:
enabled_tests = [
    "tutorial",  # All tutorial notebooks include "tutorial" in file name.
]


@click.command()
@click.option("--nb_dir", type=str, default=".")
def main(nb_dir: Path) -> None:
    nb_dir = Path(nb_dir)

    for p in nb_dir.rglob("*"):
        if p.suffix != ".ipynb":
            continue
        if "checkpoint" in p.name:
            continue

        ignore = True
        for val in enabled_tests:
            if val in p.name:
                ignore = False
                break
        if ignore:
            continue

        print("Testing notebook:", p.name)
        start = time()
        try:
            run_notebook(p)
        except BaseException as e:
            print("FAIL", p.name, e)
            raise e
        finally:
            print(f"Notebook {p.name} took {time() - start:.2f} sec")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
