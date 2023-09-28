"""A docs building code that must be ran when building TemporAI docs.

The code will:
* Update ``docs/overview.md`` based on the main ``README.md`` (carries out some string substitutions etc.).
* Create the PyPI version of ``README.md``, ``pypi.md``, with image links fixed.
"""

import os
import re

from pre_build_notebooks import do_pre_build_notebooks_tasks

REPO_URL_ROOT = "https://github.com/vanderschaarlab/temporai/"
REPO_URL_TREE = f"{REPO_URL_ROOT}tree/main/"

# -- Convert `README.md` into `overview.md`.
print("Working on `docs/overview.md`...")

README_PATH = os.path.join(os.path.dirname(__file__), "../README.md")
OVERVIEW_PATH = os.path.join(os.path.dirname(__file__), "overview.md")

REPLACE = {
    # Add more as necessary.
    "[User Guide][docs/user_guide]": "[User Guide](user_guide/index)",
    "<!-- include_docs": "",
    "include_docs_end -->": "",
    "./docs/": "",
    "docs/": "",
    "./#-": "#",
    "./": REPO_URL_TREE,
}

with open(README_PATH, "r", encoding="utf8") as file:
    readme_content = file.read()

# Replace:
for k, v in REPLACE.items():
    readme_content = readme_content.replace(k, v)

# Remove parts that should only be in repo `README.md`.
readme_content = re.sub(r"\n<!-- exclude_docs -->.*?<!-- exclude_docs_end -->", "", readme_content, flags=re.DOTALL)

# Make ```python ... ``` into sphinx.ext.doctest blocks.
skip_output_check = "import os; import sys; f = open(os.devnull, 'w'); sys.stdout = f"
# ^ Add a testsetup block to redirect output to devnull in order to skip output checking here.
readme_content = re.sub(
    r"```python(.*?)```",
    r"```{testcode}\n:hide:\n" + skip_output_check + r"\n\1```\n```python\1```",
    readme_content,
    flags=re.DOTALL,
)
# ^ Add a hidden copy of the ```python ... ``` blocks as testcode blocks with output check skipped.
readme_content = re.sub(
    r"```{testcode}(.*)\n(.*?)# doctest: \+SKIP(.*?)```python",
    r"```{testcode}\1\n#\2\3```python",
    readme_content,
    flags=re.DOTALL,
)
# ^ Comment out any lines that include "# doctest: +SKIP" (in {testcode}```...``` only).

# Make emoji representations compatible with sphinxemoji, e.g. :key: --> |:key:|
# readme_content = re.sub(r"\:[a-z1-9+\-_]{0,100}\:", (lambda x: f"|{x.group(0)}|"), readme_content)

with open(OVERVIEW_PATH, "w", encoding="utf8") as file:
    file.write(readme_content)


# -- Convert `README.md` into `pypi.md`.
print("Working on `pypi.md`...")

PYPI_README_PATH = os.path.join(os.path.dirname(__file__), "../pypi.md")

REPLACE = {
    # Add more as necessary.
    "./": REPO_URL_TREE,
}

with open(README_PATH, "r", encoding="utf8") as file:
    readme_content = file.read()

# Replace:
for k, v in REPLACE.items():
    readme_content = readme_content.replace(k, v)

# Fix images:
convert = {
    r"\"docs/assets/(.*?\..*?)\"": r"'https://raw.githubusercontent.com/vanderschaarlab/temporai/main/docs/assets/\1'",
    r"\[docs/assets/(.*?\..*?)\]": r"[https://raw.githubusercontent.com/vanderschaarlab/temporai/main/docs/assets/\1]",
}
for source, destination in convert.items():
    readme_content = re.sub(source, destination, readme_content, flags=re.DOTALL)

with open(PYPI_README_PATH, "w", encoding="utf8") as file:
    file.write(readme_content)


# Convert tutorial notebooks into user guide pages.
do_pre_build_notebooks_tasks()
