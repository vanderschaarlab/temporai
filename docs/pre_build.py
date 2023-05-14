"""A docs building code that must be ran when building TemporAI docs.

The code will:
* Update ``docs/requirements.txt`` with the up-to-date requirements from the library.
* Update ``docs/overview.md`` based on the main ``README.md`` (carries out some string substitutions etc.).
* Create the PyPI version of ``README.md``, ``pypi.md``, with image links fixed.
"""

import os
import re

REPO_URL = "https://github.com/vanderschaarlab/temporai/"


# -- Update `docs/requirements.txt` with the content of `install_requires` in `setup.cfg`.

SETUP_CFG_PATH = os.path.join(os.path.dirname(__file__), "../setup.cfg")
DOCS_REQS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(SETUP_CFG_PATH, "r", encoding="utf8") as file:
    setup_cfg_content = file.read()
final_specs = ""
for name, block in zip(("install_requires", "dev"), (r"install_requires\s=.*?\n\n", r"dev\s=.*?\n\n")):
    found = re.findall(block, setup_cfg_content, re.DOTALL)
    list_dep_specs = found[0].split("\n    ")[1:]
    list_dep_specs[-1] = list_dep_specs[-1].replace("\n", "")
    list_dep_specs = [dep for dep in list_dep_specs if dep[0] != "#"]  # Remove comment lines.
    final_specs += f"# {name}:\n" + "\n".join(sorted(list_dep_specs)) + "\n"
with open(DOCS_REQS_FILE, "r", encoding="utf8") as file:
    docs_req_content = file.read()
found = re.findall(r"# ----- auto_update -----.*# ----- auto_update -----", docs_req_content, re.DOTALL)[0]
new = f"# ----- auto_update -----\n{final_specs}# ----- auto_update -----"
docs_req_content = docs_req_content.replace(found, new)  # type: ignore

with open(DOCS_REQS_FILE, "w", encoding="utf8") as file:
    file.write(docs_req_content)

# -- Convert `README.md` into `overview.md`.

README_PATH = os.path.join(os.path.dirname(__file__), "../README.md")
OVERVIEW_PATH = os.path.join(os.path.dirname(__file__), "overview.md")

REPLACE = {
    # Add more as necessary.
    "[User Guide][docs/user_guide]": "[User Guide](user_guide/index)",
    "<!-- include_docs": "",
    "include_docs_end -->": "",
    "./docs/": "",
    "docs/": "",
    "./": REPO_URL,
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

PYPI_README_PATH = os.path.join(os.path.dirname(__file__), "../pypi.md")

REPLACE = {
    # Add more as necessary.
    "./": REPO_URL,
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
