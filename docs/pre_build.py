import os
import re

REPO_URL = "https://github.com/vanderschaarlab/temporai/"

# -- Update `docs/requirements.txt` with the content of `install_requires` in `setup.cfg`.

SETUP_CFG_PATH = os.path.join(os.path.dirname(__file__), "../setup.cfg")
DOCS_REQS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(SETUP_CFG_PATH, "r", encoding="utf8") as file:
    setup_cfg_content = file.read()
found = re.findall(r"install_requires\s=.*?\n\n", setup_cfg_content, re.DOTALL)
list_dep_specs = found[0].split("\n    ")[1:]
list_dep_specs[-1] = list_dep_specs[-1].replace("\n", "")
with open(DOCS_REQS_FILE, "r", encoding="utf8") as file:
    docs_req_content = file.read()
found = re.findall(r"# ----- auto_update -----.*# ----- auto_update -----", docs_req_content, re.DOTALL)[0]
new = "# ----- auto_update -----\n" + "\n".join(sorted(list_dep_specs)) + "\n# ----- auto_update -----"
docs_req_content = docs_req_content.replace(found, new)  # type: ignore

with open(DOCS_REQS_FILE, "w", encoding="utf8") as file:
    file.write(docs_req_content)

# -- Convert `README.md` into `overview.md`.
README_PATH = os.path.join(os.path.dirname(__file__), "../README.md")
OVERVIEW_PATH = os.path.join(os.path.dirname(__file__), "overview.md")

REPLACE = {
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
# Make emoji representations compatible with sphinxemoji, e.g. :key: --> |:key:|
readme_content = re.sub(r"\:[a-z1-9+\-_]{0,100}\:", (lambda x: f"|{x.group(0)}|"), readme_content)

with open(OVERVIEW_PATH, "w", encoding="utf8") as file:
    file.write(readme_content)
