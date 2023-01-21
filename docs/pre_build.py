import os
import re

REPO_URL = "https://github.com/vanderschaarlab/temporai/"

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

with open(OVERVIEW_PATH, "w", encoding="utf8") as file:
    file.write(readme_content)
