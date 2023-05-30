import glob
import os
import re

import nbformat
from nbconvert import MarkdownExporter

_this_dir = os.path.join(os.path.dirname(__file__))

README_PATH = os.path.join(os.path.dirname(__file__), "../README.md")
with open(README_PATH, "r", encoding="utf8") as file:
    readme_content = file.read()

DIR_MAP = {
    os.path.join(_this_dir, "../tutorials/data"): os.path.join(_this_dir, "user_guide/data"),
    os.path.join(_this_dir, "../tutorials/usage"): os.path.join(_this_dir, "user_guide/usage"),
    os.path.join(_this_dir, "../tutorials/extending"): os.path.join(_this_dir, "user_guide/extending"),
}


def remove_last_and_on(s: str, substring: str) -> str:
    i = s.rfind(substring)
    return s[:i] if i >= 0 else s


def get_notebook_link(notebook_file: str) -> str:
    find = r"- \[!\[Test In Colab\].*?" + notebook_file + r"\)"
    found = re.findall(find, readme_content)
    if len(found) == 0:
        raise RuntimeError(
            f"The section corresponding to notebook links for notebook {notebook_file} not found in README.md"
        )
    return found[0][2:]


# NOTE: Images are not handled.
def convert_nb_to_md(notebook_path: str, output_path: str, notebook_link: str) -> None:
    # Load notebook
    with open(notebook_path, encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    # Initialize markdown exporter
    md_exporter = MarkdownExporter()

    # Convert to markdown
    body, resources = md_exporter.from_notebook_node(nb)  # pylint: disable=unused-variable

    # Remove the install step.
    body = re.sub(r"\*Skip.*?temporai.git\n```", "", body, flags=re.DOTALL)
    # Remove the "congratulations!" note.
    body = remove_last_and_on(body, "## 🎉 Congratulations!")

    # Add the notebook links.
    body = re.sub(r"#(.*?)\n", r"#\1\n" + notebook_link + "\n", body, flags=re.DOTALL, count=1)

    # Write markdown to file
    with open(output_path, "w", encoding="utf8") as f:
        f.write(body)


def do_conversion():
    for nb_dir, doc_dir in DIR_MAP.items():
        for nb_file in glob.glob(os.path.join(nb_dir, "*.ipynb")):
            print(f"Converting {nb_file}...")
            filename = os.path.basename(nb_file)
            nb_links = get_notebook_link(filename)
            filename_no_ext = filename.split(".")[0]
            md_file = os.path.join(doc_dir, f"{filename_no_ext}.md")
            convert_nb_to_md(nb_file, md_file, nb_links)
