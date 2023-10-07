import glob
import os
import re

import nbformat

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


def modify_and_copy_nb(notebook_path: str, output_path: str, notebook_link: str) -> None:
    # Load notebook
    with open(notebook_path, encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    # Remove cells not needed for tutorials in docs.
    new_cells = []
    for cell in nb.cells:
        # Remove the "Congratulations!" markdown cell.
        if cell.cell_type == "markdown" and "## 🎉 Congratulations!" in cell.source:
            continue
        # Remove the "Skip the below cell..." installation markdown cell.
        if cell.cell_type == "markdown" and "*Skip the below cell" in cell.source:
            continue
        # Remove the installation cell.
        if cell.cell_type == "code" and "pip install temporai" in cell.source:
            continue
        new_cells.append(cell)

    nb.cells = new_cells

    # Add links to Colab notebook.
    # Add notebook_link as the first cell
    link_cell = nbformat.v4.new_markdown_cell(notebook_link)
    link_cell.id = ""
    new_cells.insert(0, link_cell)  # Insert at the beginning
    nb.cells = new_cells

    # Write output file
    with open(output_path, "w", encoding="utf8") as f:
        nbformat.write(nb, f)


def remove_docs_ipynb_files(docs_nb_dir):
    ipynb_files = glob.glob(os.path.join(docs_nb_dir, "*.ipynb"))
    for file_path in ipynb_files:
        try:
            os.remove(file_path)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Could not delete old notebook file `{file_path}` due to:\n{str(e)}")


def do_pre_build_notebooks_tasks():
    for nb_dir, doc_dir in DIR_MAP.items():
        remove_docs_ipynb_files(doc_dir)
        for nb_file in glob.glob(os.path.join(nb_dir, "*.ipynb")):
            print(f"Converting {nb_file}...")
            filename = os.path.basename(nb_file)
            nb_links = get_notebook_link(filename)
            out_file = os.path.join(doc_dir, filename)
            modify_and_copy_nb(nb_file, out_file, nb_links)
