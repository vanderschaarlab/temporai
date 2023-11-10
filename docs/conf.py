# This file is execfile()d with the current directory set to its containing dir.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import shutil
import sys
import time

# -- Path setup --------------------------------------------------------------

__location__ = os.path.dirname(__file__)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.join(__location__, "../src"))

# Any custom sphinx extensions for TemporAI live in docs/custom-sphinx-ext/:
sys.path.insert(0, os.path.join(__location__, "custom-sphinx-ext"))

# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See Issue:
# https://github.com/readthedocs/readthedocs.org/issues/1139
# DON'T FORGET: Check the box "Install your project inside a virtualenv using
# setup.py install" in the RTD Advanced Settings.
# Additionally it helps us to avoid running apidoc manually

try:  # for Sphinx >= 1.7
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc  # type: ignore

output_dir = os.path.join(__location__, "api")
module_dir = os.path.join(__location__, "../src/tempor")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    import sphinx

    cmd_line = f"sphinx-apidoc --implicit-namespaces -e -f -o {output_dir} {module_dir}"

    args = cmd_line.split(" ")
    if tuple(sphinx.__version__.split(".")) >= ("1", "7"):
        # This is a rudimentary parse_version to avoid external dependencies
        args = args[1:]

    apidoc.main(args)
except Exception as e:  # pylint: disable=broad-except
    print("Running `sphinx-apidoc` failed!\n{}".format(e))

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0.0"

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
    # "sphinx_immaterial.apidoc.python.apigen"
    # ^ Enable this if wishing to use https://jbms.github.io/sphinx-immaterial/apidoc/python/apigen.html
    "nbsphinx",
    # --- Custom extensions from here ---
    "sphinx-zeta-suppress",  # More specific warnings suppression.
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


# Enable markdown
extensions.append("myst_parser")

# Configure MyST-Parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]

# MyST URL schemes.
myst_url_schemes = {
    "http": None,
    "https": None,
    "ftp": None,
    "mailto": None,
    "repo-code": "https://github.com/vanderschaarlab/temporai/tree/main/{{path}}#{{fragment}}",
    # "doi": "https://doi.org/{{path}}",
    # "gh-issue": {
    #     "url": "https://github.com/executablebooks/MyST-Parser/issue/{{path}}#{{fragment}}",
    #     "title": "Issue #{{path}}",
    #     "classes": ["github"],
    # },
}

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "TemporAI"
copyright = f"{time.strftime('%Y')}, van der Schaar Lab"  # pylint: disable=redefined-builtin

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# version: The short X.Y version.
# release: The full version, including alpha/beta/rc tags.
# If you don’t need the separation provided between version and release,
# just set them both to the same value.
try:
    from tempor import __version__ as version
except ImportError:
    version = ""

if not version or version.lower() == "unknown":
    version = os.getenv("READTHEDOCS_VERSION", "unknown")  # automatically set by RTD

release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", ".dev"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "py:obj"

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
# https://pygments.org/styles/
pygments_style = "tango"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# A list of warning types to suppress arbitrary warning messages.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-suppress_warnings
suppress_warnings = ["misc.highlighting_failure"]
# ^ Highlighting failures seem to be common under this setup when types appear in literal blocks, since these are not
# critical, we suppress highlighting warnings.

# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# -- Configure autodoc ---------------------------------------------

autoclass_content = "both"

autodoc_member_order = "bysource"

# autodoc_mock_imports = ["sklearn"]  # Update as needed.

# -- Configure autodoc (end) ---------------------------------------


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# We use this theme: https://jbms.github.io/sphinx-immaterial/
html_theme = "sphinx_immaterial"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# Material theme options (see theme.conf for more information)
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://www.temporai.vanderschaar-lab.com/",
    "repo_url": "https://github.com/vanderschaarlab/temporai/",
    "repo_name": "TemporAI",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "light-blue",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "deep-purple",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    # BEGIN: version_dropdown
    "version_dropdown": False,
    "version_info": [
        {
            "version": "https://temporai.readthedocs.io/en/latest/",
            "title": "ReadTheDocs",
            "aliases": [],
        },
        # {
        #     "version": "https://jbms.github.io/sphinx-immaterial",
        #     "title": "Github Pages",
        #     "aliases": [],
        # },
    ],
    # END: version_dropdown
    "toc_title_is_page_title": True,
    # BEGIN: social icons
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/vanderschaarlab/temporai/",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/temporai/",
        },
    ],
    # END: social icons
}

# Sphinx immaterial theme's python apigen options:
# python_apigen_modules = {
#     "tempor": "src/tempor/",
# }

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "TemporAI documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "TemporAI"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "assets/TemporAI_Logo_Icon.ico"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "assets/TemporAI_Logo_Icon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "temporai-doc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {  # type: ignore
    # The paper size ("letterpaper" or "a4paper").
    # "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    # "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    # "preamble": "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [("index", "user_guide.tex", "TemporAI Documentation", "Evgeny Saveliev", "manual")]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = ""

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# `sphinx-zeta-suppress` (more specific warnings suppression) configuration.
#
# See:
# - https://sphinx-zeta-suppress.readthedocs.io/
#
# See also:
# - https://github.com/picnixz/sphinx-zeta-suppress
# - https://github.com/sphinx-doc/sphinx/issues/11325
# Note that since there is no PyPI package for sphinx-zeta-suppress, we add its python module as a custom extension,
# see: docs/custom-sphinx-ext/sphinx-zeta-suppress.py.

zeta_suppress_protect = [
    "sphinx_immaterial"
    # `sphinx-zeta-suppress` is not compatible with `sphinx_immaterial`. It will throw an error when registering the
    # filters (`_setup_filters` function). However, the problem can be overcome by adding `sphinx_immaterial` to the
    # `zeta_suppress_protect` list.
    # Note we can still suppress warnings specific to `sphinx_immaterial` by adding it to
    # the `zeta_suppress_records` list below.
]

zeta_suppress_records = [
    # The following warnings caused ultimately by sphinx_immaterial are caused by having *args/**kwargs in the
    # docstrings. We want to have those in docstrings, so we suppress these warnings.
    ["sphinx_immaterial", r".*Parameter name '\*args'.*"],
    ["sphinx_immaterial", r".*Parameter name '\*\*kwargs'.*"],
    # Similar but for other variable names used for the variadics:
    ["sphinx_immaterial", r".*Parameter name '\*dims'.*"],
    ["sphinx_immaterial", r".*Parameter name '\*\*params'.*"],
    # The annotations for `PipelineMeta` seem to have their own set of issues, ignore here:
    ["sphinx_immaterial", r".*Parameter name.*PipelineMeta.*"],
]

# `sphinx-zeta-suppress` (more specific warnings suppression) configuration [end].


# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "hyperimpute": ("https://hyperimpute.readthedocs.io/en/latest/", None),
    "xgbse": ("https://loft-br.github.io/xgboost-survival-embeddings/", None),
    "lifelines": ("https://lifelines.readthedocs.io/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}

print(f"loading configurations for {project} {version} ...", file=sys.stderr)
