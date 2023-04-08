# Contributing

Welcome to *TemporAI* contributor's guide.

> This guide is adapted from [PyScaffold] contributor's guide template.

This document focuses on getting any potential contributor familiarized with
the development processes, but [other kinds of contributions] are also appreciated.

If you are new to using [git] or have never collaborated in a project previously,
please have a look at [contribution-guide.org]. Other resources are also
listed in the excellent [guide created by FreeCodeCamp].

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt,
[Python Software Foundation's Code of Conduct] is a good reference in terms of
behavior guidelines.

> 🕮 Please also make sure to first familiarize yourself with the [`vanderschaarlab` Code of Conduct](https://github.com/vanderschaarlab/.github/blob/main/CODE_OF_CONDUCT.md).

## Issue Reports

If you experience bugs or general issues with *TemporAI*, please have a look
on the [issue tracker].
If you don't see anything useful there, please feel free to fire an issue report.

> Please don't forget to include the closed issues in your search.
Sometimes a solution was already reported, and the problem is considered
**solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.

## Documentation Improvements

You can help improve *TemporAI* docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

*TemporAI* documentation uses [Sphinx] as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.

We use [CommonMark] as the markup language for the documentation, facilitated by the [MyST] extension. This also allows the use of [reStructuredText] where necessary.

> Please notice that the [GitHub web interface] provides a quick way of
propose changes in *TemporAI*'s files. While this mechanism can
be tricky for normal code contributions, it works perfectly fine for
contributing to the docs, and can be quite handy.
>
> If you are interested in trying this method out, please navigate to
the `docs` folder in the source [repository], find which file you
would like to propose changes and click in the little pencil icon at the
top, to open [GitHub's code editor]. Once you finish editing the file,
please write a message in the form at the bottom of the page describing
which changes have you made and what are the motivations behind them and
submit your proposal.

When working on documentation changes in your local machine, you can
compile them using [tox] :

```sh
tox -e docs
```

and use Python's built-in web server for a preview in your web browser
(`http://localhost:8000`):

```sh
python3 -m http.server --directory 'docs/_build/html'
```

## Code Contributions

`TBC: Developer Guide`

### Submit an issue

Before you work on any non-trivial code contribution it's best to first create
a report in the [issue tracker] to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

### Create an environment

Before you start coding, we recommend creating an isolated [virtual environment]
to avoid any problems with your installed Python packages.
This can easily be done via either [virtualenv]:

```sh
virtualenv <PATH TO VENV>
source <PATH TO VENV>/bin/activate
```

or [Miniconda]:

```sh
conda create -n temporai python=3.8
conda activate temporai
```

### Clone the repository

1. Create an user account on GitHub if you do not already have one.

1. Fork the project [repository]: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on GitHub.

1. Clone this copy to your local disk:

   ```sh
   git clone git@github.com:YourLogin/temporai.git
   cd temporai
   ```

1. Run the following to install the package and all requirements, including dev requirements:

   ```sh
   pip install -U pip         # Update pip.
   pip install -r prereq.txt  # Install prerequisite dependencies.
   pip install -e .[dev]      # Install TemporAI in editable mode, with the `dev` extra.
   ```

1. Install [pre-commit]:

   ```sh
   pip install pre-commit
   pre-commit install
   ```

   *TemporAI* comes with a lot of [pre-commit hooks](./..pre-commit-config.yaml) configured to automatically help the
   developer to check the code being written.

### Implement your changes

1. Create a branch to hold your changes:

   ```sh
   git checkout -b my-feature
   ```

   and start making changes. Never work on the main branch!

1. Start your work on this branch. Don't forget to add [docstrings] to new
   functions, modules and classes, especially if they are part of public APIs.

1. Add yourself to the list of contributors in `AUTHORS.md`.

1. When you’re done editing, do:

   ```sh
   git add <MODIFIED FILES>
   git commit
   ```

   to record your changes in [git].

   Please make sure to see the validation messages from [pre-commit] and fix
   any eventual issues.
   This should, among other things, automatically use [flake8]/[black] to check/fix the code style
   in a way that is compatible with the project.
   See also the full list of the configured [pre-commit hooks](./..pre-commit-config.yaml).

   **Don't forget to add unit tests and documentation in case your
   contribution adds an additional feature and is not just a bugfix.**

   Moreover, writing a [descriptive commit message] is highly recommended.
   In case of doubt, you can check the commit history with:

   ```sh
   git log --graph --decorate --pretty=oneline --abbrev-commit --all
   ```

   to look for recurring communication patterns.

1. Please locally check that your changes don't break any tests. The same set of tests will be executed via [GitHub workflows] when you submit your PR.

   ```sh
   # Run unit tests on your current environment.
   pytest

   # Run unit tests on different tox python environments (this may be slow).
   # The tests will also include doctests, and a check that all notebooks execute without errors.
   tox -r
   ```

   (after having installed [tox] with `pip install tox`).

   You can also use [tox] to run several other pre-configured tasks in the
   repository. Try `tox -av` to see a list of the available checks.

1. It is also useful to check that documentation generation succeeds after your changes, run this with the following command, and make sure you do not see any [Sphinx] `WARNING`s or errors.
   ```sh
   tox -r -e docs
   ```

### Submit your contribution

1. Before submitting your contribution, make sure to read our [code of conduct].

1. If everything works fine, push your local branch to the remote server with:

   ```sh
   git push -u origin my-feature
   ```

1. Go to the web page of your fork and click "Create pull request"
   to send your changes for review.

   Find more detailed information in [creating a PR]. You might also want to open
   the PR as a draft first and mark it as ready for review after the feedbacks
   from the continuous integration (CI) system or any required fixes.

   The [PR template] will guide you through the steps of preparing your PR.

1. When your PR is submitted, some automated [GitHub workflows] will run to double check passing of tests, linting, etc.
One of our maintainers will review your PR and help if any of the checks are failing.

### Troubleshooting

The following tips can be used when facing problems to build or test the
package:

1. Make sure to fetch all the tags from the upstream [repository].
   The command `git describe --abbrev=0 --tags` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   `.eggs`, as well as the `*.egg-info` folders in the `src` folder or
   potentially in the root of your project.

1. Sometimes [tox] misses out when new dependencies are added, especially to
   `setup.cfg` and `docs/requirements.txt`. If you find any problems with
   missing dependencies when running a command with [tox], try to recreate the
   `tox` environment using the `-r` flag. For example, instead of:

   ```sh
   tox -e docs
   ```

   Try running:

   ```sh
   tox -r -e docs
   ```

1. Make sure to have a reliable [tox] installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run:

   ```sh
   tox --version
   # OR
   which tox
   ```

   If you have trouble and are seeing weird errors upon running [tox], you can
   also try to create a dedicated [virtual environment] with a [tox] binary
   freshly installed. For example:

   ```sh
   virtualenv .venv
   source .venv/bin/activate
   .venv/bin/pip install tox
   .venv/bin/tox -e all
   ```

1. [Pytest can drop you] in an interactive session in the case an error occurs.
   In order to do that you need to pass a `--pdb` option (for example by
   running `tox -- -k <NAME OF THE FALLING TEST> --pdb`).
   You can also setup breakpoints manually instead of using the `--pdb` option.



## Maintainer tasks

### Releases

We now use [GitHub workflows] for releases, see [`release.yml`](.github/workflows/release.yml).



[black]: https://pypi.org/project/black/
[code of conduct]: https://github.com/vanderschaarlab/.github/blob/main/CODE_OF_CONDUCT.md
[commonmark]: https://commonmark.org/
[contribution-guide.org]: http://www.contribution-guide.org/
[creating a pr]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
[descriptive commit message]: https://chris.beams.io/posts/git-commit
[docstrings]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[first-contributions tutorial]: https://github.com/firstcontributions/first-contributions
[flake8]: https://flake8.pycqa.org/en/stable/
[git]: https://git-scm.com
[github web interface]: https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository
[GitHub workflows]: https://github.com/vanderschaarlab/temporai/tree/main/.github/workflows
[github's code editor]: https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository
[github's fork and pull request workflow]: https://guides.github.com/activities/forking/
[guide created by freecodecamp]: https://github.com/freecodecamp/how-to-contribute-to-open-source
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[myst]: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
[PR template]: https://github.com/vanderschaarlab/.github/blob/main/.github/pull_request_template.md
[other kinds of contributions]: https://opensource.guide/how-to-contribute
[pre-commit]: https://pre-commit.com/
[pypi]: https://pypi.org/
[PyScaffold contributor's guide template]: https://github.com/pyscaffold/pyscaffold/blob/835eb4f986e37409d33fdb3f4d150e41ee07a111/src/pyscaffold/templates/contributing.template
[pyscaffold]: https://pyscaffold.org/
[pytest can drop you]: https://docs.pytest.org/en/stable/usage.html#dropping-to-pdb-python-debugger-at-the-start-of-a-test
[python software foundation's code of conduct]: https://www.python.org/psf/conduct/
[restructuredtext]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
[sphinx]: https://www.sphinx-doc.org/en/master/
[tox]: https://tox.readthedocs.io/en/stable/
[virtual environment]: https://realpython.com/python-virtual-environments-a-primer/
[virtualenv]: https://virtualenv.pypa.io/en/stable/


[repository]: https://github.com/vanderschaarlab/temporai
[issue tracker]: https://github.com/vanderschaarlab/temporai/issues
