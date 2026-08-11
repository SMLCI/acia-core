.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/SMLCI/acia-core/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

AutomatedCellularImageAnalysis could always use more documentation, whether as part of the
official AutomatedCellularImageAnalysis docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/SMLCI/acia-core/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `acia` for local development.

1. Fork the `acia` repo on GitHub (``SMLCI/acia-core``).
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/acia-core.git

3. Install your local copy in editable mode with the dev extras::

    $ cd acia-core/
    $ pip install -e ".[dev]"

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done, check that your changes pass linting, formatting and the
   tests (the same checks CI runs)::

    $ ruff check acia tests
    $ ruff format --check acia tests
    $ pytest                      # fast suite; integration tests are deselected

   Heavy integration tests (external data, trackastra/laptrack, ffmpeg) are
   marked ``@pytest.mark.integration`` and run separately::

    $ pytest -m integration

6. Add a short entry under ``## [Unreleased]`` in ``CHANGELOG.md`` describing
   your change.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md and an entry under ``[Unreleased]`` in
   ``CHANGELOG.md``.
3. The pull request should work for Python 3.10, 3.11, 3.12 and 3.13. The CI
   workflow (``.github/workflows/ci.yml``) runs lint, type-checks and the test
   matrix on every push and pull request.

Tips
----

To run a subset of tests::

$ pytest tests.test_acia


Releasing
---------

Releases are automated. A maintainer triggers them from GitHub:

1. Make sure ``CHANGELOG.md`` has the changes for this release under
   ``## [Unreleased]``.
2. Go to **Actions → Release → Run workflow** and choose the bump level
   (``patch`` / ``minor`` / ``major``). Use ``dry_run`` first to preview.

The ``release.yml`` workflow then, in one run:

* runs ``bump-my-version`` to update ``pyproject.toml`` and ``acia/__init__.py``,
  promote the ``[Unreleased]`` changelog section to a dated version, and create
  the ``vX.Y.Z`` commit and tag,
* builds the sdist + wheel and publishes them to PyPI via **OIDC trusted
  publishing** (no stored token), and
* creates a GitHub Release using the new changelog section as the notes.

Documentation is published to GitHub Pages automatically on every push to the
default branch via ``docs.yml``.

Travis will then deploy to PyPI if tests pass.
