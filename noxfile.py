#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2026.8.17"]
# ///
"""The project's commands, run the same way locally and in CI.

Every session runs in the environment that ``uv.lock`` pins, through
``uv run --locked``, so nox names the commands and uv provides the
environment. ``nox`` alone runs the sessions CI runs on every push;
``nox -s tests_full`` runs the full suite, as on pull requests.
"""

import nox

nox.needs_version = ">=2026.8.17"
nox.options.default_venv_backend = "none"

PRE_COMMIT = "pre-commit@4.6.2"


@nox.session
def lint(session: nox.Session) -> None:
    """Run the pre-commit hooks on every file."""
    session.run(
        "uvx",
        PRE_COMMIT,
        "run",
        "--all-files",
        "--show-diff-on-failure",
        external=True,
    )


@nox.session
def typecheck(session: nox.Session) -> None:
    """Type-check the package and the tests with mypy."""
    session.run(
        "uv", "run", "--locked", "mypy", *session.posargs, external=True
    )


@nox.session
def tests(session: nox.Session) -> None:
    """Run the fast suite: slow tests skipped, 20 Hypothesis examples."""
    session.run(
        "uv", "run", "--locked", "pytest", *session.posargs, external=True
    )


@nox.session(default=False)
def tests_full(session: nox.Session) -> None:
    """Run every test, 200 Hypothesis examples, with coverage."""
    session.run(
        "uv",
        "run",
        "--locked",
        "pytest",
        "--runslow",
        "--cov",
        "--cov-report=term-missing",
        *session.posargs,
        env={"HYPOTHESIS_PROFILE": "ci"},
        external=True,
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Build the documentation site, failing on any warning.

    Notebooks run as part of the build. The site is written to
    docs/_build/html.
    """
    session.run(
        "uv",
        "run",
        "--locked",
        "sphinx-build",
        "--nitpicky",  # warn on every reference that does not resolve
        "--fail-on-warning",
        "--builder=html",
        "docs",
        "docs/_build/html",
        *session.posargs,
        external=True,
    )


if __name__ == "__main__":
    nox.main()
