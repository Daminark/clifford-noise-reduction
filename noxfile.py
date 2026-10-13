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

import os

import nox

nox.needs_version = ">=2026.8.17"
nox.options.default_venv_backend = "none"

PRE_COMMIT = "pre-commit@4.6.2"


def uv_run(session: nox.Session, *args: str, **kwargs: object) -> None:
    """Run ``args`` under ``uv run --locked``, on the Python of ``UV_PYTHON``.

    nox removes ``UV_PYTHON`` from the commands it runs, so it is passed on
    as ``--python``; without it, uv takes the version in ``.python-version``.
    """
    python = os.environ.get("UV_PYTHON")
    chosen = ["--python", python] if python else []
    session.run(
        "uv", "run", "--locked", *chosen, *args, external=True, **kwargs
    )


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
    uv_run(session, "mypy", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the fast suite: slow tests skipped, 20 Hypothesis examples."""
    uv_run(session, "pytest", *session.posargs)


@nox.session(default=False)
def tests_full(session: nox.Session) -> None:
    """Run every test, 200 Hypothesis examples, with coverage."""
    uv_run(
        session,
        "pytest",
        "--runslow",
        "--cov",
        "--cov-report=term-missing",
        *session.posargs,
        env={"HYPOTHESIS_PROFILE": "ci"},
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Build the documentation site, failing on any warning.

    Notebooks whose code has changed run as part of the build. The site is
    written to ``docs/_build/html``.
    """
    uv_run(
        session,
        "sphinx-build",
        "--nitpicky",  # warn on every reference that does not resolve
        "--fail-on-warning",
        "--builder=html",
        "docs",
        "docs/_build/html",
        *session.posargs,
    )


if __name__ == "__main__":
    nox.main()
