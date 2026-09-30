"""Sphinx configuration for the documentation site."""

import importlib.metadata

project = "clifford-noise-reduction"
author = "Daminark"
copyright = f"2026, {author}"
release = importlib.metadata.version(project)

extensions = [
    "myst_nb",  # MyST Markdown pages and notebooks, executed at build time
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "CliNR"

# Notebooks run on every build, and any cell error fails it; the cache
# skips notebooks whose code has not changed since the last run.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_show_tb = True  # print the failing cell's traceback in the log

myst_enable_extensions = ["dollarmath"]

autodoc_member_order = "bysource"
numpydoc_show_class_members = False

# Stim publishes no Sphinx inventory, and its classes report a private,
# CPU-specific module (stim._stim_sse2, stim._stim_avx2), so their names
# render unlinked.
nitpick_ignore_regex = [("py:class", r"stim\._stim_\w+\.\w+")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
