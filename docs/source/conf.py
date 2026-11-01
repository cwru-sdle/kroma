# Sphinx configuration for KROMA.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _version

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "KROMA"
copyright = "2025, SDLE Research Center, Case Western Reserve University"
author = "Lam Nguyen, Erika Barcelos, Roger French, Yinghui Wu"
try:
    release = _version("kroma-om")
except PackageNotFoundError:
    from kroma import __version__ as release
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []
add_module_names = False

# The model backends are optional extras; mock them so the API reference
# builds with only the core dependencies installed.
autodoc_mock_imports = [
    "torch",
    "transformers",
    "sentence_transformers",
    "node2vec",
    "networkx",
    "openai",
    "together",
    "SPARQLWrapper",
    "nltk",
    "sklearn",
]

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rdflib": ("https://rdflib.readthedocs.io/en/stable/", None),
}

# Names in docstrings that come from mocked or untyped packages.
nitpick_ignore_regex = [
    (r"py:class", r"(networkx|gensim|torch)\..*"),
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"KROMA {release}"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
}
html_show_sourcelink = False
html_context = {
    "display_github": True,
    "github_user": "cwru-sdle",
    "github_repo": "kroma",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
