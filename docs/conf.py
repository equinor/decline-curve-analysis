# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AutoDCA"
copyright = "2024 - 2025, DSA, Equinor"
author = "DSA, Equinor"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# warn about all invalid references
nitpick_ignore = [
    ("py:class", "np.ndarray"),
    ("py:class", "np.array"),
    ("py:class", "callable"),
    ("py:class", "optional"),
    ("py:class", "None"),
]
nitpicky = False

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

# Configure napoleon
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Configure Sphinx-Gallery
# https://sphinx-gallery.github.io/stable/getting_started.html
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],  # path to your example scripts
    "gallery_dirs": "examples_gallery/",  # path to where to save gallery generated output
    "doc_module": ("dca",),
    "ignore_pattern": r"sodir",
    "reset_modules": ("matplotlib",),
    "plot_gallery": "True",
    "run_stale_examples": True,
    "reference_url": {
        # The module you locally document uses None
        "dca": None,
    },
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "version_selector": False,
    "language_selector": False,
}
html_static_path = ["_static"]
