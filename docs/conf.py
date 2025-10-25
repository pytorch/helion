# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import os
import sys
from typing import Callable
from typing import Protocol

import pytorch_sphinx_theme2  # pyright: ignore[reportMissingImports]

# -- Path setup --------------------------------------------------------------


class SphinxApp(Protocol):
    """Protocol for Sphinx application objects."""

    def connect(self, event: str, callback: Callable[..., None]) -> None:
        """Connect an event handler to a Sphinx event."""
        ...


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "Helion"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
    "pytorch_sphinx_theme2",
    "sphinxext.opengraph",
]

# MyST parser configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples",
    ],  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    "filename_pattern": r".*\.py$",  # Include all Python files
    "ignore_pattern": r"__init__\.py",  # Exclude __init__.py files
    "plot_gallery": "False",  # Don't run the examples
}

# Templates path
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]

html_theme_options = {
    "navigation_with_keys": False,
    "analytics_id": "GTM-NPLPKN5G",
    "canonical_url": "https://helionlang.com/",
    "show_lf_header": False,
    "show_lf_footer": False,
    "logo": {
        "text": "",
        "image_light": "_static/helion_nobackground.png",
        "image_dark": "_static/helion_nobackground.png",
    },
    "icon_links": [
        {
            "name": "X",
            "url": "https://x.com/PyTorch",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/pytorch/helion",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discourse",
            "url": "https://dev-discuss.pytorch.org/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/helion",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
}

html_favicon = "_static/helion_nobackground.png"

theme_variables = pytorch_sphinx_theme2.get_theme_variables()
templates_path = ["_templates"]
if pytorch_sphinx_theme2.__file__ is not None:
    templates_path.append(
        os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates")
    )

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "pytorch",
    "github_repo": "helion",
    "feedback_url": "https://github.com/pytorch/helion",
    "github_version": "main",
    "doc_path": "docs/",
    "library_links": [
        {
            "name": "Helion",
            "url": "https://pytorch.github.io/helion/",
            "current": True,
        },
        {
            "name": "torch.compiler",
            "url": "https://pytorch.org/docs/stable/torch.compiler.html",
        },
        {
            "name": "PyTorch",
            "url": "https://pytorch.org/docs/stable/",
        },
        {
            "name": "Triton",
            "url": "https://triton-lang.org/main/",
        },
    ],
}

html_sidebars = {
    "helion_puzzles": [],
    "examples/index": [],
    "installation": [],
}

html_static_path = ["_static"]

html_js_files = ["js/runllm-widget.js"]

# Output directory for HTML files
html_output_dir = "../site"

sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
]
sitemap_url_scheme = "{link}"

# Base URL for sitemap and canonical links
html_baseurl = "https://helionlang.com/"

# -- Options for autodoc extension ------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# autodoc-typehints configuration
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True


def remove_sphinx_gallery_content(
    app: SphinxApp, docname: str, source: list[str]
) -> None:
    """
    Remove sphinx-gallery generated content from the examples index.rst file.
    This runs after sphinx-gallery generates the file but before the site is built.
    """
    if docname == "examples/index":
        content = source[0]

        # Find the first toctree directive and remove everything after it
        lines = content.split("\n")
        new_lines = []
        found_toctree = False

        for line in lines:
            if line.strip().startswith(".. toctree::") and not found_toctree:
                found_toctree = True
                # Keep the line with the toctree directive
                new_lines.append(line)
                # Look for the next few lines that are part of the toctree options
                continue
            if found_toctree and (line.strip().startswith(":") or line.strip() == ""):
                # Keep toctree options and empty lines immediately after
                new_lines.append(line)
                continue
            if found_toctree:
                # We've hit content after the toctree options, stop here
                break
            # Keep everything before the toctree
            new_lines.append(line)

        # Update the source content
        source[0] = "\n".join(new_lines)


def setup(app: SphinxApp) -> dict[str, str]:
    """Setup function to register the event handler."""
    app.connect("source-read", remove_sphinx_gallery_content)
    return {"version": "0.1"}
