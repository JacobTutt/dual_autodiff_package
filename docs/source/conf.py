# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../tutorials'))

project = 'dual_autodiff'
copyright = '2024, Jacob Tutt'
author = 'Jacob Tutt'

from dual_autodiff.version import __version__ as release
version = ".".join(release.split(".")[:2])
rst_prolog = f"""
.. |version| replace:: {version}
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'myst_parser', 
]
myst_enable_extensions = [
    "dollarmath",  # For inline math using $...$
    "amsmath",     # For multiline math environments
]
autodoc_member_order = 'bysource'
templates_path = ['_templates']
exclude_patterns = []
nbsphinx_execute = 'never'  
nbsphinx_output_prompt = '[Out]'
nbsphinx_input_prompt = '[In]'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']