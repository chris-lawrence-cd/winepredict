import os

# Create docs directory if it doesn't exist
if not os.path.exists('docs'):
    os.makedirs('docs')

# Create a minimal conf.py
conf_content = """
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'WinePredict'
copyright = '2023, Chris Lawrence'
author = 'Chris Lawrence'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
"""

with open("docs/conf.py", "w") as f:
    f.write(conf_content)

# Create a minimal index.rst
index_content = """
Welcome to WinePredict's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

with open("docs/index.rst", "w") as f:
    f.write(index_content)

# Create a Makefile
makefile_content = """
# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
"""

with open("docs/Makefile", "w") as f:
    f.write(makefile_content)

# Generate API documentation
!sphinx-apidoc -o docs/ winepredict

# Build HTML documentation
print("Building HTML documentation...")
!make -C docs html

# Build PDF documentation
print("\nBuilding PDF documentation...")
!make -C docs latexpdf

# Check if the documentation was generated successfully
if os.path.exists('docs/_build/html/index.html'):
    print("\nHTML documentation generated successfully.")
else:
    print("\nError: HTML documentation not generated.")

if os.path.exists('docs/_build/latex/winepredict.pdf'):
    print("PDF documentation generated successfully.")
else:
    print("Error: PDF documentation not generated.")

# Print the contents of the docs directory
print("\nContents of the docs directory:")
for root, dirs, files in os.walk('docs'):
    level = root.replace('docs', '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 4 * (level + 1)
    for file in files:
        print(f"{sub_indent}{file}")
