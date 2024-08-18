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

latex_engine = 'xelatex'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\\usepackage{fontspec}
\\setmainfont{DejaVu Serif}
\\setsansfont{DejaVu Sans}
\\setmonofont{DejaVu Sans Mono}
''',
}
"""

with open("docs/conf.py", "w") as f:
    f.write(conf_content)
Create the necessary RST files:
rst_files = {
    "index.rst": """
Welcome to WinePredict's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""",
    "introduction.rst": "Introduction\n============\n\nWinePredict is a library for predicting wine prices.",
    "installation.rst": "Installation\n============\n\nInstall WinePredict using pip:\n\n.. code-block:: bash\n\n    pip install git+https://github.com/chris-lawrence-cd/winepredict.git",
    "usage.rst": "Usage\n=====\n\nExample usage of WinePredict.",
    "api_reference.rst": """
API Reference
=============

.. toctree::
   :maxdepth: 2

   data_processing
   model_evaluation
   model_training
   visualization
""",
    "data_processing.rst": "Data Processing\n===============\n\n.. automodule:: winepredict.data_processing\n   :members:",
    "model_evaluation.rst": "Model Evaluation\n================\n\n.. automodule:: winepredict.model_evaluation\n   :members:",
    "model_training.rst": "Model Training\n==============\n\n.. automodule:: winepredict.model_training\n   :members:",
    "visualization.rst": "Visualization\n=============\n\n.. automodule:: winepredict.visualization\n   :members:",
}

for filename, content in rst_files.items():
    with open(f"docs/{filename}", "w") as f:
        f.write(content)

# Build the HTML documentation
!make -C docs html

# Build the PDF documentation
!make -C docs latexpdf
