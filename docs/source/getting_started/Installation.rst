Installation
============

To get started using opennyai first create a new python virtual environment:

.. code-block::
    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate

Install it using pip by running the following line in your terminal

.. code-block::
    pip install opennyai

Note: if you want to utilize spacy with GPU please install [Cupy](https://anaconda.org/conda-forge/cupy)
/[cudatoolkit](https://anaconda.org/anaconda/cudatoolkit) dependency of appropriate version. For spacy installation with
cupy refer to [page](https://spacy.io/usage)

Remember you need spacy of 3.2.4 version for models to work perfectly.