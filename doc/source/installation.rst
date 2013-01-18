Installation
============

Dolo can be installed using the standard python command ``pip``.
You can choose between one of the three following options.

- To install last version from `PyPI` directories:

.. code-block:: bash

    `pip install dolo`

- To install last version from github (requires git on your computer):

.. code-block:: bash

    pip install git+https://github.com/albop/dolo.git

- To install `dolo` from your own clone of the dolo repository. Type from this directory:

.. code-block:: bash

    pip install -e .

The option -e tells the installer to make symlinks so that the installed version is automatically updated when a new
version is pulled.


Compiled extensions
-------------------

 In case, you have used the last option you can also build the ``cython`` extension with:

.. code-block:: bash

    python setup_cython.py --build-ext --inplace

Eventually, this will be enabled by default.