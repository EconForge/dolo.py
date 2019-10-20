Getting started
===============

Installation
------------

A scientific Python environement is required to run ``dolo``, for instance Anaconda Python.

In order to install the last stable version of ``dolo`` and its dependencies, open a command-line and run:

.. code-block:: bash

    `pip install dolo`

It is also possible to install the development version directly from Github with:

.. code-block:: bash

    `pip install git+git://github.com/econforge/dolo.git`


Step-by-step instructions on windows
++++++++++++++++++++++++++++++++++++

    - Download the `Anaconda installer <http://continuum.io/downloads>`_ (choose the 64 bits/python 2.7 version)

    .. .. figure::
    ..
    ..     .. image:: .//figs//anaconda_install_1.png
    ..         :width: 50%
    ..
    ..     .. image:: .//figs//anaconda_install_2.png
    ..         :width: 50%
    ..
    ..
    ..     Download page for Anaconda

    - Install it for the current user only, so that you will be able to install to update Python packages easily.

    .. figure:: .//figs//anaconda_install_2.png
        :width: 80%

        Anaconda's installer

    - Open a powershell console, and type ``pip install dolo`` then Enter. When connected to the net, this command pulls and install the last stable version

    .. figure:: .//figs//install_dolo_2.png
        :width: 80%

        Dolo install command


Running dolo
------------

After dolo is installed, try to solve a model by typing the following commands in an IPython shell:

.. code:: python

    from dolo import *                           # load the library
    model = yaml_import("https://raw.githubusercontent.com/EconForge/dolo/master/examples/models/rbc.yaml")
                                                 # import the model
    display(model)                               # display the model
    dr = time_iteration(model, verbose=True)     # solve
    sim = simulate(model, dr)                    # simulate

Setting up a work environement
------------------------------


Anylising dolo models, is usually done by editing a model file with an (``.yaml``) extension, then running and formating the calculations inside a Jupyter notebook. There are many other worflows, but Jupyter notebooks are becoming a de facto standard in opensource computational research, so that we strongly advise to try them first. Some chapters of this documentation are actually written as notebook, can be downloaded and run interactively.

The only step to setup the environment consists in choosing a folder to store the model and the notebooks. Then open a terminal in this folder and launch the notebook server using:

.. code::

    `jupyter notebook`


.. figure:: .//figs//open_command_prompt.png
        :width: 80%

        Open shell under windows in a given folder

A browser window should popup. It is Jupyter's dashboard.


.. figure:: .//figs//jupyter_dashboard.png
        :width: 80%

        Jupyter's dashboard



It lists the files in that folder. Clicking on a model file (with a ``.yaml`` extension), opens it in a new tab.

.. figure:: .//figs//text_editor.png
        :width: 80%

        Jupyter's text editor


.. note::

    Despite the fact that the files are edited in the browser through a local webserver, the files are still regular files on the harddrive. In particular, it is possible to edit them directly, using a good text editor (vim, emacs, atom...)

To create a new notebook click on ".." and choose IPython. This creates a new tab, containing the notebook ready to be edited and run. It consists in a succession of cells that can be run in any order by pressing Shift+Enter after one of them has been selected. More information [TODO: link]

.. figure:: .//figs//notebook.png
        :width: 80%

        Jupyter notebook
