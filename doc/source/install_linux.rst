Install on linux
================

Here is the currently recommended way to install Dolo on linux:

Install the dependencies. On debian/ubuntu: 

sudo apt-get install python-setuptools python-yaml python-scipy python-sympy python-matplotlib

Go to the directory where you want the library to stay : 

cd ~/path-to-source

Get latest version from github: 

git clone https://albop@github.com/albop/dynare-python.git dynare-python

if it does not work just install git by:

apt-get install git

Go in the dynare-python/dolo/ subdirectory and type: 

python setup.py develop --prefix=~/.local/

If it is not working create the directories 

~/.local//lib/python2.7/site-packages/

If it is not done already, add this line to the ~/.bashrc file export PATH=~/.local/bin:$PATH

With this set of commands, the installed version of dolo will always be the one being in the git repository, so that it you pull (git pull) the last revision or if you make modifications, it will affect the installed version.