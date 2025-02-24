This is the code for obtaining the results of the paper *Second Order Shape Optimization for an Interface Identification Problem Constrained by Nonlocal Models* by M. Schuster and V. Schulz that can be found on https://arxiv.org/abs/2406.09118.

Build and Install on Ubuntu
===========================
In order to clone the project do
::
  git clone https://github.com/schustermatthias/nlschwarz.git path/to/local_folder

| Since this code contains a customized version of **nlfem** the following **basic requirements** of nlfem are needed
| ``gcc, g++, python3-dev, python3-venv, libgmp-dev, libcgal-dev, metis, libmetis-dev, libarmadillo-dev``.
On Ubuntu, this can be done via
::
  sudo apt-get install git gcc g++ libarmadillo-dev liblapack-dev libmetis-dev
  sudo apt-get install python3-venv python3-dev libgmp-dev libcgal-dev

| See https://gitlab.uni-trier.de/pde-opt/nonlocal-models/nlfem for more information.
| Moreover, to run nlshape **legacy FEniCS(version 2019.1.0)** is required. In order to use FEniCS in a virtual environment, it may has to be installed globally and then inherited as a global site package. 
A virtual environment can be built and activated via
::
  mkdir venv
  python3 -m venv venv/
  source venv/bin/activate

Additionally, the packages from the file **requirements.txt** are neccessary and can be installed by
::
  (venv) python3 -m pip install -r requirements.txt

The creation of the virtual environment and the installation of packages from requirements.txt can probably also be done via your IDE.
Finally, nlfem can be installed by
::
  (venv) python3 setup.py build --force install
  
Running the Examples from the Paper
===================================
To run the first example of the paper, just execute main.py. In order to run the second experiment, uncomment the two lines in main.py that begin with "problem_2...", i.e., lines 10 and 11.
Moreover, to change the starting shape, go to conf_fractional.py (for the fractional kernel) or to conf_int.py (for the integrable kernel) and adjust the line beginning with " 'init_shape:' " 
to " 'init_shape': 'square', " (for the starting interface of the first example) or to " 'init_shape': 'spline', " (for the starting shape of the second experiment). 
Other parameters can also be altered in those two configuration files. 

Raw Data
========
The data of the experiments in the paper can be found in the folder "nlSecondShape/results".

License
=======
nlSecondShape is published under GNU General Public License version 3. Copyright (c) 2025 Matthias Schuster

| Parts of the project are taken from **nlfem** and have been customized.
| nlfem is published under GNU General Public License version 3. Copyright (c) 2021 Manuel Klar, Christian Vollmann
  
