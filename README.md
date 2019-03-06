## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Usage](#usage)
* [Status](#status)
* [References](#references)
* [Issues](#issues)


## General info
poreMKS is a tool for the analytics of porous molecular structures.

The Molecular Structure of DDR: ![DDR Structure](./images/DDR_structure.gif)

The pore region of DDR embedded within the molecular structure. ![All Admissible Pore Structure](./images/DDR_pore_all.gif)

The pore region of DDR accessible by a 1.5A radius probe is visualized as below: ![Accessible Structure](./images/DDR_pore_cleaned.gif)

The skeleton of the pore structure overlayed on the pore volume is visualized as below: ![Accessible Path Structure](./images/DDR_skeleton_pore.gif)

The shortest paths for the probe through the pore structure are visualized as below: ![shortest paths](./images/DDR_graph.gif)


## Technologies
Project is created with:
* scikit-image version: ---
* scipy version: ---
* numpy version: ---
* numba version: ---
* pytorch version: ---
* pytoolz version: ---


## Setup
To run this project, install it locally using conda:

```
$ cd ../<project_directory>
$ conda create -n poremks python=3.6
$ conda activate
$ pip install edt
$ conda install numba
$ conda install pytorch-cpu torchvision-cpu -c pytorch
```
Conda is a package and environment manager bundled with anaconda python distribution.
See, [https://www.continuum.io/downloads](https://www.continuum.io/downloads) for more details on installing and using Conda.  

**Windows Users** need to install visual studio build tools, in order to the compile c/c++ files assosciated with some of the dependencies.


## Usage
Refer to the [jupyter notebook](./scripts/tutorial_poreMKS.ipynb) in the doc folder.


## Status
poreMKS is currently under active development.


## References
[ReadMe](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Issues

Please send questions and issues about installation and usage of PyMKS to [apaar92@gmail.com](mailto:apaar92@gmail.com)
