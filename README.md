## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Status](#status)
* [References](#references)

## General info
poreMKS is a tool for the analytics of porous molecular structures.

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

## Status
poreMKS is currently under active development.

## Usage
Refer to the [jupyter notebook](./scripts/tutorial_poreMKS.ipynb) in the doc folder.


## References
[ReadMe](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
## Issues

Please send questions and issues about installation and usage of PyMKS to [apaar92@gmail.com](mailto:apaar92@gmail.com)
