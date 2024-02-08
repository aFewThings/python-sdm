This repository is based on https://gitlab.inria.fr/bdeneu/cnn-sdm.

Please read the original README.md for more details.

I modified some parts of the code to make it work with my purpose.

Major updates are:
- Adding traditional SDMs (statistical models, and machine learning models) and deep neural networks (DNNs)
- Adding Benchmark dataset (https://journals.ku.edu/jbi/article/view/13384) and custom dataset (environmental data of South Korea, and species occurrence data from GBIF)
  - https://osf.io/kwc4v/
  - https://www.worldclim.org/data/worldclim21.html
  - http://due.esrin.esa.int/page_globcover.php
  - https://www.gbif.org/
- Random sampling pseudo-absences using exclusion buffers based on environmental rasters
- Training and testing SDMs with presence-only (PO) and presence-absence (PA) datasets on a conventional binary classification task.

Added codes for the conventional species distribution modeling:
- deep_sdm.py
- train_traditional_sdm.py
- train_traditional_sdm_benchmark.py

If you benefit from this repository, please cite these papers:
- base repos.: ```Convolutional neural networks improve species distribution modelling by capturing the spatial structure of the environment```
- benchmark dataset: ```Presence-only and presence-absence data for comparing species distribution modeling methods```
- ```Predicting invasive species distributions using incremental ensemble-based pseudo-labeling```
