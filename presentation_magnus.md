---
marp: true
title: AI in industry
description: Project presentation
theme: gaia
paginate: true
_paginate: false
---

<style>
section {
  background: black;
  color: white;
  font-size: 26px;
  padding: 40px;
}

h1 {
  color: orange;
  text-align: center 
}

h2 {
  color: orange;
  text-align: center 
}

strong {
  font-weight: bold;
  color: orange;
}
  
code {
  color: orange;
  background: black;
}
</style>

# Project Presentation: Arctic Sea Ice Challenge by ESA

## Magnus Ostertag, Antonius Scherer
## AI in Industry, Prof. Michele Lombardi
## Winter Term 2022/2023

---

## Content

1. challenge overview (Magnus)
2. understanding the data (Magnus)
3. Our Code (Toni)
4. u-net model improvements (Magnus)
5. results (Magnus)
6. u-net model attention (Toni)
7. results (Toni)
8. summary (Toni)
9. lessons learned (and possible improvements) (Toni)

---

## Challenge Overview

see `introduction.ipynb`

* goal: automatically produce sea ice charts
* Sea Ice Concentration (SIC), the Stage Of Development (SOD), and the floe size (FLOE) to be predicted
* 493 training and 20 test (without label data) data files

Sea ice parameter | Metric | Weight in total score
---|---|---
Sea ice concentration | R2 | 2/5
Stage of development | F1 | 2/5
Floe size | F1 | 1/5

---

### Geographic Coverage

![Geographic coverage of the scenes](presentation_pics/coverage_fig1_manual.png)

see the data manual.

---

## Challenge Overview continued

* main data: Sentinel-1 active microwave (SAR) data and corresponding Microwave Radiometer (MWR) data from the AMSR2 satellite sensor
  * SAR data has ambiguities, it has a high spatial resolution
  * MWR data has good contrast between open water and ice
* auxiliary data: numerical weather prediction model data, incidence angle of the SAR sensor and distance from land

---

### Exemplary Scene

![Example polygone of a scene, sea ice concentration](presentation_pics/polygon_icechart_fig11_manual.png)

Of sea ice concentration, see the data manual.

---

## Understanding the Data

see `distributions.ipynb`

* train vs. test vs. validation distribution
* coverage of the scenes
* (inter-analyst accuracy)

---

## Our Code

A tour through the repository.

---

## U-Net Model Improvements

`*_improvements.py` (and earlier `*_transfer.py`)

* regression loss when using a regression metric `unet*.py`, `quickstart*.ipynb`
* transfer learning `unet_transfer.py`
* biased sampling of the training data: `loaders_improvements.py`
* diagnostics output `quickstart_improvements.ipynb`, `loaders_improvements.ipynb`

---

## Statistics of the Runs

TODO

---

## Error Distributions

TODO

* understand in which conditions the predictions are the worst
* resample

---

## Summary

* u-net is a good model for the task
* a larger size of the convolution filters is not that important for the performance
* attention performs very well even for small number of epochs (?)
* the scores were not as high as we expected, but the problem was made harder and the computational resources were lacking

---

## Lessons Learned

* RAM problems with the very large image data and python
* use the home GPU
* adding code to a complex code base
* using remote computing and `mlflow`

### and possible improvements

* featurize additional information like ice chart provider, month (?)
* play with the learning rate
