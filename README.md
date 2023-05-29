# CS 546 - GSV Building Identification Based on 2D Maps

This code is built upon the official repo for *Neurocomputing 2022* paper **GSV-Cities: Toward Appropriate Supervised Visual Place Recognition**

[[ArXiv](https://arxiv.org/abs/2210.10239)]

## Datasets
* The GSV dataset can be dowloaded from [[Dataset](https://www.kaggle.com/datasets/amaralibey/gsv-cities)]. Make a folder corresponding to each city inside `datasets/gsv-cities/images/City.`
* Upload the metada for each city in `datasets/gsv-cities/Dataframes.`
* The Nordland Dataset, SPED, and Pittsburg datasets can be downloaded from [[here]https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W]. 
* Training can be run from `main.py`. The code has comments for using datasets for different cities, different aggregators and benchmark datasets.
* Dataset zip files are uploaded in their respective folders. Unzip them inside those folder to run training and evaluation.
* The files for different aggregators are present in models/aggregators. 
---
