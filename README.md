# CS 546 - GSV Building Identification Based on 2D Maps

This code is built upon the official repo for *Neurocomputing 2022* paper **GSV-Cities: Toward Appropriate Supervised Visual Place Recognition**

[[ArXiv](https://arxiv.org/abs/2210.10239)]

## Datasets
* The GSV dataset can be dowloaded from [Dataset](https://www.kaggle.com/datasets/amaralibey/gsv-cities). Make a folder corresponding to each city inside `datasets/gsv-cities/images/City.`
* Upload the metada for each city in `datasets/gsv-cities/Dataframes.`
* The Nordland Dataset, SPED, and Pittsburg datasets can be downloaded from [here](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W). 
* The MSLS data set can be downloaded from the here [Mapillary Street-level Sequences Dataset](https://www.mapillary.com/dataset/places).
* Unzip the dataset files into their corresponding folders inside `datasets`.

## Training and Validation
* Training can be run from `main.py`. The code has comments for using datasets for different cities, different aggregators and benchmark datasets.
* The files for different aggregators are present in `models/aggregators`. The aggregator fields and configurations are commented and can be changed in `VPRmodel`. 
* The validation data set can be changed through `GSVCitiesDataModule(val_set_names = [...])`.
* The `fast_dev_run` field can be used before training to check if the code is in order.

The code to run the model from some previous checkpoint is as follows 

```python
from main import VPRModel

model = VPRModel(backbone_arch='resnet50', 
                 layers_to_crop=[],
                 agg_arch='...',
                 agg_config={...},
                )

state_dict = torch.load('./LOGS/resnet50...')
model.load_state_dict(state_dict)
model.eval()

```
---

## 
