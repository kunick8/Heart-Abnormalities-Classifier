# Heart imagery abnormalities classifier
## Quick overhaul
This project's objective is to create a highly accurate classifier for detecting abnormalities in heart imagery
from patterns created from Single Proton Emission Computed Tomography (SPECT) images. 


## Dataset
Dataset comes from https://archive.ics.uci.edu/dataset/96/spectf+heart, It contains 267 instances grouped into two categories: normal and abnormal.
The data consists of 44 independent, continuous variables ranging from 0 to 100 and a binary variable representing abnormality, where 0 = normal and 1 = abnormal. There are 55 normal instances and 212 abnormal instances.
There wasn't any missing data and files could be interpreted as csv's. The data had been originally split into 187 test instances and 80 train instances,
yet I've decided to use a train/test split of 70/30 to achieve a more accurately trained model. To offset this difference in the test set, I've decided to use the Stratified
K- fold cross validation algorithm.


## Structure 
data preprocessing -> train test split -> feature scailing -> feature elimination -> classification -> kfold to choose model -> model to choose hyperparameters -> deployment


## Model Selection
I've decided to use 100 optuna trials per model to choose the best performing one.

| Model                    | Best F1 score |
|--------------------------|---------------|
| Logistic regression      | 0.86          |
| Random Forest Classifier | 0.9           | 
| GaussianNB               | 0.84          |
| XGB Classifier           | 0.88          |
| SVC                      | 0.92          |