# Heart imagery abnormalities classifier
## Quick overhaul
This project's objective is to create a highly accurate classifier for detecting abnormalities in heart imagery
from patterns created from Single Proton Emission Computed Tomography (SPECT) images. My personal goal with this project is to learn how to properly structure ML projects, build proper pipelines and how to use optuna, streamlit and MLflow.


## Dataset
Dataset comes from https://archive.ics.uci.edu/dataset/96/spectf+heart, It contains 267 instances grouped into two categories: normal and abnormal.
The data consists of 44 independent, continuous variables ranging from 0 to 100 and a binary variable representing abnormality, where 0 = normal and 1 = abnormal. There are 55 normal instances and 212 abnormal instances.
There wasn't any missing data and files could be interpreted as csv's. The data had been originally split into 187 test instances and 80 train instances,
yet I've decided to use a train/test split of 70/30 to achieve a more accurately trained model. To offset this difference in the test set, I've decided to use the Stratified
K- fold cross validation algorithm.


## Structure 
data extraction -> data cleaning -> train test split -> feature scaling -> feature elimination -> classification -> kfold to choose model -> model to choose hyperparameters -> deployment


## Model Selection
I've decided to use 100 optuna trials per model to choose the best performing one, the goal was to maximize f1 score.
I've used f1 score rather than accuracy because the classes are highly imbalanced, it ensures that the model won't try to classify every instance it's not entirely sure about as abnormal. 

| Model                    | Best F1 score | Standard Deviation |
|--------------------------|---------------| -------------------|
| Logistic regression      | 0.87          | 0.04 |
| Random Forest Classifier | 0.88          | 0.05 |
| GaussianNB               | 0.8           | 0.08 |
| XGB Classifier           | 0.89          | 0.04 |
| SVC                      | 0.89          | 0.02 |

I've also created an ANN based on the Keras framework, i've used 100 trial optuna tuning
to train 100-epoch models, the best accuracy I've achieved was 84%.
Such accuracy in the most optimistic scenario would amount to an F1 score of 0.91.
Thus I've decided to use SVC for the final model.

## File structure

## How to run the project

## wnioski i interpretacja po ang xd