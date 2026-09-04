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
### Optuna Results


| Model                    | Best F1 score | Standard Deviation |
|--------------------------|---------------|--------------------|
| Logistic regression      | 0.87          | 0.04               |
| Random Forest Classifier | 0.88          | 0.05               |
| GaussianNB               | 0.8           | 0.08               |
| XGB Classifier           | 0.89          | 0.04               |
| SVC                      | 0.89          | 0.02               |

### Test Set Results
| Model                    | Best F1 score | Accuracy |
|--------------------------|---------------|----------|
| Logistic regression      | 0.82          | 0.74     |
| Random Forest Classifier | 0.85          | 0.75     |
| GaussianNB               | 0.8           | 0.74     |
| XGB Classifier           | 0.84          | 0.75     |
| SVC                      | 0.86          | 0.77     |
| ANN                      | 0.85          | 0.75     |

Based on these results I've decided to go with SVC for the final classifier



## File structure

## How to run the project

### Option 1 - run the local hosted site with charts, interpretations and predictions

#### Step 1 - install poetry and dependencies through the terminal
```
pip install poetry

```

#### Step 2 - launch the streamlit local host
```
streamlit run app/streamlit_app.py

```

All set, your browser should automatically open the local hosted site, if not then copy the url from the terminal

### Option 2 - train the final model and then run the local hosted site

#### Step 1 - install poetry and dependencies through the terminal
```
pip install poetry

```

#### Step 2 - setup mlflow server
```
mlflow server --host 127.0.0.1 --port 5000 
 
```

#### Step 3 - run main.py to train the model
```
python3 src.main.py                                                                             
```

#### Step 4 - launch the streamlit local host
```
streamlit run app/streamlit_app.py

```

All set, your browser should automatically open the local hosted site, if not then copy the url from the terminal

### Option 3 - run the whole evaluation and training pipeline

#### Step 1 - install poetry and dependencies through the terminal
```
pip install poetry

```

#### Step 2 - setup mlflow server
```
mlflow server --host 127.0.0.1 --port 5000 
 
```

#### step 3 run evaluation_pipeline.py
```
python3 src.evaluation_pipeline.py
```

#### step 4 check out the results on the mlflow site

open http:localhost:5000 in your browser

#### Step 5 - run main.py to train the model
```
python3 src.main.py                                                                             
```

#### Step 6 - launch the streamlit local host
```
streamlit run app/streamlit_app.py

```

All set, your browser should automatically open the local hosted site, if not then copy the url from the terminal


## wnioski i interpretacja po ang xd