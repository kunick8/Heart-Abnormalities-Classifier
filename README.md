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


| Model                    | Best F1 score(macro) | Standard Deviation |
|--------------------------|----------------------|--------------------|
| Logistic regression      | 0.73                 | 0.04               |
| Random Forest Classifier | 0.74                 | 0.05               |
| GaussianNB               | 0.75                 | 0.03               |
| XGB Classifier           | 0.69                 | 0.07               |
| SVC                      | 0.73                 | 0.04               |
| ANN                      | 0.71                 | -                  |

### Test Set Results
| Model                    | F1 score(macro) | Accuracy |
|--------------------------|-----------------|----------|
| Logistic regression      | 0.67            | 0.78     |
| Random Forest Classifier | 0.77            | 0.85     |
| GaussianNB               | 0.65            | 0.72     |
| XGB Classifier           | 0.71            | 0.85     |
| SVC                      | 0.71            | 0.78     |
| ANN                      | 0.62            | 0.79     |

Based on these results I've decided to go with Random Forest for the final classifier



## File structure

## How to run the project

### Option 1 - run the local hosted site with charts, interpretations and predictions

#### Step 1 - clone repo
in your local terminal run:
```
git clone https://github.com/kunick8/Heart-Abnormalities-Classifier.git
```


#### Step 2 - install poetry and dependencies through the terminal
If you don't have poetry installed:
```
pip install poetry
```
If you have poetry installed:
```
poetry install
```

#### Step 3 - launch the streamlit local host
```
streamlit run app/streamlit_app.py

```

All set, your browser should automatically open the local hosted site, if not then copy the url from the terminal

### Option 2 - train the final model and then run the local hosted site


#### Step 1 - clone repo
in your local terminal run:
```
git clone https://github.com/kunick8/Heart-Abnormalities-Classifier.git
```

#### Step 2 - install poetry and dependencies through the terminal
If you don't have poetry installed:
```
pip install poetry
```
If you have poetry installed:
```
poetry install
```

#### Step 3 - setup mlflow server
```
mlflow server --host 127.0.0.1 --port 5000 
 
```

#### Step 4 - run main.py to train the model
```
python3 src.main.py                                                                             
```

#### Step 5 - launch the streamlit local host
```
streamlit run app/streamlit_app.py

```

All set, your browser should automatically open the local hosted site, if not then copy the url from the terminal

### Option 3 - run the whole evaluation and training pipeline

#### Step 1 - clone repo
in your local terminal run:
```
git clone https://github.com/kunick8/Heart-Abnormalities-Classifier.git
```


#### Step 2 - install poetry and dependencies through the terminal

If you don't have poetry installed:
```
pip install poetry
```
If you have poetry installed:
```
poetry install
```

#### Step 3 - remove all previous mlflow experiments
```
rm -rf src/mlruns/
rm -f src/mlflow.db
rm -f mlflow.db   
```
#### Step 4 - setup mlflow server
```
mlflow server --host 127.0.0.1 --port 5000 
```

#### Step 5 - run evaluation_pipeline.py
```
python3 src.evaluation_pipeline.py
```

#### Step 6 - check out the results on the mlflow site

open http://127.0.0.1:5000 in your browser

#### Step 7 - run main.py to train the final model
```
python3 src.main.py                                                                             
```

#### Step 7 - launch the streamlit local host
```
streamlit run app/streamlit_app.py
```

All set, your browser should automatically open the local hosted site, if not then copy the url from the terminal


## wnioski i interpretacja po ang xd