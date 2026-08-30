import joblib
import numpy as np



def predict(data:np.ndarray):
    model = joblib.load('artifacts/model/svc.joblib')
    scaler = joblib.load('artifacts/model/scaler.joblib')
    selector = joblib.load('artifacts/model/selector.joblib')

    data = scaler.transform(data)
    data = selector.transform(data)

    prediction = model.predict(data)

    if prediction == 0:
        return 'Normal'
    else:
        return 'Abnormal'

