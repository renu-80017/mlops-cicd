import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("model.pkl")
    data = load_iris()
    acc = accuracy_score(data.target, model.predict(data.data))
    assert acc >= 0.90

def test_prediction_shape():
    model = joblib.load("model.pkl")
    pred = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert len(pred) == 1
