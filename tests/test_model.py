import joblib
from src.config import MODEL_PATH
from src.data_prep import get_clean_data
from src.features import split_xy


def test_saved_model_outputs_valid_probabilities():
    model = joblib.load(MODEL_PATH)
    X, _ = split_xy(get_clean_data())
    p = model.predict_proba(X.head(50))[:, 1]
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()
