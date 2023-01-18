import numpy as np
from src.models.predict import make_prediction


def test_make_prediction(sample_input_data):

    expected_first_prediction = 0
    expected_13th_prediction = 1
    expected_last_prediction = 0
    expected_prediction_size = 418
    expected_prediction_type = np.ndarray
    expected_single_prediction_type = np.int64
    expected_errors = None

    results = make_prediction(sample_input_data)
    predictions = results.get('predictions')
    errors = results.get('errors')

    assert expected_first_prediction == predictions[0]
    assert expected_13th_prediction == predictions[12]
    assert expected_last_prediction == predictions[-1]
    assert expected_prediction_size == len(predictions)
    assert isinstance(predictions, expected_prediction_type)
    assert isinstance(predictions[0], expected_single_prediction_type)
    assert expected_errors == errors





