import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    relevant_data = test_data.replace('?', np.nan)
    relevant_data.dropna(subset = ['age'], inplace = True)
    print(relevant_data)
    relevant_data['age'] = relevant_data['age'].astype(float)
    relevant_data['age'] = relevant_data['age'].round().astype(int)
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": relevant_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then

    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] is None
    assert prediction_data["predictions"][0] == 0