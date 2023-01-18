import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from app.schemas import Health, PredictionResults, MultipleTitanicDataInputs
from typing import Any
from app.config import settings
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
from loguru import logger
from src.models.predict import make_prediction


root_router = APIRouter()
api_router = APIRouter()


@root_router.get('/')
def index(request: Request) -> Any:
    """
    Basic HTML response.
    """
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)


@api_router.get('/health', response_model = Health, status_code=200)
def health() -> dict:
    """
    Root Get.
    """
    health = Health(name = settings.PROJECT_NAME,
                    api_version = settings.API_VERSION,
                    model_version = settings.MODEL_VERSION)
    return health.dict()


@api_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleTitanicDataInputs) -> Any:
    """
    Make titanic survival predictions with the latest model.
    """
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results