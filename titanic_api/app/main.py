from fastapi import FastAPI

from app.api import api_router, root_router
from app.config import LoggingSettings, settings

_logger = LoggingSettings().get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)
_logger.info("Application instance created")

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
