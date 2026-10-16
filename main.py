import uvicorn

from visitbrief.api import create_app
from visitbrief.settings import Settings

app = create_app()


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False)