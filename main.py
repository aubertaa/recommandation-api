from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from recommendation import generate_recommendations
from models import UserData

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Endpoint de base pour tester que l'API est en ligne
@app.get("/")
async def root():
    return {"message": "API de recommandation en ligne"}


# Endpoint pour envoyer les données d'utilisateur et récupérer des recommandations
@app.post("/recommendations/")
async def get_recommendations(data: UserData):
    try:
        logger.info("Received POST request with data: %s", data)

        recommendations = generate_recommendations(data)
        return {"recommendations": recommendations}

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
