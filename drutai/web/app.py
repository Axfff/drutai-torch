from fastapi import FastAPI, HTTPException

from drutai.web.schemas import PredictRequest, PredictResponse, MethodInfo
from drutai.web import service

app = FastAPI(title="Drutai", description="Drug-Target Interaction Prediction")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/methods", response_model=list[MethodInfo])
async def get_methods():
    return service.list_methods()


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        return await service.predict(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
