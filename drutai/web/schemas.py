from pydantic import BaseModel


class SmTarget(BaseModel):
    sm: int
    target: str


class SmSmile(BaseModel):
    sm: int
    smile: str


class FastaEntry(BaseModel):
    target_id: str
    sequence: str


class PredictRequest(BaseModel):
    method: str
    sm_targets: list[SmTarget]
    sm_smiles: list[SmSmile]
    fasta_entries: list[FastaEntry]


class Prediction(BaseModel):
    sm: int
    target: str
    prob_inter: float
    pred_type: str


class PredictResponse(BaseModel):
    method: str
    num_samples: int
    predictions: list[Prediction]


class MethodInfo(BaseModel):
    name: str
    available: bool
