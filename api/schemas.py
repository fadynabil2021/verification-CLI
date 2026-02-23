from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    log: str


class ClassifyResponse(BaseModel):
    label: str
    explanation: str
    confidence: float
