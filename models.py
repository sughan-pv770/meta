from pydantic import BaseModel
from typing import Optional

class YourAction(BaseModel):
    action: str

class YourObservation(BaseModel):
    result: str
    success: bool
    error: Optional[str] = None

class YourState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
