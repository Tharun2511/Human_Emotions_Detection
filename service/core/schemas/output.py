from pydantic import BaseModel

class APIOutput(BaseModel):
    emotion: str
    time_taken: str
    time_taken_preprocess: str