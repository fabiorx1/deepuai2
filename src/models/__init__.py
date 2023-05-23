from pydantic import BaseModel

class Identified(BaseModel):
    id: str

from .application import Application
from .dataset import Dataset
from .model import Model