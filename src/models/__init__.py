from pydantic import BaseModel

class Identified(BaseModel):
    id: str

from .application import DeepUaiApplication, utils as app_utils
from .dataset import Dataset
from .model import Model