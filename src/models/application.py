from pydantic import BaseModel
from typing import Callable, Any
from os.path import join, isdir
from keras import Model

APPLIC_PATH_PREFIX = ('data','applications')


class Application(BaseModel):
    model_id: str
    dataset_id: str
    input_shape: tuple
    version: int = 1

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        print('Configurando ', self.get_path())

    def get_path(self):
        prefix = APPLIC_PATH_PREFIX
        return join(prefix[0], prefix[1], self.model_id, self.dataset_id)

class IApplication:
    k : Model
    outer : Application
    preprocess: Callable
    decoder: Callable

    def __init__(self, k, o, p, d) -> None:
        self.k = k
        self.outer = o
        self.preprocess = p
        self.decoder = d