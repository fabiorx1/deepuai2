from typing import List, Any
from pydantic import BaseModel
from src.models import Application, Dataset, Model
from keras import applications as keras_apps, __version__ as keras_v
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow import __version__ as tf_v
from os.path import isdir

models_utils = {}

class MetaData(BaseModel):
    keras_version = keras_v
    tensorflow_version = tf_v

    training_queue = []

    models: List[Model] = []
    datasets: List[Dataset] = []
    applications: List[Application] = []

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        # inicialização base
        dataset_id = 'imagenet'
        
        apps = []
        self._download_apps(apps, dataset_id)
        
        for app, preprocess, decoder in apps:
            _app = Application(model_id=app.name, dataset_id=dataset_id, input_shape=app.input_shape)
            
            path = _app.get_path()
            
            if not isdir(path): app.save(path)
            
            self.applications.append(_app)
            models_utils[path] = preprocess, decoder

    def _download_apps(self, apps: list, dataset_id: str='imagenet'):
        # print('Carregando VGG16...')
        # apps.append((keras_apps.VGG16(weights=dataset_id), preprocess_input, decode_predictions))
        # print('Carregando VGG19...')
        # apps.append((keras_apps.VGG19(weights=dataset_id), preprocess_input, decode_predictions))
        print('Carregando Xception...')
        apps.append((keras_apps.Xception(weights=dataset_id), lambda x: preprocess_input(x, mode='tf'), decode_predictions))
        print('Carregando ResNet50...')
        apps.append((keras_apps.ResNet50(weights=dataset_id), preprocess_input, decode_predictions))