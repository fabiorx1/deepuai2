from typing import List, Any
from pydantic import BaseModel
from src.models import DeepUaiApplication, Dataset, Model, app_utils
from keras import applications as keras_apps, __version__ as keras_v
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow import __version__ as tf_v
from os.path import isdir, join
from os import listdir

LOAD = True

class MetaData(BaseModel):
    keras_version = keras_v
    tensorflow_version = tf_v

    training_queue = []

    models: List[Model] = []
    datasets: List[Dataset] = []
    applications: List[DeepUaiApplication] = []

    def _download_apps(self, dataset_id: str='imagenet'):
        apps = []
        # print('Carregando VGG16...')
        # apps.append((keras_apps.VGG16(weights=dataset_id), preprocess_input, decode_predictions))
        # print('Carregando VGG19...')
        # apps.append((keras_apps.VGG19(weights=dataset_id), preprocess_input, decode_predictions))
        print('Carregando Xception...')
        apps.append((keras_apps.Xception(weights=dataset_id), lambda x: preprocess_input(x, mode='tf'), decode_predictions))
        print('Carregando ResNet50...')
        apps.append((keras_apps.ResNet50(weights=dataset_id), preprocess_input, decode_predictions))
        return apps

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        # inicializando datasets
        for id in listdir(join('data', 'datasets')):
            self.datasets.append(Dataset(id=id, path=join('datasets', id)))
        
        # inicialização redes neurais
        dataset_id = 'imagenet'
        apps = self._download_apps(dataset_id)
        
        for app, preprocess, decoder in apps:
            deepuai_app = DeepUaiApplication(model_id=app.name, dataset_id=dataset_id, input_shape=app.input_shape)
            
            app_root = deepuai_app.get_path()
            if not isdir(app_root): app.save(app_root)
            app_utils[app_root] = preprocess, decoder
            
            self.applications.append(deepuai_app)
    
    def find_app(self, model_id: str, dataset_id: str):
        apps = [app for app in self.applications if app.model_id == model_id and app.dataset_id == dataset_id]
        if len(apps) == 0: return None
        else: return apps[0]


if LOAD:
    print('Inicializando Redes Neurais...')
    metadata = MetaData()