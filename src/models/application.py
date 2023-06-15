
from PIL import Image
from typing import Any
from os.path import join
from fastapi import UploadFile
from pydantic import BaseModel
from numpy import asarray, expand_dims
from keras.models import load_model, Model, Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.python.data import Dataset

APPLIC_PATH_PREFIXES = ('data','applications')
utils = {}

class DeepUaiApplication(BaseModel):
    model_id: str
    dataset_id: str
    input_shape: tuple
    epochs: int = 0
    version: int = 1

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        print('Configurando ', self.get_path())

    def get_path(self):
        return join(*APPLIC_PATH_PREFIXES, self.model_id, self.dataset_id)
    
    def predict_image(self, image: UploadFile):
        keras_model: Model = load_model(self.get_path())
        img = Image.open(image.file)
        batch, width, height, colors = self.input_shape
        img = img.resize((width, height))
        sample = asarray(img)
        img.close()
        sample = expand_dims(sample, 0)
        preprocess, classes_decoder = utils[self.get_path()]
        predictions = classes_decoder(keras_model.predict(preprocess(sample)))[0]
        return {label:float(value) for _, label, value in predictions}

    def transfer_learning(self, dataset_id: str, dataset: Dataset, activation: str = 'softmax'):
        classes = dataset.class_names
        base_model: Model = load_model(self.get_path())
        body = Model(base_model.input, base_model.layers[-2].output)
        for layer in body.layers: layer.trainable = False

        new_model = Sequential()
        new_model.add(body)
        new_model.add(Dense(512, activation='relu'))
        new_model.add(Dense(len(classes), activation=activation))

        loss = ('sparse_categorical_crossentropy'
            if len(classes) > 2 else 'binary_crossentropy')
        new_model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])

        from src.metadata import metadata
        saved_checkpoint = metadata.find_app(base_model.name,dataset_id)
        checkpoint = (saved_checkpoint if saved_checkpoint else
                   DeepUaiApplication(model_id=base_model.name,
                                      dataset_id=dataset_id, input_shape=new_model.input_shape))
        

        new_model.fit(dataset, epochs=5)
        new_model.save(checkpoint.get_path())
        checkpoint.epochs += 5
        
        if not saved_checkpoint:
            metadata.applications.append(checkpoint)