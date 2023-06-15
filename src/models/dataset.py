from . import Identified
from os.path import join
from keras.utils import image_dataset_from_directory
class Dataset(Identified):
    path: str

    def as_keras_dataset(self, size: tuple):
        dataset = image_dataset_from_directory(
            image_size=size, directory=join('data', self.path), seed=42)
        return dataset
