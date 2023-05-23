from fastapi import FastAPI, UploadFile, File, Response
from .metadata import MetaData, models_utils
from keras.models import load_model, Model
from PIL import Image
from numpy import asarray, expand_dims
from json import dumps
from src.models import Application

print('Inicializando Redes Neurais...')
metadata = MetaData()
app = FastAPI()



def _predict_image(image: UploadFile, application: Application):
    keras_model: Model = load_model(application.get_path())
    img = Image.open(image.file)
    batch, width, height, colors = application.input_shape
    img = img.resize((width, height))
    sample = asarray(img)
    img.close()
    sample = expand_dims(sample, 0)
    preprocess, classes_decoder = models_utils[application.get_path()]
    predictions = classes_decoder(keras_model.predict(preprocess(sample)))[0]
    return {label:float(value) for _, label, value in predictions}


@app.get("/")
async def root():
    return {"message": "Hello World", 'deepuai': metadata}

@app.post("/apps/{model_id}/{dataset_id}/predict")
async def predict_image(model_id: str = 'resnet50', dataset_id: str = 'imagenet', image: UploadFile = File(...)):
    
    application = [applc for applc in metadata.applications if (
                                applc.model_id == model_id 
                                and applc.dataset_id == dataset_id)]
    
    if len(application) == 0:
        return Response(status_code=204, content=dumps({'message': 'The application was not found.'}))
    

    return Response(
        # headers={'Content-Type':'application/json'},
        media_type='application/json',
        content=dumps(_predict_image(image, application[0])))