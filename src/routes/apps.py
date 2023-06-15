from fastapi import APIRouter, Response, UploadFile, File
from src.metadata import metadata
from json import dumps


router = APIRouter()

@router.post("/apps/{model_id}/{dataset_id}/predict")
async def predict_image(model_id: str = 'resnet50', dataset_id: str = 'imagenet', image: UploadFile = File(...)):
    
    application = [applc for applc in metadata.applications if (
                                applc.model_id == model_id 
                                and applc.dataset_id == dataset_id)]
    
    if len(application) == 0:
        return Response(
            status_code=204,
            media_type='application/json',
            content=dumps({'message': 'The application was not found.'}))
    else:
        application = application[0]
        return Response(
            media_type='application/json',
            content=dumps(application.predict_image(image)))

@router.post("/apps/{model_id}/{base_dataset_id}/train")
async def transfer_learning(
    model_id: str = 'resnet50', base_dataset_id: str = 'imagenet', dataset_id: str = 'frutas'):
    
    _applications = [applc for applc in metadata.applications if (
                                applc.model_id == model_id 
                                and applc.dataset_id == base_dataset_id)]
    if len(_applications) == 0:
        return Response(
            status_code=204,
            media_type='application/json',
            content=dumps({'message': 'The base application was not found.'}))
    
    
    _datasets = [x for x in metadata.datasets if x.id == dataset_id]
    if len(_datasets) == 0:
        return Response(
            status_code=204,
            media_type='application/json',
            content=dumps({'message': 'The dataset was not found.'}))
    
    application = _applications[0]
    dataset = _datasets[0]

    _, width, height, __ = application.input_shape
    dataset = dataset.as_keras_dataset(size=(width, height))
    print(type(dataset))
    application.transfer_learning(dataset_id=dataset_id, dataset=dataset)
    return Response(media_type='application/json', content=dumps({}))