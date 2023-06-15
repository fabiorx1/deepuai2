from fastapi import APIRouter, UploadFile, File, Response
from fastapi.staticfiles import StaticFiles
from os.path import join, split, exists, basename
from os import remove
import aiofiles, zipfile
from src.metadata import metadata
from src.models import Dataset

CHUNK_SIZE = 1024 * 1024
router = APIRouter()
statics = StaticFiles(directory=join('data','datasets'))

async def load_zip(zip_file: UploadFile):
    fname = basename(zip_file.filename)
    id = fname.split('.')[0]
    assert not exists(join('data', 'datasets', id))
    zip_path = join('data', 'datasets', fname)
    async with aiofiles.open(zip_path, 'wb') as file:
        while (chunk := await zip_file.read(CHUNK_SIZE)):
            await file.write(chunk)
    await zip_file.close()
    return zip_path, id

def extract_zip(path: str):
    assert exists(path)
    head = split(path)[0]
    with zipfile.ZipFile(path, 'r') as file:
        file.extractall(head)
    remove(path)
    return path.split('.')[0]

@router.post("/datasets/upload")
async def upload_dataset_as_zip(zip_file: UploadFile = File(...)):
    try:
        zip_path, id = await load_zip(zip_file)
        extract_zip(zip_path)
        path = join('datasets', id)
        metadata.datasets.append(Dataset(id=id, path=path))
        return {'path': path, 'dataset_id': id}
    except:
        return Response(status_code=409)