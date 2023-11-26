import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import io
import torch
import os
import logging
from typing import List
import boto3
from botocore.exceptions import ClientError

from llm_model.scrutari_ocularis_model import ScrutariOcularisModel
from llm_model.scrutari_ocularis__model_utils import ScrutariOcularisModelUtils

logger = logging.getLogger("uvicorn")

app = FastAPI()

class ImageProcessingService:
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageProcessingService, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        if not self._is_initialized:
            self._is_initialized = True

            use_cuda = torch.cuda.is_available()
            use_mps = torch.backends.mps.is_available()
            logger.info(f"use_cuda model {use_cuda}")
            logger.info(f"use_mps model {use_mps}")

            self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
            
            self.model = ScrutariOcularisModel().to(self.device)
            name_file_model = "../models/scrutari_ocularis_model_v_1.pt"

            if os.path.exists(name_file_model):
                model_state = torch.load(name_file_model, map_location=self.device)
                self.model.load_state_dict(model_state)
                logger.info("Model loaded successfully")
            else:
                logger.info("No se encontró un modelo previo.")

            self.model.eval()

    async def process_image(self, image_data: bytes):
        # Asegurarse de que el servicio esté inicializado
        if not self._is_initialized:
            raise RuntimeError("El servicio no ha sido inicializado.")
        
        # Llamar al método preprocess del handler con los datos de la imagen
        # En este caso, simulamos una lista de diccionarios con claves 'body' que TorchServe pasaría
        preprocessed_data = ScrutariOcularisModelUtils.preprocess(logger, [{'body': image_data}])
        preprocessed_data = preprocessed_data.to(self.device)

        # Simular una inferencia llamando al método inference del handler
        # Necesitarías asegurarte de que tu modelo está cargado correctamente para esto
        inference_result = ScrutariOcularisModelUtils.inference(self.model, preprocessed_data)

        # Llamar al método postprocess para obtener la respuesta final
        postprocessed_result = ScrutariOcularisModelUtils.postprocess(inference_result)

        return postprocessed_result

# Instanciamos el servicio
image_service = ImageProcessingService()
image_service.initialize()

class ImageData(BaseModel):
    data: str  

@app.post("/process-image")
async def process_image(image_data: ImageData):
    try:
        # Procesar la imagen
        image_bytes = base64.b64decode(image_data.data)
        image_bytearray = bytearray(image_bytes)
        result = await image_service.process_image(image_bytearray)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class DataPaths(BaseModel):
    paths: List[str]  

# Configura tus credenciales y endpoint de MinIO aquí
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
ENDPOINT_URL = os.getenv('ENDPOINT_URL')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Configuración del cliente de S3
s3_client = boto3.client('s3',
                         endpoint_url=ENDPOINT_URL,
                         aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY)

# Método para descargar archivos de S3 a memoria
def download_file_to_memory(bucket_name, key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return response['Body'].read()
    except ClientError as e:
        print(f"No se pudo descargar el archivo: {e}")
        return None

@app.post("/process-image-s3")
async def process_image_s3(data_paths: DataPaths):
    results = []
    for path in data_paths.paths:
        try:
            # Descargar imagen de S3
            image_bytes = download_file_to_memory(BUCKET_NAME, path)
            if image_bytes:
                # Procesar la imagen como antes
                image_bytearray = bytearray(image_bytes)
                result = await image_service.process_image(image_bytearray)
                results.append({"path": path, "predicted": result})
            else:
                results.append({"path": path, "message": "Error al descargar la imagen o imagen no encontrada."})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results