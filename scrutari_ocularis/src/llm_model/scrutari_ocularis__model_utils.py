from PIL import Image
import io
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

class ScrutariOcularisModelUtils():
    @staticmethod
    def preprocess(logger, data):
        # Define las transformaciones
        logger.info(f'Received {data} texts.')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # No se especifica la normalización aquí, pero podrías incluir una línea similar a:
            # transforms.Normalize(mean=[mean_value], std=[std_value]) si es necesario
        ])

        images = []

        # Aplica las transformaciones a cada imagen
        for row in data:
            image = row.get('data') or row.get('body')
            # Convertir los bytes o bytearray a un objeto de imagen PIL
            image = Image.open(io.BytesIO(image)) if isinstance(image, (bytearray, bytes)) else Image.open(image)
            
            # Convertir a escala de grises y a 'L' si es necesario
            if image.mode != 'L':
                image = F.to_grayscale(image, num_output_channels=1)
            
            # Aplicar la transformación
            image = transform(image)
            images.append(image)

        return torch.stack(images)

    @staticmethod
    def inference(model, preprocessed_data):
        with torch.no_grad():  # No necesitamos calcular gradientes para la inferencia
            return model(preprocessed_data)

    @staticmethod
    def postprocess(inference_output):
        # Convertir la salida del modelo en una respuesta JSON, o en el formato deseado
        predicted_classes = [output.argmax().item() for output in inference_output]
        return predicted_classes