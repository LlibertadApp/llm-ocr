import io
import os
import torch
from PIL import Image
import logging
import torchvision.transforms as transforms
from ts.torch_handler.base_handler import BaseHandler
from scrutari_ocularis__model_utils import ScrutariOcularisModelUtils

from scrutari_ocularis_model import ScrutariOcularisModel

logger = logging.getLogger(__name__)

class ScrutariOcularisHandler(BaseHandler):
    """
    Modelo personalizado para PyTorch Serve.
    """

    def __init__(self):
        super(ScrutariOcularisHandler, self).__init__()   
        self.initialized = False
             
    def initialize(self, context):
        """Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing

        """

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        # use GPU if available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f'Using device {self.device}')

        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = ScrutariOcularisModel()
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'Successfully loaded model from {model_file}')
        else:
            raise RuntimeError('Missing the model file')
        
        self.initialized = True
            
    def preprocess(self, data):
        return ScrutariOcularisModelUtils.preprocess(logger, data)

    def inference(self, preprocessed_data):
        return ScrutariOcularisModelUtils.inference(self.model, preprocessed_data)
    
    def postprocess(self, inference_output):
        return ScrutariOcularisModelUtils.postprocess(logger, self.mapping, inference_output)
