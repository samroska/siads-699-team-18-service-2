import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import logging
from typing import Dict, Union, Optional
import os
import zipfile
import tempfile
import shutil
import glob

logger = logging.getLogger(__name__)

# Dictionary to store multiple models and their loaded state
_models: Dict[str, Optional[tf.keras.Model]] = {}
_models_loaded: Dict[str, bool] = {}
_temp_dirs: Dict[str, Optional[str]] = {}

class SkinLesionClassifier:
    @staticmethod
    def _extract_model_if_zipped(model_path: str, model_name: str = 'default') -> str:
        """
        If the model_path is a zip file, extract and return the .keras file path.
        Otherwise, return the model_path as-is.
        """
        if model_path.endswith('.zip'):
            global _temp_dirs
            if model_name not in _temp_dirs or not _temp_dirs[model_name]:
                _temp_dirs[model_name] = tempfile.mkdtemp(suffix='_model')
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(_temp_dirs[model_name])
            for root, dirs, files in os.walk(_temp_dirs[model_name]):
                for file in files:
                    if file.endswith('.keras'):
                        return os.path.join(root, file)
            raise FileNotFoundError(f"No .keras model file found in the zip archive for {model_name}")
        else:
            return model_path

    CLASS_NAMES = ['nevus',"melanoma","other","squamous cell carcinoma","solar lentigo","basal cell carcinoma", "melanoma metastasis" , "seborrheic keratosis", "actinic keratosis","dermatofibroma", "scar", "vascular lesion"]
    INPUT_SIZE = (224, 224)
    DEFAULT_MODEL_ZIP = 'BCN20000.keras.zip'
    MODEL_CONFIGS = {}  # Add this line to avoid attribute errors. Populate as needed.
    
    @staticmethod
    def _extract_model_from_zip(zip_path: str) -> str:
        """Extract model from BCN20000.keras.zip and return the .keras file path."""
        global _temp_dirs
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Model zip file not found: {zip_path}")

        if 'default' not in _temp_dirs or not _temp_dirs['default']:
            _temp_dirs['default'] = tempfile.mkdtemp(suffix='_model')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(_temp_dirs['default'])

        for root, dirs, files in os.walk(_temp_dirs['default']):
            for file in files:
                if file.endswith('.keras'):
                    model_path = os.path.join(root, file)
                    logger.info(f"Found model file: {model_path}")
                    return model_path
        raise FileNotFoundError("No .keras model file found in the zip archive")
    @staticmethod
    def _ensure_model_loaded():
        """Ensure the model is loaded from BCN20000.keras.zip."""
        global _models, _models_loaded
        if 'default' in _models_loaded and _models_loaded['default'] and _models.get('default') is not None:
            return
        zip_path = SkinLesionClassifier.DEFAULT_MODEL_ZIP
        try:
            model_path = SkinLesionClassifier._extract_model_from_zip(zip_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            _models['default'] = tf.keras.models.load_model(model_path)
            _models_loaded['default'] = True
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            SkinLesionClassifier._cleanup_temp_files()
            raise
    
    @staticmethod
    def _ensure_model_loaded(model_name: str = 'default'):
        """Ensure the specified model is loaded."""
        global _models, _models_loaded
        
        if model_name in _models_loaded and _models_loaded[model_name] and _models.get(model_name) is not None:
            return
        
        if model_name in SkinLesionClassifier.MODEL_CONFIGS:
            model_path = SkinLesionClassifier.MODEL_CONFIGS[model_name]
        elif model_name == 'default':
            model_path = SkinLesionClassifier.DEFAULT_MODEL_ZIP
        else:
            model_path = model_name
        
        try:
            actual_model_path = SkinLesionClassifier._extract_model_if_zipped(model_path, model_name)
            
            if not os.path.exists(actual_model_path):
                raise FileNotFoundError(f"Model file not found for {model_name}: {actual_model_path}")
            
            _models[model_name] = tf.keras.models.load_model(actual_model_path)
            _models_loaded[model_name] = True
            logger.info(f"Model '{model_name}' loaded successfully from {actual_model_path}")
            
        except Exception as e:
            logger.error(f"Error loading {model_name} model: {e}")
            SkinLesionClassifier._cleanup_temp_files(model_name)
            raise
    
    @staticmethod
    def _cleanup_temp_files(model_name: Optional[str] = None):
        global _temp_dirs
        if model_name is not None:
            temp_dir = _temp_dirs.get(model_name)
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory for {model_name}: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary directory for {model_name}: {e}")
            _temp_dirs.pop(model_name, None)
        else:
            for temp_dir in _temp_dirs.values():
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Could not clean up temporary directory: {e}")
            _temp_dirs.clear()
    
    @staticmethod
    def preprocess_image(image: Union[Image.Image, str]) -> np.ndarray:
        try:
            if isinstance(image, str):
                image_rgb = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image_rgb = image.convert('RGB')
            else:
                raise ValueError("Image must be a PIL Image object or file path")
            image_array = img_to_array(image_rgb)
            resized_image = tf.image.resize(image_array, SkinLesionClassifier.INPUT_SIZE)
            processed_array = img_to_array(resized_image).reshape(1, SkinLesionClassifier.INPUT_SIZE[0], SkinLesionClassifier.INPUT_SIZE[1], 3)
            processed_array = processed_array / 255.0
            return processed_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
            
    @staticmethod
    def predict(image: Union[Image.Image, str]) -> Dict[str, float]:
        """
        Make prediction using the default model.
        """
        try:
            SkinLesionClassifier._ensure_model_loaded()
            if 'default' not in _models or _models['default'] is None:
                raise RuntimeError("Model failed to load.")
            processed_image = SkinLesionClassifier.preprocess_image(image)
            prediction = _models['default'].predict(processed_image, verbose=0)
            results = {}
            for i, class_name in enumerate(SkinLesionClassifier.CLASS_NAMES):
                results[class_name] = float(round(prediction[0][i], 3))
            logger.info(f"Prediction completed: {results}")
            return results
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    @staticmethod
    def get_top_prediction(image: Union[Image.Image, str]) -> tuple:
        predictions = SkinLesionClassifier.predict(image)
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]
        return top_class, confidence
    
    @staticmethod
    def print_predictions(image: Union[Image.Image, str]):
        predictions = SkinLesionClassifier.predict(image)
        print('\nProbabilities:')
        for class_name, probability in predictions.items():
            print(f'{class_name}: {probability}')
    
    @staticmethod
    def get_prediction_summary(image: Union[Image.Image, str]) -> Dict:
        predictions = SkinLesionClassifier.predict(image)
        top_class, confidence = SkinLesionClassifier.get_top_prediction(image)
        return {
            'top_prediction': {
                'class': top_class,
                'confidence': confidence
            },
            'all_predictions': predictions,
            'model_info': {
                'classes': SkinLesionClassifier.CLASS_NAMES,
                'input_size': SkinLesionClassifier.INPUT_SIZE
            }
        }
    
    @staticmethod
    def cleanup():
        global _models, _models_loaded
        SkinLesionClassifier._cleanup_temp_files()
        _models.clear()
        _models_loaded.clear()
        logger.info("Static classifier resources cleaned up")

# Backward compatibility functions
def load_model():
    """Load the model (for backward compatibility)."""
    SkinLesionClassifier._ensure_model_loaded()

def inference_function(image: Union[Image.Image, str]) -> Dict[str, float]:
    """Legacy inference function that uses the static classifier."""
    return SkinLesionClassifier.predict(image)