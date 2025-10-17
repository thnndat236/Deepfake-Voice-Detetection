import numpy as np
from ai_edge_litert.interpreter import Interpreter
from core.config import settings
import logging
logger = logging.getLogger(__name__)


# Default model path from settings
DFV_TFLITE_MODEL = settings.DFV_TFLITE_MODEL

class DFVClassifier(object):
    """
    DeepFake Voice Classifier using TensorFlow Lite model.
    Supports batch inference on audio features.
    """
    def __init__(self, model_path=DFV_TFLITE_MODEL, num_thread=1):
        """
        Initialize the TFLite interpreter.

        Args:
            model_path (str or Path): Path to the TFLite model file.
            num_thread (int): Number of threads for inference (default: 1).
        """
        logger.info("init model in dfv_detector.py")
        # Load and allocate the TFLite interpreter
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_thread
        )
        self.interpreter.allocate_tensors()
        
        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def __call__(self, feature_list):
        """
        Perform inference on a list of features.

        Args:
            feature_list (list or np.array): List of feature vectors (each 26-dim).
                Can be single sample (1D) or batch (2D).

        Returns:
            np.array or int: Binary prediction (0: REAL, 1: FAKE) for each sample.
                Scalar for single sample, array for batch.
        """
        # Convert input to numpy array with float32 dtype
        feature_array = np.array(feature_list, dtype=np.float32)
        
        # If input is 1D (single sample), add batch dimension
        if feature_array.ndim == 1:
            feature_array = np.expand_dims(feature_array, axis=0)
        
        # Resize input tensor to match the batch size
        batch_size = feature_array.shape[0]
        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'],
            [batch_size, feature_array.shape[1]]  # [batch, num_features]
        )
        self.interpreter.allocate_tensors()
        
        # Set the input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            feature_array
        )

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor
        result = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        # Squeeze output to remove extra dimensions
        probs = np.squeeze(result)

        # Threshold at 0.5 to get binary prediction (1: REAL if probs > 0.5)
        if probs.ndim == 0:
            # Single sample case: return scalar index
            result_index = 1 if probs > 0.5 else 0
        else:
            # Batch case: return array of indices
            result_index = (probs > 0.5).astype(int)

        return result_index
    