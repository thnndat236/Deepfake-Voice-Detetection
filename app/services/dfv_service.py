from fastapi import HTTPException, status
import logging
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from datetime import datetime, timezone
from core.config import settings
from services.dfv_detector import DFVClassifier
from fastapi.responses import JSONResponse
import numpy as np
import librosa
from core.config import settings
import joblib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.model_initialized = False
        self.scaler_initialized = False
        self._initialize_model(model_path=settings.DFV_TFLITE_MODEL)
        self.sr = 22050

    def _initialize_model(self, **kwargs):
        with self.tracer.start_as_current_span("dfv_model_initialization") as span:
            # Load DeepFakeVoice Model
            try:
                # DeepFakeVoice Labels
                self.dfv_classifier_labels = ["FAKE", "REAL"]
                span.set_attribute("model.labels.count", len(self.dfv_classifier_labels))
                span.set_attribute("model.labels.id2label", str(dict(enumerate(self.dfv_classifier_labels))))

                # DeepFakeVoice Scaler
                self.dfv_scaler = joblib.load(settings.DFV_SCALER)
                self.scaler_initialized = True
                span.set_attribute("scaler.initialization.success", True)

                # DeepFakeVoice Classifier
                span.set_attribute("model.type", "dfv_classifier")
                self.dfv_classifier = DFVClassifier(**kwargs)
                self.model_initialized = True
                span.set_attribute("model.initialization.success", True)

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))                
                span.set_attribute("scaler.initialization.success", False)
                self.scaler_initialized = False
                span.set_attribute("model.initialization.success", False)
                self.model_initialized = False
                logger.error(f"Error initializing DeepFakeVoice Classifier: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize DeepFakeVoice Classifier: {str(e)}"
                )

    def is_ready(self) -> bool:
        return self.scaler_initialized and self.model_initialized

    def get_available_labels(self) -> JSONResponse:
        try:
            if not self.model_initialized or not self.scaler_initialized:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="DeepFakeVoice Classifier not initialized"
                )

            response_data = {
                "available_labels": self.dfv_classifier_labels,
                "available_labels_count": len(self.dfv_classifier_labels),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            return JSONResponse(
                content=response_data,
                status_code=status.HTTP_200_OK
            )
    
        except HTTPException as e:
            raise e

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get available labels: {str(e)}"
            )

    def health_check(self) -> JSONResponse:
        try:
            response_data = {
                "status": "healthy" if self.model_initialized and self.scaler_initialized else "not_ready",
                "scaler_initialized": self.scaler_initialized,
                "model_initialized": self.model_initialized,
                "available_labels_count": len(self.dfv_classifier_labels) if self.model_initialized else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            return JSONResponse(
                content=response_data,
                status_code=status.HTTP_200_OK
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )

    def _extract_features(self, audio_segment):
        """Extract 26 features from audio segment"""
        with self.tracer.start_as_current_span("extract_audio_features") as span:
            span.set_attribute("audio.segment.length", len(audio_segment))
            try:
                # Ensure segment is correct length
                segment_samples = self.sr  # 1 second
                if len(audio_segment) < segment_samples:
                    audio_segment = np.pad(audio_segment, (0, segment_samples - len(audio_segment)))
                    logger.info(f"Padded audio segment to {segment_samples} samples")
                elif len(audio_segment) > segment_samples:
                    audio_segment = audio_segment[:segment_samples]
                    logger.info(f"Truncated audio segment to {segment_samples} samples")
                
                # Extract 6 statistical features
                span.set_attribute("features.type", "statistical")
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio_segment, sr=self.sr))
                rms = np.mean(librosa.feature.rms(y=audio_segment))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
                
                # Extract 20 MFCCs
                span.set_attribute("features.type", "mfcc")
                mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=20)
                mfccs_mean = np.mean(mfccs, axis=1)
                
                # Combine all features
                features = np.array([chroma_stft, rms, spectral_centroid, spectral_bandwidth, 
                                rolloff, zcr, *mfccs_mean], dtype=np.float32)
                span.set_attribute("features.count", len(features))
                span.set_attribute("features.shape", str(features.shape))
                # logger.info(f"Successfully extracted {len(features)} features from audio segment")

                return features
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Feature extraction failed: {str(e)}"))
                logger.error(f"Error extracting features from audio segment of length {len(audio_segment)}: {str(e)}")
                raise Exception(f"Feature extraction failed: {str(e)}")
    
    def detect_audio_file(self, file_path):
        """Process entire audio file"""
        with self.tracer.start_as_current_span("detect_audio_file") as span:
            span.set_attribute("input.file_path", file_path)
            try:
                # Load audio
                span.set_attribute("step", "load_audio")
                y, sr = librosa.load(file_path, sr=self.sr, mono=True)
                audio_duration = len(y) / sr
                span.set_attribute("audio.duration_seconds", audio_duration)
                span.set_attribute("audio.sample_rate", sr)
                span.set_attribute("audio.samples_count", len(y))
                logger.info(f"Loaded audio file {file_path} - duration: {audio_duration:.2f}s - samples: {len(y)}")

                # Split into 1-second segments
                span.set_attribute("step", "split_segments")
                segment_samples = self.sr
                num_segments = int(np.ceil(len(y) / segment_samples))
                span.set_attribute("segments.count", num_segments)
                logger.info(f"Split audio into {num_segments} segments")

                features_list = []
                for i in range(num_segments):
                    start = i * segment_samples
                    end = min(len(y), (i + 1) * segment_samples)
                    segment = y[start:end]
                    
                    with self.tracer.start_as_current_span(f"extract_segment_features_{i}") as segment_span:
                        segment_span.set_attribute("segment.index", i)
                        segment_span.set_attribute("segment.length", len(segment))
                        try:
                            features = self._extract_features(segment)
                            features_list.append(features)
                        except Exception as seg_e:
                            segment_span.record_exception(seg_e)
                            segment_span.set_status(Status(StatusCode.ERROR, str(seg_e)))
                            logger.warning(f"Failed to extract features for segment {i}: {str(seg_e)}. Skipping segment.")
                            continue

                if not features_list:
                    raise Exception("No valid segments could be processed")

                # Scale features list
                span.set_attribute("step", "scale")
                features_scaled = self.dfv_scaler.transform(features_list)

                # Batch prediction
                span.set_attribute("step", "predict")
                prob_reals = self.dfv_classifier(features_scaled)
                
                # Handle single segment case
                if isinstance(prob_reals, (float, np.floating)):
                    prob_reals = np.array([prob_reals])
                
                span.set_attribute("prediction.segments_processed", len(prob_reals))

                # Calculate overall prediction
                avg_prob_real = np.mean(prob_reals)
                prediction = "FAKE" if avg_prob_real < 0.5 else "REAL"
                confidence = float(avg_prob_real if prediction == "REAL" else 1 - avg_prob_real)

                span.set_attribute("prediction.result", prediction)
                span.set_attribute("prediction.confidence", confidence)

                # Prepare response
                result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "num_segments": num_segments,
                    "segments_processed": len(prob_reals),
                    "audio_duration": float(audio_duration)
                }
                logger.info(f"Audio detection completed: {prediction} (confidence: {confidence:.4f}), processed {len(prob_reals)}/{num_segments} segments")
                return result
        
            except librosa.LibrosaError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Librosa error in audio processing: {str(e)}"))
                logger.error(f"Librosa error processing {file_path}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid audio file: {str(e)}"
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Audio processing failed: {str(e)}"))
                logger.error(f"Unexpected error processing audio file {file_path}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Audio processing failed: {str(e)}"
                )
