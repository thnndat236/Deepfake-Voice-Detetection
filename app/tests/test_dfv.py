from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch 
import pytest
from datetime import datetime
import tempfile
import os


# Fixture to provide a TestClient instance with module scope
@pytest.fixture(scope="module")
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def mock_model_service():
    """Mock ModelService"""
    with patch('api.routes.dfv_router.model_service') as mock_service:
        mock_service.is_ready.return_value = True
        mock_service.model_initialized = True
        mock_service.scaler_initialized = True
        mock_service.dfv_classifier_labels = ["FAKE", "REAL"]
        yield mock_service


@pytest.fixture
def sample_audio_file():
    """Create a temporary audio file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        # Write some dummy audio data
        f.write(b"RIFF" + b"\x00" * 100)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestHealthEndpoint:
    """Tests for /dfv/health endpoint"""
    
    def test_health_check_success(self, client, mock_model_service):
        """Test successful health check"""
        mock_model_service.health_check.return_value = {
            "status": "healthy",
            "scaler_initialized": True,
            "model_initialized": True,
            "available_labels_count": 2,
            "timestamp": datetime.now().isoformat()
        }
        
        response = client.get("/dfv/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_initialized"] is True
        assert data["scaler_initialized"] is True
    
    def test_health_check_service_not_ready(self, client):
        """Test health check when service is not ready"""
        with patch('api.routes.dfv_router.model_service') as mock_service:
            mock_service.is_ready.return_value = False
            
            response = client.get("/dfv/health")
            
            assert response.status_code == 503
            assert "not ready" in response.json()["detail"].lower()


class TestLabelsEndpoint:
    """Tests for /dfv/labels endpoint"""
    
    def test_get_labels_success(self, client, mock_model_service):
        """Test successful labels retrieval"""
        mock_model_service.get_available_labels.return_value = {
            "available_labels": ["FAKE", "REAL"],
            "available_labels_count": 2,
            "timestamp": datetime.now().isoformat()
        }
        
        response = client.get("/dfv/labels")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_labels" in data
        assert len(data["available_labels"]) == 2
        assert "FAKE" in data["available_labels"]
        assert "REAL" in data["available_labels"]
    
    def test_get_labels_service_not_ready(self, client):
        """Test labels endpoint when service is not ready"""
        with patch('api.routes.dfv_router.model_service') as mock_service:
            mock_service.is_ready.return_value = False
            
            response = client.get("/dfv/labels")
            
            assert response.status_code == 503


class TestDetectEndpoint:
    """Tests for /dfv/detect endpoint"""
    
    def test_detect_audio_success_real(self, client, mock_model_service, sample_audio_file):
        """Test successful audio detection - REAL audio"""
        mock_model_service.detect_audio_file.return_value = {
            "prediction": "REAL",
            "confidence": 0.85,
            "num_segments": 5,
            "segments_processed": 5,
            "audio_duration": 5.0
        }
        
        with open(sample_audio_file, "rb") as f:
            response = client.post(
                "/dfv/detect",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "REAL"
        assert data["confidence"] == 0.85
        assert data["num_segments"] == 5
        assert "timestamp" in data
    
    def test_detect_audio_success_fake(self, client, mock_model_service, sample_audio_file):
        """Test successful audio detection - FAKE audio"""
        mock_model_service.detect_audio_file.return_value = {
            "prediction": "FAKE",
            "confidence": 0.92,
            "num_segments": 3,
            "segments_processed": 3,
            "audio_duration": 3.0
        }
        
        with open(sample_audio_file, "rb") as f:
            response = client.post(
                "/dfv/detect",
                files={"file": ("fake_audio.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "FAKE"
        assert data["confidence"] == 0.92
    
    def test_detect_audio_no_file(self, client, mock_model_service):
        """Test detection without providing a file"""
        response = client.post("/dfv/detect")
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_detect_audio_service_not_ready(self, client, sample_audio_file):
        """Test detection when service is not ready"""
        with patch('api.routes.dfv_router.model_service') as mock_service:
            mock_service.is_ready.return_value = False
            
            with open(sample_audio_file, "rb") as f:
                response = client.post(
                    "/dfv/detect",
                    files={"file": ("test_audio.wav", f, "audio/wav")}
                )
            
            assert response.status_code == 503
    
    def test_detect_audio_processing_error(self, client, mock_model_service, sample_audio_file):
        """Test detection with processing error"""
        mock_model_service.detect_audio_file.side_effect = Exception("Processing failed")
        
        with open(sample_audio_file, "rb") as f:
            response = client.post(
                "/dfv/detect",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 500
        assert "Audio detection failed" in response.json()["detail"]
    
    def test_detect_audio_temp_file_cleanup(self, client, mock_model_service, sample_audio_file):
        """Test that temporary files are cleaned up"""
        created_temp_files = []
        
        original_tempfile = tempfile.NamedTemporaryFile
        
        def mock_tempfile(*args, **kwargs):
            temp = original_tempfile(*args, **kwargs)
            created_temp_files.append(temp.name)
            return temp
        
        with patch('tempfile.NamedTemporaryFile', side_effect=mock_tempfile):
            mock_model_service.detect_audio_file.return_value = {
                "prediction": "REAL",
                "confidence": 0.85,
                "num_segments": 5,
                "segments_processed": 5,
                "audio_duration": 5.0
            }
            
            with open(sample_audio_file, "rb") as f:
                response = client.post(
                    "/dfv/detect",
                    files={"file": ("test_audio.wav", f, "audio/wav")}
                )
            
            assert response.status_code == 200
            
            # Check that temp files were cleaned up
            for temp_path in created_temp_files:
                assert not os.path.exists(temp_path)


# Integration tests
class TestIntegration:
    """Integration tests for the entire router"""
    
    def test_full_workflow(self, client, mock_model_service, sample_audio_file):
        """Test complete workflow: health -> labels -> detect"""
        # 1. Check health
        mock_model_service.health_check.return_value = {
            "status": "healthy",
            "scaler_initialized": True,
            "model_initialized": True,
            "available_labels_count": 2,
            "timestamp": datetime.now().isoformat()
        }
        
        health_response = client.get("/dfv/health")
        assert health_response.status_code == 200
        
        # 2. Get labels
        mock_model_service.get_available_labels.return_value = {
            "available_labels": ["FAKE", "REAL"],
            "available_labels_count": 2,
            "timestamp": datetime.now().isoformat()
        }
        
        labels_response = client.get("/dfv/labels")
        assert labels_response.status_code == 200
        
        # 3. Detect audio
        mock_model_service.detect_audio_file.return_value = {
            "prediction": "REAL",
            "confidence": 0.85,
            "num_segments": 5,
            "segments_processed": 5,
            "audio_duration": 5.0
        }
        
        with open(sample_audio_file, "rb") as f:
            detect_response = client.post(
                "/dfv/detect",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        assert detect_response.status_code == 200
        assert detect_response.json()["prediction"] in ["FAKE", "REAL"]