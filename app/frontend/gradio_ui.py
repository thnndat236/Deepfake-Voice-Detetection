# app/frontend/gradio_ui.py
import gradio as gr
import requests
import os
import json
from datetime import datetime
import argparse

def predict_audio(api_url: str, audio_file):
    """
    Predict deepfake voice by calling FastAPI endpoint.
    
    Args:
        api_url (str): Base FastAPI URL.
        audio_file: Uploaded audio file path from Gradio.
    
    Returns:
        str: Formatted prediction result.
    """
    if not audio_file:
        return "Please upload an audio file."
    
    try:
        # Read the entire file into bytes to avoid closed file handle issues
        with open(audio_file, "rb") as f:
            content_bytes = f.read()
        
        # Determine MIME type based on file extension (fallback to audio/wav)
        filename = os.path.basename(audio_file)
        if filename.lower().endswith('.mp3'):
            mime_type = "audio/mpeg"
        elif filename.lower().endswith('.wav'):
            mime_type = "audio/wav"
        else:
            mime_type = "audio/wav"  # Default
        
        # Prepare files dict with bytes (no open file handle)
        files = {"file": (filename, content_bytes, mime_type)}
        
        # Call FastAPI /detect endpoint
        response = requests.post(
            f"{api_url}/api/dfv/detect",
            files=files,
            timeout=30  # 30s timeout for processing
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        # Parse JSON response
        result = response.json()
        
        # Format output
        timestamp = result.get("timestamp", datetime.now().isoformat())
        formatted_result = f"""
### DeepFake Voice Detection Result
- **Prediction**: {result.get('prediction', 'Unknown')}
- **Confidence**: {result.get('confidence', 0):.4f}
- **Audio Duration**: {result.get('audio_duration', 0):.2f} seconds
- **Segments Processed**: {result.get('segments_processed', 0)}/{result.get('num_segments', 0)}
- **Timestamp**: {timestamp}

{ '🔴 Suspected Fake!' if result.get('prediction') == 'FAKE' else '✅ Likely Real!' }
"""
        
        return formatted_result
    
    except requests.exceptions.RequestException as e:
        return f"Connection error to FastAPI server: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Invalid response from API: {str(e)}"
    except Exception as e:
        return f"Error during detection: {str(e)}"

def get_labels(api_url: str):
    """Get available labels from FastAPI."""
    try:
        response = requests.get(f"{api_url}/api/dfv/labels", timeout=10)
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        labels_data = response.json()
        labels = labels_data.get('available_labels', [])
        return f"Available Labels: {', '.join(labels)}"
    except Exception as e:
        return "Error fetching labels."

def health_check(api_url: str):
    """Check service health from FastAPI."""
    try:
        response = requests.get(f"{api_url}/api/dfv/health", timeout=10)
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        health_data = response.json()
        status = health_data.get('status', 'Unknown')
        return f"Service Status: {status}\nDetails: {health_data}"
    except Exception as e:
        return f"Health check failed: {str(e)}"

def gradio_launch(api_url: str):
    # Create Gradio Interface
    with gr.Blocks(title="DeepFake Voice Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 DeepFake Voice Detector")
        gr.Markdown("Upload an audio file (WAV, MP3, etc.) to detect if it contains deepfake voice.")
        
        with gr.Tab("Detect Audio"):
            audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
            predict_btn = gr.Button("Detect DeepFake", variant="primary")
            output = gr.Markdown()
            
            predict_btn.click(
                fn=lambda audio_file: predict_audio(api_url, audio_file),
                inputs=[audio_input],
                outputs=output
            )
        
        with gr.Tab("Info"):
            info_output = gr.Markdown()
            gr.Button("Load Labels").click(
                fn=lambda: get_labels(api_url),
                outputs=info_output
            )
        
        with gr.Tab("Health"):
            health_output = gr.Markdown()
            gr.Button("Check Service Health").click(
                fn=lambda: health_check(api_url),
                outputs=health_output
            )
    
    demo.launch()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url', 
        type=str, 
        default="http://localhost:30000",
        help="Base FastAPI URL"
    )
    args = parser.parse_args()
    api_url = args.url
    gradio_launch(api_url)

if __name__ == "__main__":
    main()