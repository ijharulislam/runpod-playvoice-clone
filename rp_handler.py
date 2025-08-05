import os
import traceback
import requests
import boto3
import runpod
from playdiffusion import PlayDiffusion, TTSInput
from botocore.exceptions import ClientError, ConnectionClosedError
from botocore.config import Config
from uuid import uuid4
import io
import mimetypes
import numpy as np
from scipy.io import wavfile
from urllib.parse import urlparse
import tempfile
import torch


def upload_to_s3(audio_data: bytes, bucket_name: str = None, object_key_prefix: str = "", file_extension: str = ".wav") -> str:
    """
    Upload audio data to a DigitalOcean Spaces bucket (S3-compatible) with public-read permissions, creating the bucket if it doesn't exist.

    Args:
        audio_data (bytes): Audio data to upload.
        bucket_name (str, optional): Name of the Spaces bucket. If None, uses default 'denoise'.
        object_key_prefix (str): Optional prefix for the S3 object key. Default: "".
        file_extension (str): File extension for the uploaded file. Default: ".wav".

    Returns:
        str: Spaces URL of the uploaded file, publicly accessible.

    Raises:
        ValueError: If audio_data is missing or bucket_name cannot be determined.
        ClientError: If bucket creation or upload fails due to permissions or other issues.
        RuntimeError: If a connection error occurs during upload.
    """
    if not audio_data:
        raise ValueError("audio_data is required")

    # Use environment variables for credentials
    aws_access_key_id = os.environ.get(
        "AWS_ACCESS_KEY_ID", "DO801QRYN7XNMKV79HBC")
    aws_secret_access_key = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", "inKxzsLVWYaxS3kY4R5i9MvwMRw/0h3Ym7CeHV8T6U4")
    endpoint_url = os.environ.get(
        "SPACES_ENDPOINT_URL", "https://sfo3.digitaloceanspaces.com")

    # Use default bucket name if not provided
    if not bucket_name:
        print("Warning: bucket_name not provided. Using default bucket 'denoise'.")
        bucket_name = "denoise"

    # Configure boto3 client with retries
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
        config=Config(retries={'max_attempts': 3, 'mode': 'standard'})
    )

    # Check if bucket exists, create if it doesn't
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Bucket '{bucket_name}' does not exist. Creating bucket...")
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"Created bucket '{bucket_name}'.")
            except ClientError as ce:
                raise ClientError(
                    f"Failed to create bucket '{bucket_name}': {str(ce)}", operation_name="create_bucket")
        else:
            raise ClientError(
                f"Error checking bucket '{bucket_name}': {str(e)}", operation_name="head_bucket")

    # Determine ContentType based on file extension
    content_type = mimetypes.guess_type(f"file{file_extension}")[
        0] or f"audio/{file_extension.lstrip('.')}"

    object_key = f"{object_key_prefix}{uuid4()}{file_extension}" if object_key_prefix else f"{uuid4()}{file_extension}"

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=audio_data,
            ContentType=content_type,
            ACL='public-read'  # Make the file publicly readable
        )
        # Construct the correct public URL
        parsed_endpoint = urlparse(endpoint_url)
        base_domain = parsed_endpoint.netloc
        spaces_url = f"https://{bucket_name}.{base_domain}/{object_key}"
        print(
            f"Uploaded file to s3://{bucket_name}/{object_key}, accessible at {spaces_url}")
        return spaces_url
    except ClientError as e:
        raise ClientError(
            f"Failed to upload to Spaces: {str(e)}", operation_name="put_object")
    except ConnectionClosedError as e:
        raise RuntimeError(
            f"Connection error during S3 upload: {str(e)}\nTry checking network stability or endpoint URL.")


def text_to_speech(
    reference_audio_url: str,
    transcript: str,
    bucket_name: str = None,
    object_key_prefix: str = "",
    num_steps: int = 30,
    init_temp: float = 1.0,
    init_diversity: float = 1.0,
    guidance: float = 0.5,
    rescale: float = 0.7,
    topk: int = 25,
    use_manual_ratio: bool = False,
    audio_token_syllable_ratio: float = None
) -> str:
    """
    Perform text-to-speech using PlayDiffusion and upload the result to DigitalOcean Spaces.

    Args:
        reference_audio_url (str): URL to the reference audio file for voice cloning.
        transcript (str): Text transcript to convert to speech.
        bucket_name (str, optional): Name of the Spaces bucket. Default: None (will use 'denoise' in upload_to_s3).
        object_key_prefix (str): Optional prefix for the S3 object key. Default: "".
        num_steps (int): Number of sampling steps for TTS (1-100). Default: 30.
        init_temp (float): Initial temperature (0.5-10). Default: 1.0.
        init_diversity (float): Initial diversity (0-10). Default: 1.0.
        guidance (float): Guidance scale (0-10). Default: 0.5.
        rescale (float): Guidance rescale factor (0-1). Default: 0.7.
        topk (int): Sampling from top-k logits (1-10000). Default: 25.
        use_manual_ratio (bool): Whether to use a manual audio token syllable ratio. Default: False.
        audio_token_syllable_ratio (float): Manual audio token syllable ratio (5.0-25.0). Default: None.

    Returns:
        str: Spaces URL of the uploaded TTS audio, publicly accessible.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If TTS fails or returns an unexpected value, with full traceback.
    """
    if not reference_audio_url:
        raise ValueError("reference_audio_url is required")
    if not transcript:
        raise ValueError("transcript cannot be empty")
    if not (1 <= num_steps <= 100):
        raise ValueError("num_steps must be between 1 and 100")
    if not (0.5 <= init_temp <= 10):
        raise ValueError("init_temp must be between 0.5 and 10")
    if not (0 <= init_diversity <= 10):
        raise ValueError("init_diversity must be between 0 and 10")
    if not (0 <= guidance <= 10):
        raise ValueError("guidance must be between 0 and 10")
    if not (0 <= rescale <= 1):
        raise ValueError("rescale must be between 0 and 1")
    if not (1 <= topk <= 10000):
        raise ValueError("topk must be between 1 and 10000")
    if use_manual_ratio and (audio_token_syllable_ratio is None or not (5.0 <= audio_token_syllable_ratio <= 25.0)):
        raise ValueError(
            "audio_token_syllable_ratio must be between 5.0 and 25.0 when use_manual_ratio is True")

    print(
        f"TTS input: reference_audio_url={reference_audio_url}, transcript={transcript}")

    # Download reference audio to a temporary file
    print(f"Downloading reference audio from: {reference_audio_url}")
    response = requests.get(reference_audio_url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download reference audio: HTTP {response.status_code}")

    # Create a temporary file for the reference audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(response.content)
        temp_audio_path = temp_audio.name

    print(f"Reference audio downloaded to: {temp_audio_path}")

    tts_engine = PlayDiffusion()
    if not use_manual_ratio:
        audio_token_syllable_ratio = None

    tts_input = TTSInput(
        voice=temp_audio_path,
        output_text=transcript,
        num_steps=num_steps,
        init_temp=init_temp,
        init_diversity=init_diversity,
        guidance=guidance,
        rescale=rescale,
        topk=topk,
        audio_token_syllable_ratio=audio_token_syllable_ratio
    )

    try:
        # Call TTS and debug the output
        tts_result = tts_engine.tts(tts_input)
        print(f"TTS result: {type(tts_result)}, {tts_result}")

        # Check if result is a tuple with two elements
        if not isinstance(tts_result, tuple) or len(tts_result) != 2:
            raise RuntimeError(
                f"Expected TTS to return a tuple with 2 elements (frequency, audio), got: {type(tts_result)}, {tts_result}")

        output_frequency, output_audio = tts_result

        # Validate output types
        if not isinstance(output_frequency, int):
            print(
                f"Warning: output_frequency is not an integer, got {type(output_frequency)}: {output_frequency}. Using default 16000 Hz.")
            output_frequency = 16000
        if not isinstance(output_audio, np.ndarray):
            raise RuntimeError(
                f"Expected output_audio to be numpy.ndarray, got: {type(output_audio)}")

        # Validate array shape and contents
        print(
            f"Raw output audio shape: {output_audio.shape}, dtype: {output_audio.dtype}")
        if output_audio.size == 0:
            raise RuntimeError(
                f"Output audio array is empty: {output_audio.shape}")
        if output_audio.ndim not in (1, 2):
            raise RuntimeError(
                f"Expected output_audio to be 1D or 2D, got {output_audio.ndim}D: {output_audio.shape}")

        # Reshape 1D array to (samples, 1) for mono audio if necessary
        if output_audio.ndim == 1:
            output_audio = output_audio.reshape(-1, 1)
        elif output_audio.shape[1] not in (1, 2):
            raise RuntimeError(
                f"Expected 1 or 2 channels, got {output_audio.shape[1]}: {output_audio.shape}")

        # Log audio details for debugging
        print(
            f"Processed output audio shape: {output_audio.shape}, dtype: {output_audio.dtype}, Frequency: {output_frequency} Hz")

        # Convert numpy.ndarray to WAV bytes using scipy.io.wavfile
        with io.BytesIO() as wav_buffer:
            wavfile.write(wav_buffer, output_frequency, output_audio)
            wav_bytes = wav_buffer.getvalue()

        # Upload to S3 and return the URL
        spaces_url = upload_to_s3(
            audio_data=wav_bytes,
            bucket_name=bucket_name,
            object_key_prefix=object_key_prefix,
            file_extension=".wav"
        )

        return spaces_url

    except Exception as e:
        raise RuntimeError(
            f"Failed to perform text-to-speech: {str(e)}\nTraceback: {traceback.format_exc()}")
    finally:
        # Clean up temporary reference audio file
        if os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                print(
                    f"Cleaned up temporary reference audio file: {temp_audio_path}")
            except Exception as e:
                print(
                    f"Failed to clean up temporary reference audio file: {str(e)}")


def call_webhook(url: str, data: dict):
    """
    Call a webhook with the given data.
    """
    try:
        response = requests.post(url, json=data)
        print(f"Webhook response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error calling webhook: {str(e)}")


def handler(event):
    """
    Processes incoming requests to the Serverless endpoint for text-to-speech.
    Downloads reference audio from a URL, performs TTS, and returns the Spaces URL.

    Args:
        event (dict): Contains the input data with 'reference_audio_url', 'transcript',
                      'bucket_name' (optional), and 'object_key_prefix' (optional).

    Returns:
        dict: Result containing the Spaces URL of the generated audio or error message.
    """
    print(f"Worker Start")
    try:
        # Extract input data
        input_data = event.get('input', {})
        reference_audio_url = input_data.get('reference_audio_url')
        transcript = input_data.get('transcript')
        bucket_name = input_data.get('bucket_name')  # Optional
        object_key_prefix = input_data.get(
            'object_key_prefix', "playvoice-clone")
        userid = input_data.get('userid')
        webhook_url = input_data.get(
            'webhook_url', "https://voicekiller.com/api/clone/webhook/")

        # Validate inputs
        if not reference_audio_url:
            raise ValueError("reference_audio_url is required")
        if not transcript:
            raise ValueError("transcript is required")

        print(f"Reference audio URL: {reference_audio_url}")
        print(f"Transcript: {transcript}")
        print(f"Bucket name: {bucket_name}")

        # Perform TTS and upload to S3
        spaces_url = text_to_speech(
            reference_audio_url=reference_audio_url,
            transcript=transcript,
            bucket_name=bucket_name,
            object_key_prefix=object_key_prefix
        )

        print(f"Uploaded TTS audio to: {spaces_url}")

        data = {
            'userid': userid,
            'audio_url': spaces_url
        }
        call_webhook(webhook_url, data)

        return {
            'status': 'success',
            'audio_url': spaces_url,
            'transcript': transcript
        }

    except (ValueError, RuntimeError, ClientError) as e:
        print(f"Error: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }
    except Exception as e:
        print(
            f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {
            'status': 'error',
            'message': f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}"
        }


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
