import os
import torch
import time
import tempfile
import requests
import boto3
import runpod
from playdiffusion import PlayDiffusion, InpaintInput
from botocore.exceptions import ClientError
from uuid import uuid4
import io
import mimetypes
import numpy as np
import soundfile as sf  # For saving NumPy array to WAV in-memory


def upload_to_s3(file_path: str = None, audio_data: bytes = None, bucket_name: str = None, object_key_prefix: str = "", file_extension: str = ".wav") -> str:
    """
    Upload a file or audio data to a DigitalOcean Spaces bucket (S3-compatible).

    Args:
        file_path (str, optional): Path to the file to upload. Mutually exclusive with audio_data.
        audio_data (bytes, optional): Audio data to upload directly. Mutually exclusive with file_path.
        bucket_name (str): Name of the Spaces bucket.
        object_key_prefix (str): Optional prefix for the S3 object key. Default: "".
        file_extension (str): File extension for the uploaded file. Default: ".wav".

    Returns:
        str: Spaces URL of the uploaded file.

    Raises:
        ValueError: If neither file_path nor audio_data is provided, or both are provided, or bucket_name is missing.
        FileNotFoundError: If the file_path does not exist.
        ClientError: If the upload fails.
    """
    if not bucket_name:
        raise ValueError("bucket_name is required")

    # Determine ContentType based on file extension
    content_type = mimetypes.guess_type(f"file{file_extension}")[
        0] or f"audio/{file_extension.lstrip('.')}"

    # Use environment variables for credentials (recommended for security)
    aws_access_key_id = os.environ.get(
        "AWS_ACCESS_KEY_ID", "DO801QRYN7XNMKV79HBC")
    aws_secret_access_key = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", "inKxzsLVWYaxS3kY4R5i9MvwMRw/0h3Ym7CeHV8T6U4")
    endpoint_url = "https://denoise.sfo3.cdn.digitaloceanspaces.com"

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url
    )

    object_key = f"{object_key_prefix}{uuid4()}{file_extension}" if object_key_prefix else f"{uuid4()}{file_extension}"

    try:
        if file_path:
            s3_client.upload_file(
                Filename=file_path,
                Bucket=bucket_name,
                Key=object_key,
                ExtraArgs={'ContentType': content_type}
            )
        else:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=audio_data,
                ContentType=content_type
            )
        print(f"Uploaded file to s3://{bucket_name}/{object_key}")
        spaces_url = f"{endpoint_url}/{bucket_name}/{object_key}"
        return spaces_url
    except ClientError as e:
        raise ClientError(
            f"Failed to upload to Spaces: {str(e)}", operation_name="upload_file" if file_path else "put_object")


def audio_inpainting(
    audio_path: str,
    input_text: str,
    output_text: str,
    word_times: list,
    num_steps: int = 30,
    init_temp: float = 1.0,
    init_diversity: float = 1.0,
    guidance: float = 0.5,
    rescale: float = 0.7,
    topk: int = 25,
    use_manual_ratio: bool = False,
    audio_token_syllable_ratio: float = None
) -> tuple[str, np.ndarray, int]:
    """
    Perform audio inpainting using PlayDiffusion.

    Args:
        audio_path (str): Path to the input audio file.
        input_text (str): Transcribed text from the input audio.
        output_text (str): Desired output text for inpainting.
        word_times (list): List of dictionaries with word timings (e.g., [{"word": str, "start": float, "end": float}, ...]).
        num_steps (int): Number of sampling steps for inpainting (1-100). Default: 30.
        init_temp (float): Initial temperature (0.5-10). Default: 1.0.
        init_diversity (float): Initial diversity (0-10). Default: 1.0.
        guidance (float): Guidance scale (0-10). Default: 0.5.
        rescale (float): Guidance rescale factor (0-1). Default: 0.7.
        topk (int): Sampling from top-k logits (1-10000). Default: 25.
        use_manual_ratio (bool): Whether to use a manual audio token syllable ratio. Default: False.
        audio_token_syllable_ratio (float): Manual audio token syllable ratio (5.0-25.0). Default: None.

    Returns:
        tuple[str, np.ndarray, int]: Path to the inpainted audio file, the audio data as a numpy.ndarray, and the output frequency.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If input parameters are invalid.
        RuntimeError: If inpainting fails or returns an unexpected value.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not input_text or not output_text:
        raise ValueError("Input and output text cannot be empty")
    if not word_times:
        raise ValueError("Word timings cannot be empty")
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

    inpainter = PlayDiffusion()
    if not use_manual_ratio:
        audio_token_syllable_ratio = None

    inpaint_input = InpaintInput(
        input_text=input_text,
        output_text=output_text,
        input_word_times=word_times,
        audio=audio_path,
        num_steps=num_steps,
        init_temp=init_temp,
        init_diversity=init_diversity,
        guidance=guidance,
        rescale=rescale,
        topk=topk,
        audio_token_syllable_ratio=audio_token_syllable_ratio
    )

    try:
        # Call inpaint and debug the output
        inpaint_result = inpainter.inpaint(inpaint_input)
        print(f"inpaint result: {type(inpaint_result)}, {inpaint_result}")

        # Check if result is a tuple with two elements
        if not isinstance(inpaint_result, tuple) or len(inpaint_result) != 2:
            raise RuntimeError(
                f"Expected inpaint to return a tuple with 2 elements (frequency, audio), got: {inpaint_result}")

        output_frequency, output_audio = inpaint_result

        # Validate output types
        if not isinstance(output_frequency, int):
            print(
                f"Warning: output_frequency is not an integer, got {type(output_frequency)}: {output_frequency}. Using default 16000 Hz.")
            output_frequency = 16000
        if not isinstance(output_audio, np.ndarray):
            raise RuntimeError(
                f"Expected output_audio to be numpy.ndarray, got: {type(output_audio)}")

        # Create a temporary file path for the output audio (required by PlayDiffusion)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            output_audio_path = temp_output.name
            sf.write(output_audio, output_audio_path,
                     output_frequency, format="WAV")

        return output_audio_path, output_audio, output_frequency
    except Exception as e:
        raise RuntimeError(f"Failed to perform audio inpainting: {str(e)}")


def handler(event):
    """
    Processes incoming requests to the Serverless endpoint for audio inpainting.
    Downloads audio from a URL, uses provided transcript, generates word timings using OpenAI Whisper API,
    performs inpainting, and uploads the result to a DigitalOcean Spaces bucket.

    Args:
        event (dict): Contains the input data with 'audio_url', 'input_text', 'output_text', 'word_times',
                      'bucket_name', and optional 'object_key_prefix'.

    Returns:
        dict: Result containing the Spaces URL of the inpainted audio or error message.
    """
    print(f"Worker Start")
    try:
        # Extract input data
        input_data = event.get('input', {})
        audio_url = input_data.get('audio_url')
        input_text = input_data.get('input_text')
        output_text = input_data.get('output_text')
        word_times = input_data.get('word_times')
        bucket_name = input_data.get(
            'bucket_name', "playdiffusion-inpainted-audio")
        object_key_prefix = input_data.get('object_key_prefix', "")

        # Validate inputs
        if not audio_url:
            raise ValueError("audio_url is required")
        if not input_text:
            raise ValueError("input_text is required")
        if not output_text:
            raise ValueError("output_text is required")
        if not word_times:
            raise ValueError("word_times is required")
        if not bucket_name:
            raise ValueError("bucket_name is required")

        # Download audio to a temporary file
        print(f"Downloading audio from: {audio_url}")
        response = requests.get(audio_url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download audio: HTTP {response.status_code}")

        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name

        print(f"Audio downloaded to: {temp_audio_path}")
        print(f"Provided input text: {input_text}")
        print(f"Provided output text: {output_text}")
        print(f"Word times: {word_times}")
        print(f"Audio path: {temp_audio_path}")

        # Call audio inpainting with default parameters
        output_audio_path, output_audio, output_frequency = audio_inpainting(
            audio_path=temp_audio_path,
            input_text=input_text,
            output_text=output_text,
            word_times=word_times
        )

        print(
            f"Inpainting completed. Output audio path: {output_audio_path}, Frequency: {output_frequency}")

        # Convert numpy.ndarray to WAV bytes using the output frequency
        with io.BytesIO() as wav_buffer:
            sf.write(output_audio, wav_buffer, output_frequency, format="WAV")
            wav_bytes = wav_buffer.getvalue()

        # Upload inpainted audio to DigitalOcean Spaces
        spaces_url = upload_to_s3(
            audio_data=wav_bytes,
            bucket_name=bucket_name,
            object_key_prefix=object_key_prefix,
            file_extension=".wav"
        )

        print(f"Uploaded inpainted audio to: {spaces_url}")

        # Clean up temporary files
        os.unlink(temp_audio_path)
        print(f"Cleaned up temporary input file: {temp_audio_path}")
        if os.path.exists(output_audio_path):
            os.unlink(output_audio_path)
            print(f"Cleaned up temporary output file: {output_audio_path}")

        return {
            'status': 'success',
            'spaces_url': spaces_url,
            'input_text': input_text,
            'word_times': word_times
        }

    except (FileNotFoundError, ValueError, RuntimeError, ClientError) as e:
        print(f"Error: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            'status': 'error',
            'message': f"Unexpected error: {str(e)}"
        }
    finally:
        # Ensure temporary files are deleted
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                print(f"Cleaned up temporary input file: {temp_audio_path}")
            except Exception as e:
                print(f"Failed to clean up temporary input file: {str(e)}")
        if 'output_audio_path' in locals() and os.path.exists(output_audio_path):
            try:
                os.unlink(output_audio_path)
                print(f"Cleaned up temporary output file: {output_audio_path}")
            except Exception as e:
                print(f"Failed to clean up temporary output file: {str(e)}")


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
