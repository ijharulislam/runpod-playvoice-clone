import os
import time
import tempfile
import requests
import boto3
import runpod
from playdiffusion import PlayDiffusion, InpaintInput
from botocore.exceptions import ClientError
from uuid import uuid4
import io
import soundfile as sf


def upload_audio_bytes_to_s3(audio_data: bytes, bucket_name: str, object_key_prefix: str = "", file_extension=".wav") -> str:
    """
    Uploads raw audio bytes (e.g., from BytesIO) directly to a DigitalOcean Spaces bucket.

    Args:
        audio_data (bytes): Byte content of the audio file.
        bucket_name (str): Target bucket.
        object_key_prefix (str): Optional key prefix in the bucket.
        file_extension (str): File extension (e.g., ".wav", ".mp3").

    Returns:
        str: Public URL to the uploaded audio.
    """
    if not audio_data:
        raise ValueError("audio_data is empty")
    if not bucket_name:
        raise ValueError("bucket_name is required")

    aws_access_key_id = "DO801QRYN7XNMKV79HBC"
    aws_secret_access_key = "inKxzsLVWYaxS3kY4R5i9MvwMRw/0h3Ym7CeHV8T6U4"
    endpoint_url = "https://denoise.sfo3.cdn.digitaloceanspaces.com"

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url
    )

    object_key = f"{object_key_prefix}{uuid4()}{file_extension}"

    try:
        s3_client.upload_fileobj(io.BytesIO(
            audio_data), bucket_name, object_key)
        print(f"Uploaded audio to s3://{bucket_name}/{object_key}")
        return f"{endpoint_url}/{bucket_name}/{object_key}"
    except ClientError as e:
        raise RuntimeError(f"Failed to upload audio: {e}")


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
):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

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
        _, out_audio = inpainter.inpaint(inpaint_input)
        return out_audio
    except Exception as e:
        raise RuntimeError(f"Failed to perform audio inpainting: {str(e)}")


def handler(event):
    print(f"Worker Start")
    try:
        input_data = event.get('input', {})
        audio_url = input_data.get('audio_url')
        input_text = input_data.get('input_text')
        output_text = input_data.get('output_text')
        word_times = input_data.get('word_times')
        bucket_name = input_data.get(
            'bucket_name', "playdiffusion-inpainted-audio")
        object_key_prefix = input_data.get('object_key_prefix', "")

        if not audio_url or not input_text or not output_text or not word_times:
            raise ValueError("Missing required input fields")

        # Download original audio
        print(f"Downloading audio from: {audio_url}")
        response = requests.get(audio_url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download audio: HTTP {response.status_code}")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name

        print(f"Audio saved to: {temp_audio_path}")

        # Run inpainting
        out_audio = audio_inpainting(
            audio_path=temp_audio_path,
            input_text=input_text,
            output_text=output_text,
            word_times=word_times
        )

        # Save inpainted output to memory buffer
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, out_audio, samplerate=48000, format='WAV')
        audio_buffer.seek(0)

        # Upload to S3
        spaces_url = upload_audio_bytes_to_s3(
            audio_data=audio_buffer.read(),
            bucket_name=bucket_name,
            object_key_prefix=object_key_prefix,
            file_extension=".wav"
        )

        print(f"Uploaded inpainted audio to: {spaces_url}")

        return {
            'status': 'success',
            'audio_url': spaces_url,
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
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                print(f"Cleaned up temp file: {temp_audio_path}")
            except Exception as e:
                print(f"Cleanup failed: {e}")


# Start the serverless function
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
