# PlayDiffusion Text-to-Speech Serverless Function

This is a serverless function that uses PlayDiffusion to perform text-to-speech (TTS) with voice cloning capabilities. The function takes a reference audio file and a transcript, then generates speech in the voice of the reference audio.

## Features

- **Voice Cloning**: Uses reference audio to clone the voice for TTS generation
- **High-Quality Audio**: Leverages PlayDiffusion's advanced TTS models
- **Cloud Storage**: Automatically uploads generated audio to DigitalOcean Spaces
- **Serverless**: Runs on RunPod's serverless platform for scalable processing

## API Usage

### Input Parameters

The function expects a JSON payload with the following structure:

```json
{
    "input": {
        "reference_audio_url": "https://example.com/reference_audio.wav",
        "transcript": "Hello, this is the text I want to convert to speech.",
        "bucket_name": "tts-output",
        "object_key_prefix": "tts/"
    }
}
```

#### Required Parameters

- `reference_audio_url` (string): URL to the reference audio file (WAV format recommended)
- `transcript` (string): Text to convert to speech

#### Optional Parameters

- `bucket_name` (string): DigitalOcean Spaces bucket name (defaults to "denoise")
- `object_key_prefix` (string): Prefix for the uploaded file name (defaults to "")
- `num_steps` (int): Number of sampling steps (1-100, default: 30)
- `init_temp` (float): Initial temperature (0.5-10, default: 1.0)
- `init_diversity` (float): Initial diversity (0-10, default: 1.0)
- `guidance` (float): Guidance scale (0-10, default: 0.5)
- `rescale` (float): Guidance rescale factor (0-1, default: 0.7)
- `topk` (int): Sampling from top-k logits (1-10000, default: 25)
- `audio_token_syllable_ratio` (float): Manual audio token syllable ratio (5.0-25.0, optional)

### Response

The function returns a JSON response with the following structure:

```json
{
    "status": "success",
    "audio_url": "https://bucket-name.sfo3.digitaloceanspaces.com/generated_audio.wav",
    "transcript": "Hello, this is the text I want to convert to speech."
}
```

#### Success Response

- `status`: "success"
- `audio_url`: Public URL to the generated audio file
- `transcript`: The original transcript that was converted

#### Error Response

- `status`: "error"
- `message`: Error description

## Environment Variables

The following environment variables are used for DigitalOcean Spaces configuration:

- `AWS_ACCESS_KEY_ID`: DigitalOcean Spaces access key
- `AWS_SECRET_ACCESS_KEY`: DigitalOcean Spaces secret key
- `SPACES_ENDPOINT_URL`: DigitalOcean Spaces endpoint URL

## Example Usage

### Using curl

```bash
curl -X POST "https://your-runpod-endpoint.runpod.net" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Using Python

```python
import requests

url = "https://your-runpod-endpoint.runpod.net"
payload = {
    "input": {
        "reference_audio_url": "https://example.com/reference_audio.wav",
        "transcript": "Hello, this is a test of the TTS functionality.",
        "bucket_name": "tts-output"
    }
}

response = requests.post(url, json=payload)
result = response.json()

if result["status"] == "success":
    print(f"Generated audio: {result['audio_url']}")
else:
    print(f"Error: {result['message']}")
```

## Technical Details

- **Model**: Uses PlayDiffusion's TTS model for high-quality voice synthesis
- **Audio Format**: Output is 16-bit PCM WAV format
- **Sample Rate**: 16kHz (configurable)
- **Voice Cloning**: Extracts voice characteristics from reference audio
- **Text Processing**: Automatically splits long text into manageable chunks

## Limitations

- Reference audio should be clear and of good quality
- Maximum text length is 500 characters per chunk
- Processing time depends on text length and model parameters
- Requires GPU for optimal performance

## Dependencies

- PlayDiffusion
- PyTorch
- boto3 (for S3/Spaces upload)
- scipy (for audio processing)
- requests (for HTTP requests)
# runpod-playvoice-clone
# runpod-playvoice-clone
