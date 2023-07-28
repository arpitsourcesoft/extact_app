# from fastapi import Response
from flask import Flask, jsonify, request, send_file, Response
# import flask
from werkzeug.utils import secure_filename
import os
import random
import librosa
from io import BytesIO
# from matplotlib import pyplot as plt
from stable_diffusion_videos import StableDiffusionWalkPipeline, generate_images, get_timesteps_arr
from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import torch
import soundfile as sf
# import numpy as np
from pathlib import Path

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'audio_files/' # directory to save uploads
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'} # set of allowed extensions



# Endpoint for generate video
@app.route('/generate_video', methods=['POST'])
def generate_video_endpoint():
    # load the stable_diffusion model
    pipe = StableDiffusionWalkPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        safety_checker=None,
        vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda"),
        scheduler=LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
    ).to("cuda")

    # Enable memory efficient attention if xformers are available
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # If the file is of an allowed type, process it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Try to save the uploaded file
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': 'Failed to save uploaded file. '+str(e)}), 500

        try:
            # Load the audio file and process it
            audio = filepath
            y, sr = librosa.load(audio, offset=1, duration=15)
            T = get_timesteps_arr(audio_data_to_buffer(y, sr), 0, 10, fps=24, margin=1, smooth=0)
        except Exception as e:
            return jsonify({'error': 'Failed to process the audio file. '+str(e)}), 500

        try:
            # Obtain the prompts and seeds from the request
            prompt_a = request.form.get('prompt_a')
            prompt_b = request.form.get('prompt_b')
            seed_a = request.form.get('seed_a', random.randint(0, 9999999))
            seed_b = request.form.get('seed_b', random.randint(0, 9999999))

            output_dir = Path('./dreams/images')  # Set the directory for saving generated images

            # Generate the video
            video_filepath = pipe.walk(prompts=[prompt_a, prompt_b], seeds=[seed_a, seed_b], num_interpolation_steps=int(10 * 12), output_dir=output_dir, fps=10, num_inference_steps=50, guidance_scale=7.5, height=512, width=512, upsample=False, batch_size=1, audio_filepath=audio, audio_start_sec=1, margin=1, smooth=0)
        except Exception as e:
            return jsonify({'error': 'Failed to generate the video. '+str(e)}), 500

        try:
            # Send the video as a response with the appropriate headers
            with open(video_filepath, "rb") as f:
                video = f.read()
            response = Response(video, mimetype='video/mp4')
            response.headers.set('Content-Disposition', 'attachment', filename='generated_video.mp4')
            return response
        except Exception as e:
            return jsonify({'error': 'Failed to return the video file. '+str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
# Define a function to convert audio data to a buffer
def audio_data_to_buffer(y, sr):
    try:
        audio_filepath = BytesIO()
        audio_filepath.name = 'audio.wav'
        sf.write(audio_filepath, y, samplerate=sr, format='WAV')  # Write the audio data to the buffer using the provided sample rate
        
        # Reset the buffer position to the beginning
        audio_filepath.seek(0)
        return audio_filepath
    except Exception as e:
        print(f"An error occurred in audio_data_to_buffer: {e}")
        return None

# Define a function to check if a file is of an allowed type
def allowed_file(filename):
    try:
        # Check if the filename has a dot ('.') indicating an extension, and the extension (everything after the last dot) is in the set of allowed extensions
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    except Exception as e:
        print(f"An error occurred in allowed_file: {e}")
        return False
       
# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)