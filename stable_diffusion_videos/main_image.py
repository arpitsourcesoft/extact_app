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




# Endpoint for generating images
@app.route('/generate_images', methods=['POST'])
def generate_images_endpoint():
    # load the stable_diffusion model
    pipe = StableDiffusionWalkPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        safety_checker=None,
        vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cpu"),
        scheduler=LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
    ).to("cpu")

    # Enable memory efficient attention if xformers are available
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    try:
        if request.json is None:
            return jsonify({'error': 'No JSON in request'}), 400
        prompt = request.json.get('prompt')
        if prompt is None:
            return jsonify({'error': 'No prompt in JSON'}), 400
        seed = request.json.get('seed', random.randint(0, 9999999))
        
        # Generate and save image
        output_dir = Path('./dreams/images') # difine the output directory
        image_fpath = generate_images(pipe, prompt, seeds=[seed], num_inference_steps=50, guidance_scale=7.5, height=512, width=512, upsample=False, output_dir=output_dir)[0]
        response = send_file(image_fpath, mimetype='image/png', as_attachment=True)
        response.headers["Content-Disposition"] = "attachment; filename=generated_image.png"
        return response
    except Exception as e:
        return jsonify({'error': 'Failed to process the image. '+str(e)}), 500
    
# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)