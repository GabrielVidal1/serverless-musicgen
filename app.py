from transformers import pipeline
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import base64


def gen(model: MusicGen, prompt:str, samples=1, duration=8) :
    model.set_generation_params(duration=duration) 

    wav = model.generate([prompt] * samples)  # generates 3 samples.

    results = []
    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        path = audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f'Saving {path} with prompt: {prompt}')
        # read the file and convert it to base64
        with open(path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read())
            results.append({
                "prompt": prompt,
                "audio": encoded_string
            })
    return results
            

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGen.get_pretrained('melody', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    duration = model_inputs.get('duration', 8)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = gen(model, prompt, duration)

    # Return the results as a dictionary
    return result
