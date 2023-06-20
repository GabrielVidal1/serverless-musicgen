from potassium import Potassium, Request, Response
import torch
from audiocraft.models import MusicGen
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

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGen.get_pretrained('melody', device=device)

    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    duration = request.json.get("duration", 8)
    samples = request.json.get("samples", 1)
    model = context.get("model")
    
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    result = gen(model, prompt, samples, duration)


    return Response(
        json = {"outputs": result}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()