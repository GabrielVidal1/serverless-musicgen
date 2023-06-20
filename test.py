# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import base64
import requests

model_inputs = {
        "prompt": "drum and bass beat with intense percussions",
        "duration": 2,
        "samples": 1
    }

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())

for i, output in enumerate(res.json().get("outputs")):
    with open(f"{i}.wav", "wb") as f:
        f.write(base64.b64decode(output.get("audio")))