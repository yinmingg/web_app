from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle CORS issues
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Determine device and set precision accordingly
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    revision = "fp16"
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    revision = "fp16"
else:
    device = "cpu"
    dtype = torch.float32
    revision = None  # Use default revision for CPU

model_id = "CompVis/stable-diffusion-v1-4"

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        use_auth_token=True  
    )
except Exception as e:
    print("Error loading the model:", e)
    exit(1)

pipe = pipe.to(device)

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        # Generate the image based on the prompt.
        image = pipe(prompt).images[0]

        # Save image to a bytes buffer.
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        # Encode the image in base64 to send it as JSON.
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return jsonify({"image": img_str})
    except Exception as e:
        print("Error generating image:", e)
        return jsonify({"error": "Error generating image."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
