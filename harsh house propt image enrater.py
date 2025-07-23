# ================================
# 🧠 INSTALL & IMPORT DEPENDENCIES
# ================================

# Required libraries installation for Colab environment
!pip install transformers diffusers lpips accelerate

# ----------------------------------
# 🔐 Authenticate Hugging Face Hub
# ----------------------------------
from huggingface_hub import notebook_login
notebook_login()  # Prompts for your Hugging Face token to access private models

# ----------------------------------
# 🔧 Core Libraries
# ----------------------------------
import torch  # PyTorch for tensor operations and deep learning
from transformers import CLIPTextModel, CLIPTokenizer  # For text encoding using CLIP
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler  # Components of Stable Diffusion
from tqdm.auto import tqdm  # Progress bar for loops
from torch import autocast  # Mixed precision inference
from PIL import Image  # Image loading and saving
from matplotlib import pyplot as plt  # For image display
import numpy  # Numerical operations
from torchvision import transforms as tfms  # Image preprocessing utilities

# ----------------------------------
# 🎥 Video Display (Jupyter/Colab)
# ----------------------------------
from IPython.display import HTML  # For embedding video in notebooks
from base64 import b64encode  # Encode video for HTML display

# ----------------------------------
# 🖥️ Set Device (GPU if available)
# ----------------------------------
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------
# 🧠 AI Clients and Utilities
# ----------------------------------
from openai import OpenAI  # GPT-4 API access
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP image captioning
from PIL import Image  # Re-imported (can remove one if desired)
import json  # JSON read/write for GPT structured output

# ----------------------------------
# 📁 Mount Google Drive (Colab only)
# ----------------------------------
from google.colab import drive
drive.mount('/content/drive')  # Gives access to your Google Drive files

# ===============================
# 🔑 OPENAI CLIENT INITIALIZATION
# ===============================

# Example (DO NOT hardcode in public repos):
# client = OpenAI(api_key="your-openai-api-key")

# 👇 Initialize OpenAI client — RECOMMENDED: Load key from environment variable or config file
client = OpenAI("")  # TODO: Replace with a secure way to load your API key, e.g., os.environ['OPENAI_API_KEY']

# ===============================
# 🖼️ INPUT IMAGE PATH
# ===============================
image_path = "image4.jpg"  # Path to the uploaded house image (can be relative or full path)

# ===============================
# 📸 STEP 1: IMAGE CAPTIONING USING BLIP
# ===============================
def get_caption(image_path):
    """
    Generate a natural language caption for the given house image using the BLIP model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Generated caption describing the image.
    """
    # Load the BLIP model and processor from Hugging Face
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()  # Set model to inference mode

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Generate caption using the model
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


# ===============================================
# 🧠 STEP 2: ANALYZE HOUSE ARCHITECTURE USING GPT-4
# ===============================================
def analyze_house(caption):
    """
    Uses GPT-4 to generate a structured analysis of a house image caption.
    The response includes architectural structure, visible regions, style suggestions, and a written summary.

    Args:
        caption (str): Caption describing the house image (from BLIP model)

    Returns:
        dict: Parsed JSON response with keys: structure, regions_detected, style_suggestions, and summary
    """

    # 👇 Build the system prompt with strict formatting instructions
    full_prompt = f"""
Analyze this image of a home exterior and return a detailed breakdown in structured JSON format with the following keys:

- structure: Include number of stories, roof type, and architectural style (e.g., Modern, Farmhouse).

- regions_detected: Detect and list all key visible components in this fixed array format: [wall, door, garage, roof, window, trim, light].
  For each detected region, provide its approximate geographic position within the image using one of the following labels:
    - top-left, top, top-right
    - center-left, center, center-right
    - bottom-left, bottom, bottom-right

  Example format:
    [
      {{"region": "wall", "position": "center"}},
      {{"region": "window", "position": "top-left"}},
      {{"region": "roof", "position": "top"}}
    ]

- style_suggestions: Based on the detected regions, provide at least 10 creative and realistic enhancement ideas for beautifying or subtly improving the home exterior. Each suggestion must include:
    - title: A short, descriptive title (e.g., “Colorful Front Door”).
    - prompt: A thoughtful and actionable design suggestion focusing on visual appeal, styling, or minor architectural updates (e.g., “Paint the front door a vibrant color to add a fun pop that enhances curb appeal and invites warmth.”). Suggestions may involve materials, textures, paint, trims, lighting, plants, or small structural changes, but should avoid drastic modifications.
    - target_region: One or more applicable regions from the detected list (e.g., "door", "porch", "roof"). Mix single-region and multi-region suggestions to provide variety.

- summary: Write a concise and descriptive summary of the house shown in the image. Include key visual and architectural details such as:
    - Number of stories and overall size
    - Architectural style (e.g., modern, colonial, farmhouse)
    - Roof type and materials
    - Prominent visible features such as porch, garage, windows, doors, columns, trim, etc.
    - General color scheme and design elements
    - Any unique or eye-catching details

The summary should be visually descriptive, easy to read, and give a clear understanding of the house's appearance and character. Keep it under 150 words.

Caption: \"{caption}\"

Return only valid JSON with keys: structure, regions_detected, style_suggestions, and summary.
The response must begin directly with a JSON object — do not include any additional explanation or text.
"""

    # Send the prompt to OpenAI's GPT-4 model
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            { "role": "system", "content": "You are a house architecture design expert." },
            { "role": "user", "content": full_prompt }
        ]
    )

    # 🔎 Print raw output from GPT for debugging
    reply = response.choices[0].message.content.strip()
    print("🔎 GPT RAW OUTPUT:\n", reply)

    # 🚨 If output doesn't start with JSON, raise an error
    if not reply.startswith("{"):
        raise ValueError("⚠️ GPT did not return JSON. You may want to review the caption or format.")

    # ✅ Try parsing JSON output
    try:
        return json.loads(reply)

    # ❌ Handle JSON parsing errors
    except json.JSONDecodeError:
        print("⚠️ Could not parse response. Raw output:\n", reply)
        raise


# ============================================
# 🚀 STEP 3: EXECUTION PIPELINE STARTS HERE
# ============================================
if __name__ == "__main__":
    # Step 1: Generate image caption using BLIP
    caption = get_caption(image_path)
    print("🖼️ Image Caption (BLIP):", caption)

    # Step 2: Get structured analysis from GPT-4
    structured_json = analyze_house(caption)
    print("\n📦 Final Structured Output:")
    print(json.dumps(structured_json, indent=2))  # Pretty-print the result

    # Step 3: Save GPT output as JSON file
    with open("house_output.json", "w") as f:
        json.dump(structured_json, f, indent=2)
    print("\n✅ Saved to 'house_output.json'")

# ==================================================
# 🎯 Extract Prompts for Visual Enhancement Suggestions
# ==================================================

# Extract 'prompt' field from each style suggestion
prompt_lines = [item["prompt"] for item in structured_json.get("style_suggestions", [])]

# Combine all prompts into a single string for processing
write = ""
for prom in prompt_lines:
    write += prom + "\n"
print(write)  # Optional: View the original prompt suggestions

# Clean up and format the prompt block
original_prompt = write.strip()
prompt_list = [line.strip() for line in original_prompt.split('\n') if line.strip()]
updated_prompts = prompt_list.copy()  # Copy for future use or modification

# ==================================================
# ✏️ FUNCTION TO UPDATE/ADD STYLE SUGGESTIONS (GPT-4)
# ==================================================
def gpt_reword_or_add(original_prompt_block, user_request):
    """
    Use GPT-4 to reword, modify, or add a suggestion to the original architectural prompts.

    Args:
        original_prompt_block (str): The list of original prompts as a block of text.
        user_request (str): The new instruction or desired change from the user.

    Returns:
        str: Updated list of style suggestions.
    """
    gpt_prompt = f"""
You are a helpful design assistant for house architecture suggestions.

Here are current home improvement suggestions (for a house exterior):
{original_prompt_block}

Now, the user wants to apply this update or change:
"{user_request}"

Please do the following:
1. If the suggestion already exists and matches the user request, update it if needed.
2. If it doesn’t exist, add a new suggestion related to the user input.
3. If there are multiple matches or ambiguity (e.g., multiple walls or styles), ask the user for clarification.
4. Keep all suggestions realistic and implementable in real-world home exteriors.
5. Return only the full updated suggestion list, clearly formatted as bullet points or lines.

Output the updated suggestion list only.
"""

    # Send to GPT-4
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a smart and practical house architecture expert."},
            {"role": "user", "content": gpt_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# ==================================================
# 🧠 HANDLE USER INTERACTION TO MODIFY SUGGESTIONS
# ==================================================

# Pass the original suggestions block
original_prompt_block = write.strip()

# Get user update via input (e.g., "Add plant suggestions around the garage")
user_request = input("ENTER THE UPDATES IN IMAGE...")

# Generate revised suggestions
updated_prompt_block = gpt_reword_or_add(original_prompt_block, user_request)

# Print updated suggestions
print(updated_prompt_block)


# ==================================================
# 🖼️ STEP 4: IMAGE GENERATION WITH STABLE DIFFUSION
# ==================================================

# --------------------------------------------------
# 📦 Load Pretrained Models
# --------------------------------------------------

# Autoencoder to decode latents into image space
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae",
    use_auth_token=True  # Ensure your Hugging Face token is set up
)

# Tokenizer & Text Encoder (CLIP) to convert prompt text into embeddings
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# UNet — the core generator that produces image latents from noise
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    use_auth_token=True
)

# Scheduler controls the noise removal process during denoising (LMS = Linear Multistep)
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)

# --------------------------------------------------
# 🚀 Send Models to GPU (if available)
# --------------------------------------------------
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# --------------------------------------------------
# 📝 Prepare Prompt and Inference Settings
# --------------------------------------------------
prompt = [updated_prompt_block]  # List format required for batch processing
height = 512                     # Output image height
width = 768                      # Output image width
num_inference_steps = 100        # Number of denoising steps
guidance_scale = 10.0            # Higher = more focus on prompt guidance
batch_size = 1
generator = torch.manual_seed(8)  # Set random seed for reproducibility

# --------------------------------------------------
# 🔠 Convert Prompt Text → Embeddings
# --------------------------------------------------

# Tokenize prompt
text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)

# Encode text into embeddings
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# Classifier-free guidance needs empty (unconditional) embeddings too
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
)

with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

# Stack unconditional + conditional embeddings for guidance
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# --------------------------------------------------
# 🌫️ Initialize Random Latents (Noise Vector)
# --------------------------------------------------
scheduler.set_timesteps(num_inference_steps)

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator
).to(torch_device)

# Scale latents to match sigma space
latents = latents * scheduler.sigmas[0]

# --------------------------------------------------
# 🔁 Denoising Loop (Stable Diffusion Core)
# --------------------------------------------------
with autocast("cuda"):  # Enable mixed precision for faster inference
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # Expand latents to apply classifier-free guidance (2 batches)
        latent_model_input = torch.cat([latents] * 2)

        # Normalize with current sigma value
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

        # Predict noise for current timestep
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            )["sample"]

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Update latents using predicted noise
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

# --------------------------------------------------
# 🧯 Decode Latents → RGB Image
# --------------------------------------------------
latents = latents / 0.18215  # Scale factor from model training

with torch.no_grad():
    image = vae.decode(latents).sample  # Output is in [-1, 1]

# --------------------------------------------------
# 🖼️ Post-Process and Convert to PIL
# --------------------------------------------------

# Convert tensor to [0, 1] range and to uint8 format
image = (image / 2 + 0.5).clamp(0, 1)  # Normalize
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")  # Convert to 8-bit

# Convert to PIL Image
pil_images = [Image.fromarray(image) for image in images]

# Display first generated image
pil_images[0]
