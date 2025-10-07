import os
import csv
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import re 

# ==== Model Setup ====
model_id = "Salesforce/blip2-flan-t5-xl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading BLIP-2 model...")
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
).to(device)

# ==== Input directory ====
image_dir = "/Users/akhilkakarla/Desktop/profwang/mapillary/test_dir"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# ==== Output file ====
output_file = "blip2_image_ratings_wealth_COT.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "description", "wealth_rating"])

    # ==== Loop over images ====
    for img_name in image_files:
        try:
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert("RGB")

            # === Description ===
            desc_input = processor(images=image, text="Describe what you see in the image.", return_tensors="pt").to(device)
            desc_output = model.generate(**desc_input, max_new_tokens=50)
            description = processor.batch_decode(desc_output, skip_special_tokens=True)[0]

            # === Rating ===
            prompt = "Rate how wealthy this scene looks on a scale from 1 (not very wealthy and looks poor) to 10 (very wealthy and looks lavish). First, explain your reasoning based on what's visible in the image. Then, give your final rating as a single number."
            
            rating_input = processor(images=image, text=prompt
            , return_tensors="pt").to(device)
            
            # rating_output = model.generate(**rating_input, max_new_tokens=10)
            rating_output = model.generate(
                **rating_input,
                max_new_tokens=5,
                do_sample=True,
                top_k=20,
                temperature=0.8
            )
            
            rating = processor.batch_decode(rating_output, skip_special_tokens=True)[0]
            match = re.search(r"\b([1-9]|10)\b", rating)
            rating_number = int(match.group(1)) if match else 0

            print(f"{img_name}: {rating} — {description}")
            writer.writerow([img_name, description, rating])

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
