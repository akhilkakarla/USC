import os
import requests
import time
import csv

# --- CONFIGURATION ---
ACCESS_TOKEN = "MLY|10037581989607326|d73edd8d9f2bcba9e1d28b0921b9df03"  # Replace with your actual token
SAVE_DIR = "Austin_SVI_Images_3"
CSV_FILE = "austin_svi_coordinates.csv"
TOTAL_IMAGES = 2000
BBOX = "-97.9000,30.1000,-97.4500,30.5000"  # Bounding box around Austin

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Prepare CSV file
csv_file = open(CSV_FILE, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image ID", "Latitude", "Longitude", "Image File"])

# API base URL
BASE_URL = "https://graph.mapillary.com/images"
FIELDS = "id,geometry,thumb_1024_url"
LIMIT_PER_PAGE = 1000  # max per request
downloaded = 0
cursor = None

# --- DOWNLOAD LOOP ---
while downloaded < TOTAL_IMAGES:
    params = {
        "access_token": ACCESS_TOKEN,
        "bbox": BBOX,
        "fields": FIELDS,
        "limit": LIMIT_PER_PAGE,
    }
    if cursor:
        params["after"] = cursor

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print("Error:", response.text)
        break

    data = response.json()
    images = data.get("data", [])
    print(f"Retrieved {len(images)} images...")

    for image in images:
        if downloaded >= TOTAL_IMAGES:
            break

        image_id = image.get("id")
        coords = image.get("geometry", {}).get("coordinates", [])
        image_url = image.get("thumb_1024_url")

        if not image_url or len(coords) != 2:
            continue

        # Download image
        img_file_name = f"{image_id}.jpg"
        img_path = os.path.join(SAVE_DIR, img_file_name)
        try:
            img_data = requests.get(image_url).content
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            # Save metadata
            csv_writer.writerow([image_id, coords[1], coords[0], img_file_name])
            downloaded += 1
            print(f"[{downloaded}] Saved {img_file_name}")
        except Exception as e:
            print(f"Failed to download {image_id}: {e}")

    # Handle pagination
    cursor = data.get("paging", {}).get("next", None)
    if not cursor:
        print("No more images to fetch.")
        break

    time.sleep(1)  # Optional: Be gentle with the API

# Cleanup
csv_file.close()
print(f"Download complete: {downloaded} images saved in '{SAVE_DIR}'")
print(f"Coordinates saved in '{CSV_FILE}'")

