import os
import numpy as np
from PIL import Image

# Configuration
INPUT_DIR = "Microsoft_dataset/Original/train"  # Directory of .bytes files
OUTPUT_DIR = "ProcessedData/microsoft_images/_train"  
TARGET_WIDTH = 64
TARGET_HEIGHT = 784
TOTAL_PIXELS = TARGET_WIDTH * TARGET_HEIGHT  # 64x784 = 50,176 bytes

def create_malware_image(file_path):
    """Convert binary file to 64x784 grayscale image"""
    with open(file_path, "rb") as f:
        data = f.read()
    
    # 1. Replace ?? with 0xFF (paper specifies this)
    data = data.replace(b'??', b'\xff')
    
    # 2. Pad/truncate to exactly 50,176 bytes
    if len(data) < TOTAL_PIXELS:
        data += b'\x00' * (TOTAL_PIXELS - len(data))
    else:
        data = data[:TOTAL_PIXELS]
    
    # 3. Convert to numpy array and reshape
    byte_array = np.frombuffer(data, dtype=np.uint8)
    image_array = byte_array.reshape((TARGET_HEIGHT, TARGET_WIDTH))
    
    return Image.fromarray(image_array, mode='L')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process all .bytes files
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".bytes"):
        continue
    
    input_path = os.path.join(INPUT_DIR, filename)
    
    try:
        # Generate and save image
        img = create_malware_image(input_path)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.png")
        img.save(output_path)
        print(f"Processed: {filename} => {output_path}")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")