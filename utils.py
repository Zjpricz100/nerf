from PIL import Image
import numpy as np
import os

def read_img(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    return img_array

def save_img(img, save_path):
    img_to_save = (img.clip(0, 1) * 255.0).astype(np.uint8)
    
    pil_image = Image.fromarray(img_to_save)
    pil_image.save(save_path)
    
    print(f"Reconstructed image saved as {save_path}")

def load_imgs_from_repo(repo_path):
    images_data = []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

    print(f"Attempting to load images from: {repo_path}")

    if not os.path.isdir(repo_path):
        print(f"Error: Path '{repo_path}' is not a valid directory.")
        return []

    for filename in os.listdir(repo_path):
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in valid_extensions:
            full_path = os.path.join(repo_path, filename)
            
            try:
                with Image.open(full_path) as img:
                    image_object = img.copy()
                    images_data.append(np.array(image_object))
                    
            except Exception as e:
                print(f"Error loading '{filename}': {e}")
        else:
            pass

    return images_data


