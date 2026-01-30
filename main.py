"""
Image Segmentation using Hugging Face Inference API
Segmentation of clothes and body parts using segformer_b3_clothes model
"""

import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import base64
import io
import time
import csv
from datetime import datetime

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# =============================================================================
# Configuration
# =============================================================================

# TODO: Modifiez ces valeurs selon votre configuration
image_dir = r"top_influenceurs_2024\IMG"
output_dir = r"top_influenceurs_2024\Mask"
max_images = 50

# IMPORTANT: Remplacez par votre véritable token API Hugging Face
# Ne partagez jamais votre token publiquement.
if load_dotenv is None:
    raise ImportError("python-dotenv is required to load API_TOKEN from .env. Install it with: pip install python-dotenv")

load_dotenv()
api_token = os.getenv("API_TOKEN")

# =============================================================================
# API Setup
# =============================================================================

API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
headers = {
    "Authorization": f"Bearer {api_token}"
}

# =============================================================================
# Class Mapping and Colors for Segmentation Labels
# =============================================================================

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

# Color palette for visualization (RGB)
CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 255, 0),    # Hat - Yellow
    2: (255, 165, 0),    # Hair - Orange
    3: (255, 0, 255),    # Sunglasses - Magenta
    4: (255, 0, 0),      # Upper-clothes - Red
    5: (0, 0, 255),      # Skirt - Blue
    6: (0, 255, 0),      # Pants - Green
    7: (128, 0, 128),    # Dress - Purple
    8: (255, 192, 203),  # Belt - Pink
    9: (255, 140, 0),    # Left-shoe - Dark Orange
    10: (255, 140, 0),   # Right-shoe - Dark Orange
    11: (135, 206, 235), # Face - Sky Blue
    12: (128, 128, 128), # Left-leg - Gray
    13: (128, 128, 128), # Right-leg - Gray
    14: (100, 149, 237), # Left-arm - Cornflower Blue
    15: (100, 149, 237), # Right-arm - Cornflower Blue
    16: (210, 180, 140), # Bag - Tan
    17: (0, 128, 128),   # Scarf - Teal
}

CLASS_NAMES_FR = {
    0: "Arrière-plan",
    1: "Chapeau",
    2: "Cheveux",
    3: "Lunettes de soleil",
    4: "Haut (vêtement)",
    5: "Jupe",
    6: "Pantalon",
    7: "Robe",
    8: "Ceinture",
    9: "Chaussure gauche",
    10: "Chaussure droite",
    11: "Visage",
    12: "Jambe gauche",
    13: "Jambe droite",
    14: "Bras gauche",
    15: "Bras droit",
    16: "Sac",
    17: "Écharpe",
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size


def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            combined_mask[mask_array > 0] = 0  # Class ID for Background is 0

    return combined_mask


def get_image_paths(directory, max_count=None):
    """
    Get list of image paths from a directory.

    Args:
        directory (str): Path to the directory containing images.
        max_count (int, optional): Maximum number of images to return.

    Returns:
        list: List of image file paths.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    
    if not os.path.exists(directory):
        print(f"Le dossier '{directory}' n'existe pas.")
        return image_paths
    
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(directory, filename))
            if max_count and len(image_paths) >= max_count:
                break
    
    return image_paths


def segment_single_image(image_path):
    """
    Segment a single image using the Hugging Face API.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Segmentation mask, or None if failed.
    """
    try:
        # Get image dimensions
        width, height = get_image_dimensions(image_path)
        
        # Determine content type
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            content_type = "image/jpeg"
        elif ext == '.png':
            content_type = "image/png"
        else:
            content_type = "image/jpeg"
        
        # Read image in binary mode
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Set headers with content type
        request_headers = headers.copy()
        request_headers["Content-Type"] = content_type
        
        # Send request to API
        response = requests.post(API_URL, headers=request_headers, data=image_data)
        response.raise_for_status()
        
        # Parse response
        results = response.json()
        
        # Create combined mask
        mask = create_masks(results, width, height)
        
        return mask
        
    except requests.exceptions.HTTPError as e:
        print(f"Erreur HTTP pour {image_path}: {e}")
        if response.status_code == 503:
            print("Le modèle est en cours de chargement. Réessayez dans quelques secondes.")
        return None
    except Exception as e:
        print(f"Erreur pour {image_path}: {e}")
        return None


def segment_images_batch(list_of_image_paths, delay=1.0):
    """
    Segment a list of images using the Hugging Face API.

    Args:
        list_of_image_paths (list): List of paths to image files.
        delay (float): Delay in seconds between API calls.

    Returns:
        list: List of segmentation masks (NumPy arrays).
              Contains None if an image could not be processed.
    """
    batch_segmentations = []
    
    for image_path in tqdm(list_of_image_paths, desc="Segmentation"):
        mask = segment_single_image(image_path)
        batch_segmentations.append(mask)
        
        # Pause between API calls to avoid rate limiting
        time.sleep(delay)
    
    return batch_segmentations


def save_mask(mask, output_path):
    """
    Save a segmentation mask as an image file.

    Args:
        mask (np.ndarray): Segmentation mask.
        output_path (str): Path to save the mask image.
    """
    mask_image = Image.fromarray(mask)
    mask_image.save(output_path)


def create_colored_mask(mask):
    """
    Convert a class-index mask to a colored RGB image.

    Args:
        mask (np.ndarray): Segmentation mask with class indices.

    Returns:
        np.ndarray: RGB colored mask.
    """
    height, width = mask.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    
    return colored


def create_overlay(original_image, mask, alpha=0.5):
    """
    Create an overlay of the colored segmentation on the original image.

    Args:
        original_image (PIL.Image): Original image.
        mask (np.ndarray): Segmentation mask.
        alpha (float): Transparency for the overlay (0-1).

    Returns:
        np.ndarray: Blended image with segmentation overlay.
    """
    original_array = np.array(original_image.convert('RGB'))
    colored_mask = create_colored_mask(mask)
    
    # Only blend where there's a non-background class
    blended = original_array.copy()
    non_bg_mask = mask > 0
    blended[non_bg_mask] = (
        (1 - alpha) * original_array[non_bg_mask] + 
        alpha * colored_mask[non_bg_mask]
    ).astype(np.uint8)
    
    return blended


def extract_clothing_piece(original_image, mask, class_id):
    """
    Extract a specific clothing piece from the image.

    Args:
        original_image (PIL.Image): Original image.
        mask (np.ndarray): Segmentation mask.
        class_id (int): Class ID to extract.

    Returns:
        PIL.Image: Image with only the specified clothing piece (RGBA with transparency).
    """
    original_array = np.array(original_image.convert('RGBA'))
    piece_mask = (mask == class_id).astype(np.uint8) * 255
    
    result = original_array.copy()
    result[:, :, 3] = piece_mask  # Set alpha channel
    
    return Image.fromarray(result)


def save_colored_visualization(image_path, mask, colored_dir, overlay_dir):
    """
    Save colored mask and overlay visualization.

    Args:
        image_path (str): Path to original image.
        mask (np.ndarray): Segmentation mask.
        colored_dir (str): Directory to save colored mask.
        overlay_dir (str): Directory to save overlay.
    """
    original = Image.open(image_path)
    name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save colored mask
    colored_mask = create_colored_mask(mask)
    colored_path = os.path.join(colored_dir, f"colored_{name}.png")
    Image.fromarray(colored_mask).save(colored_path)

    # Save overlay
    overlay = create_overlay(original, mask, alpha=0.5)
    overlay_path = os.path.join(overlay_dir, f"overlay_{name}.png")
    Image.fromarray(overlay).save(overlay_path)
    
    return colored_path, overlay_path


def get_detected_classes(mask):
    """
    Get list of detected classes in a segmentation mask.

    Args:
        mask (np.ndarray): Segmentation mask.

    Returns:
        list: List of tuples (class_id, class_name_fr, color, pixel_count).
    """
    unique_classes = np.unique(mask)
    detected = []
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        pixel_count = np.sum(mask == class_id)
        class_name = CLASS_NAMES_FR.get(class_id, f"Classe {class_id}")
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        detected.append((class_id, class_name, color, pixel_count))
    
    # Sort by pixel count (largest first)
    detected.sort(key=lambda x: x[3], reverse=True)
    return detected


def create_legend_image(detected_classes, height=400, width=200):
    """
    Create a legend image showing detected classes with their colors.

    Args:
        detected_classes (list): List from get_detected_classes().
        height (int): Height of legend image.
        width (int): Width of legend image.

    Returns:
        np.ndarray: RGB legend image.
    """
    from PIL import ImageDraw, ImageFont
    
    legend = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    y_offset = 10
    box_size = 20
    spacing = 30
    
    draw.text((10, y_offset), "Vêtements détectés:", fill=(0, 0, 0), font=font)
    y_offset += spacing
    
    for class_id, name, color, pixel_count in detected_classes:
        # Draw color box
        draw.rectangle([10, y_offset, 10 + box_size, y_offset + box_size], fill=color, outline=(0, 0, 0))
        # Draw label
        draw.text((40, y_offset + 2), name, fill=(0, 0, 0), font=font)
        y_offset += spacing
        
        if y_offset > height - 30:
            break
    
    return np.array(legend)


def display_segmented_images_batch(original_image_paths, segmentation_masks, max_display=10):
    """
    Display original images, colored masks, and overlays side by side (3 panels).

    Args:
        original_image_paths (list): List of paths to original images.
        segmentation_masks (list): List of segmented masks (NumPy arrays).
        max_display (int): Maximum number of images to display.
    """
    num_images = min(len(original_image_paths), len(segmentation_masks), max_display)
    
    if num_images == 0:
        print("Aucune image à afficher.")
        return
    
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 5 * num_images))
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        image_path = original_image_paths[i]
        mask = segmentation_masks[i]
        original_image = Image.open(image_path)
        
        # Panel 1: Original image
        axes[i][0].imshow(original_image)
        axes[i][0].set_title(f"Original: {os.path.basename(image_path)}")
        axes[i][0].axis('off')
        
        if mask is not None:
            # Panel 2: Colored segmentation mask with legend
            colored_mask = create_colored_mask(mask)
            axes[i][1].imshow(colored_mask)
            
            # Get and display detected classes
            detected = get_detected_classes(mask)
            legend_text = "Détecté: " + ", ".join([d[1] for d in detected[:5]])
            if len(detected) > 5:
                legend_text += "..."
            axes[i][1].set_title(legend_text, fontsize=9)
            axes[i][1].axis('off')
            
            # Panel 3: Overlay on original
            overlay = create_overlay(original_image, mask, alpha=0.5)
            axes[i][2].imshow(overlay)
            axes[i][2].set_title("Superposition")
            axes[i][2].axis('off')
        else:
            axes[i][1].text(0.5, 0.5, "Échec de la segmentation", 
                           ha='center', va='center', fontsize=12)
            axes[i][1].set_title("Erreur")
            axes[i][1].axis('off')
            axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Check API token
    if not api_token:
        print("=" * 60)
        print("ATTENTION : Vous devez configurer votre token API Hugging Face !")
        print("1. Créez un compte sur https://huggingface.co/")
        print("2. Allez dans Settings -> Access Tokens")
        print("3. Créez un nouveau token (rôle 'read')")
        print("4. Ajoutez API_TOKEN=... dans un fichier .env à la racine du projet")
        print("=" * 60)
    else:
        # Get image paths
        image_paths = get_image_paths(image_dir, max_images)
        
        if not image_paths:
            print(f"Aucune image trouvée dans '{image_dir}'.")
        else:
            print(f"{len(image_paths)} image(s) trouvée(s) à traiter.")
            
            # Process images + Timer to stress test
            print("\nTraitement des images en cours...")
            t0 = time.perf_counter()
            batch_results = segment_images_batch(image_paths, delay=1.0)
            t1 = time.perf_counter()
            processed_count = sum(1 for m in batch_results if m is not None)
            total_s = t1 - t0
            avg_s = (total_s / processed_count) if processed_count else 0.0
            print(f"\nTemps total pour {processed_count}/{len(image_paths)} images : {total_s:.2f} s")
            print(f"Temps moyen par image (réussies) : {avg_s:.2f} s")
            
            os.makedirs(output_dir, exist_ok=True)
            run_dir = os.path.join(output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            masks_dir = os.path.join(run_dir, "masks")
            colored_dir = os.path.join(run_dir, "colored")
            overlays_dir = os.path.join(run_dir, "overlays")
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(colored_dir, exist_ok=True)
            os.makedirs(overlays_dir, exist_ok=True)
            manifest_path = os.path.join(run_dir, "results.csv")
            
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "source_image",
                        "mask_path",
                        "colored_path",
                        "overlay_path",
                        "detected_classes",
                    ],
                )
                writer.writeheader()
        
                for i, (image_path, mask) in enumerate(zip(image_paths, batch_results)):
                    if mask is None:
                        writer.writerow(
                            {
                                "source_image": image_path,
                                "mask_path": "",
                                "colored_path": "",
                                "overlay_path": "",
                                "detected_classes": "",
                            }
                        )
                        continue
        
                    filename = os.path.basename(image_path)
                    name, _ = os.path.splitext(filename)
        
                    mask_filename = f"mask_{name}.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    save_mask(mask, mask_path)
                    print(f"Masque sauvegardé : {mask_path}")
        
                    colored_path, overlay_path = save_colored_visualization(image_path, mask, colored_dir, overlays_dir)
                    print(f"Masque coloré : {colored_path}")
                    print(f"Overlay : {overlay_path}")
        
                    detected = get_detected_classes(mask)
                    detected_names = [d[1] for d in detected]
                    if detected_names:
                        print(f"  → Vêtements détectés : {', '.join(detected_names)}")
        
                    writer.writerow(
                        {
                            "source_image": image_path,
                            "mask_path": mask_path,
                            "colored_path": colored_path,
                            "overlay_path": overlay_path,
                            "detected_classes": ";".join(detected_names),
                        }
                    )
        
            print(f"\nRésultats organisés dans : {run_dir}")
            print(f"Fichier récapitulatif : {manifest_path}")
            
            # Display results
            print("\nAffichage des résultats...")
            display_segmented_images_batch(image_paths, batch_results)
            
            print("\nTraitement terminé !")
