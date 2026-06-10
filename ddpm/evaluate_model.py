import argparse
import pandas as pd
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def calculate_accuracy(csv_path, images_dir, device=None):
    device = device or auto_device()
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for test set
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    print(f"Found {len(test_df)} images in test set.")

    # Get unique classes
    classes = sorted(df['label'].unique())
    # Remove 'unknown' if present, or keep it if it's a valid class to predict
    # Usually we only care about specific classes. Let's keep it for now but maybe warn.
    print(f"Classes: {classes}")

    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    correct = 0
    total = 0
    missing = 0

    print("Starting evaluation...")
    # Prepare text inputs once (descriptions of classes)
    # prompt_templates = [f"a satellite image of a {c}" for c in classes]
    # Actually, let's use the raw class names or simple templates.
    text_inputs = processor(text=[f"a satellite image of a {c}" for c in classes], return_tensors="pt", padding=True).to(device)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['filename']
        true_label = row['label']
        
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            # If the image doesn't exist, we can't evaluate it.
            # In a real scenario, we might generate it here.
            # For now, we'll skip and count as missing.
            missing += 1
            continue

        try:
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            
            # We need to combine inputs. 
            # CLIPProcessor can make this easier but let's do manual forward for efficiency if needed.
            # Actually, let's just use the model.get_text_features and get_image_features
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                text_features = model.get_text_features(**text_inputs)
                
                # Normalize
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # Similarity
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                probs = logits_per_image.softmax(dim=1)
                
                prediction_idx = probs.argmax().item()
                predicted_label = classes[prediction_idx]
                
                if predicted_label == true_label:
                    correct += 1
                
                total += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            missing += 1

    if total == 0:
        print("No images found to evaluate.")
        return

    accuracy = (correct / total) * 100
    print(f"\nResults:")
    print(f"Total Evaluated: {total}")
    print(f"Missing Images: {missing}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy (OA): {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot Classification Accuracy")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(REPO_ROOT, 'RSICD_optimal', 'dataset_rsicd.csv'),
                        help="Path to dataset CSV")
    parser.add_argument("--images_dir", type=str,
                        default=os.path.join(REPO_ROOT, 'RSICD_optimal', 'RSICD_images'),
                        help="Path to images directory (generated or ground truth)")
    parser.add_argument("--device", type=str, default=None,
                        choices=[None, 'cuda', 'mps', 'cpu'],
                        help="Compute device (auto if omitted).")

    args = parser.parse_args()

    calculate_accuracy(args.csv, args.images_dir, device=args.device)
