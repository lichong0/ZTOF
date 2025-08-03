import os
from PIL import Image
import numpy as np
import torch
import json
import clip
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

# Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
tt_scale = 2.0

# ------------------------------
# Function: build zero-shot text classifier
# ------------------------------
def zeroshot_classifier(model, textnames, json_file, device):
    with open(json_file, "r") as f:
        templates = json.load(f)

    zeroshot_weights = []

    with torch.no_grad():
        for i in tqdm(range(len(textnames)), desc="Building classifier"):
            texts = templates[textnames[i]]  # list of prompts
            if i == 0:
                print(f"Prompt templates for first class '{textnames[i]}':\n", texts)

            label = f"a photo of a {textnames[i].replace('_', ' ')}."
            label_tokens = clip.tokenize(label, truncate=True).to(device)
            label_embedding = model.encode_text(label_tokens)
            label_embedding = label_embedding / label_embedding.norm(dim=-1, keepdim=True)

            texts_tensor = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = model.encode_text(texts_tensor)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            weight = class_embeddings @ label_embedding.T
            weight = (weight * tt_scale).softmax(dim=0)
            class_embedding = (class_embeddings * weight).sum(dim=0)
            class_embedding = class_embedding / class_embedding.norm()

            zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device).float()  # shape: [512, num_classes]
    return zeroshot_weights

# ------------------------------
# main
# ------------------------------

# Loading CLIP Models
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# image path
image_path = "/root/shared-nvme/lichong/ZTOF/data/pet/oxford-iiit-pet/images"

# Category name (must correspond to the key in cupl.json)
textnames = ["Russian Blue", "newfoundland", "japanese chin"]
template_json = "pet.json"

# Image Preprocessing
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

# Image Feature Extraction
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Text Feature Generation
zeroshot_weights = zeroshot_classifier(model, textnames, template_json, device)

# Similarity calculation
with torch.no_grad():
    similarity = (image_features @ zeroshot_weights).cpu().numpy()  # shape: [1, num_classes]


print("similarity:", similarity)

# ------------------------------
# visualization
# ------------------------------

import matplotlib.offsetbox as offsetbox

# ------------------------------

fig, ax = plt.subplots(figsize=(10, 2))
im = ax.imshow(similarity, cmap="Greens", interpolation="nearest", aspect='auto')
plt.colorbar(im, ax=ax)

ax.set_xlabel("Class Names")

ax.set_xticks(range(len(textnames)))
ax.set_xticklabels(textnames, rotation=0)

ax.set_yticks([0])
ax.set_yticklabels([""])

imagebox = offsetbox.OffsetImage(image.resize((76, 75)), zoom=1)
ab = offsetbox.AnnotationBbox(imagebox, ( -0.5, 0), frameon=False, box_alignment=(1, 0.5))
ax.add_artist(ab)

plt.tight_layout()
plt.savefig("image.png", bbox_inches='tight')
plt.show()
