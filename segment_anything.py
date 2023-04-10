from segment_anything import build_sam, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
from grounded_sam import show_mask
import os
import torch
image = cv2.imread('example.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="checkpoints/sam_vit_h_4b8939.pth").to(device))
masks = mask_generator.generate(image)
segment_masks = []
for mask in masks:
    segment_masks.append(mask['segmentation'])

plt.figure(figsize=(10, 10))
plt.imshow(image)

for mask in segment_masks:
    show_mask(mask, plt.gca(), random_color=True)

plt.axis('off')
plt.savefig(os.path.join('outputs', "segment_example.jpg"), bbox_inches="tight")