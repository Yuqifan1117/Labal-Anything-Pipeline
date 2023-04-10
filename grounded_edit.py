import json
import torch
import os
import argparse
import numpy as np
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# stable diffusion
from diffusers import StableDiffusionInpaintPipeline

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# clip to filter ambuguite regions
import clip
# utils
from grounded_sam import load_image, load_model, show_box
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.ops import box_convert

# load open-world detection models
@torch.no_grad()
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (num_query, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (num_query, 4)
    logits.shape[0]
    # filter output box with > box_threshold (match with caption) Language-Guided Query Selection
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        # convert ids to token (filter stop-words in captions to get tokens)
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, pred_phrases, torch.Tensor(scores)

@torch.no_grad()
def mask_fusion(final_mask):
    final_mask = np.sum(final_mask, axis=0)
    mask_image = Image.fromarray(np.uint8(final_mask))
    return mask_image

@torch.no_grad()
def generate_masks_with_grounding(image_pil, boxes):
    mask = np.zeros_like(image_pil)
    x0, y0, x1, y1 = boxes
    mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask

@torch.no_grad()
def retriev(elements, search_text):
    preprocessed_images = [clip_preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = clip_model.encode_image(stacked_images)
    text_features = clip_model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)
# load sd image for image edit
MY_TOKEN = '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'
# # LOW_RESOURCE = False 
# # NUM_DIFFUSION_STEPS = 50
# # GUIDANCE_SCALE = 7.5
# # MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable_inpaint = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", revision="fp16",
    torch_dtype=torch.float16, use_auth_token=MY_TOKEN).to(device)
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is

# image = ldm_stable_inpaint(prompt=prompt, image=image, mask_image=mask_image).images[0]
# tokenizer = ldm_stable.tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--edit_prompt", type=str, required=True, help="edit prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    edit_prompt = args.edit_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image and process with PIL
    image_pil, image = load_image(image_path)
    # load image for visualization
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # Load CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    # run grounding dino model
    total_boxes = []
    total_predphrases = []
    total_scores = []
    boxes_filt, pred_phrases, pred_scores = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)
    for valid_box, valid_phrase, valid_score in zip(boxes_filt, pred_phrases, pred_scores):
        total_boxes.append(valid_box)
        total_predphrases.append(valid_phrase)
        total_scores.append(valid_score)
    for i in range(len(total_boxes)):
        size = image_pil.size
        H, W = size[1], size[0]
        boxes_filt = total_boxes[i] * torch.Tensor([W, H, W, H])
        boxes_filt[:2] -= boxes_filt[2:] / 2
        boxes_filt[2:] += boxes_filt[:2]
        boxes_filt = boxes_filt.cpu()
        total_boxes[i] = boxes_filt

    # objects mask with bounding box
    if len(total_boxes) > 0:
        # Cut out all masks and select the most correct bounding box with clip
        cropped_boxes = []
        for box, pred_phrase in zip(total_boxes, total_predphrases):
            cropped_boxes.append(image_pil.crop(box.tolist()))
        idx = torch.argmax(retriev(cropped_boxes, text_prompt)).item()
        box = total_boxes[idx]
        image_mask = generate_masks_with_grounding(image_pil, box.numpy())
        mask_image = Image.fromarray(image_mask)    
        mask_image.save(os.path.join(output_dir, f'mask_{pred_phrase}.jpg'))
        image_source_for_inpaint = image_pil.resize((512, 512))
        image_mask_for_inpaint = mask_image.resize((512, 512))
        image_inpainting = ldm_stable_inpaint(prompt=edit_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
        image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
        image_inpainting.save(os.path.join(output_dir, f'edit_result_{pred_phrase}.jpg'))
    
    # background edit with segmentation mask
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks_list = []
    if len(total_boxes) > 0:
        idx = torch.argmax(torch.stack(total_scores)).item()
        boxes_filt = total_boxes[idx]
        pred_phrase = total_predphrases[idx]
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
        mask_image = Image.fromarray(masks[0][0].cpu().numpy())
        phrase, logit = pred_phrase.split('(')
        logit = logit[:-1] 
        mask_image.save(os.path.join(output_dir, f'mask_{phrase}_{logit}.jpg'))
        image_source_for_inpaint = image_pil.resize((512, 512))
        image_mask_for_inpaint = mask_image.resize((512, 512))
        image_inpainting = ldm_stable_inpaint(prompt=edit_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
        image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
        image_inpainting.save(os.path.join(output_dir, f'edit_result_{phrase}_{logit}.jpg'))
        

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)

    # non-ambiguity annotation
    for boxes_filt, pred_phrases in zip(total_boxes, total_predphrases):
    # for valid_box, valid_phrase in zip(boxes_filt, pred_phrases):
        show_box(boxes_filt.numpy(), plt.gca(), pred_phrases, random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "test.jpg"), bbox_inches="tight")

