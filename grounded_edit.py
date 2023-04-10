import json
import torch
import os
import argparse
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# stable diffusion
from diffusers import StableDiffusionInpaintPipeline

# utils
from grounded_sam import load_image, load_model
import matplotlib.pyplot as plt

# load open-world detection models
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
    for logit, box in zip(logits_filt, boxes_filt):
        # convert ids to token (filter stop-words in captions to get tokens)
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
# load sd image for image edit
# MY_TOKEN = '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'
# # LOW_RESOURCE = False 
# # NUM_DIFFUSION_STEPS = 50
# # GUIDANCE_SCALE = 7.5
# # MAX_NUM_WORDS = 77
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# ldm_stable_inpaint = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", revision="fp16",
#     torch_dtype=torch.float16, use_auth_token=MY_TOKEN).to(device)
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
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image and process with PIL
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)
    print(boxes_filt)
    print(pred_phrases)


        # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # non-ambiguity annotation
    for valid_box, valid_phrase in zip(valid_boxes, valid_phrases):
        show_box(valid_box.numpy(), plt.gca(), valid_phrase, random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "annotation_output.jpg"), bbox_inches="tight")

