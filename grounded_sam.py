import argparse
import json
import os
import copy
import nltk
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# caption anything
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def generate_caption(raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
def generate_tags(raw_text):
    # generate specific categories in the caption
    tags = {'nouns':[], 'adj':[]}
    text = nltk.word_tokenize(raw_text)
    tagged=nltk.pos_tag(text)
    for i in tagged:
        if i[1][0] == "N":
            tags['nouns'].append(i[0])
        elif i[1][0] == "J":
            tags['adj'].append(i[0])
    return tags

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

# if iou > 0.9, we consider they are the same box
def IoU(b1, b2):
    if b1[2] <= b2[0] or \
        b1[3] <= b2[1] or \
        b1[0] >= b2[2] or \
        b1[1] >= b2[3]:
        return 0
    b1b2 = np.vstack([b1,b2])
    minc = np.min(b1b2, 0)
    maxc = np.max(b1b2, 0)
    union_area = (maxc[2]-minc[0])*(maxc[3]-minc[1])
    int_area = (minc[2]-maxc[0])*(minc[3]-maxc[1])
    return float(int_area)/float(union_area)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def box_process(box, image_pil):
    processed_box = box.clone()
    size = image_pil.size
    H, W = size[1], size[0]
    processed_box = processed_box * torch.Tensor([W, H, W, H])
    processed_box[:2] = processed_box[:2] - processed_box[2:] / 2
    processed_box[2:] = processed_box[:2] + processed_box[2:]
    return processed_box

def show_box(box, ax, label, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def save_mask_data(output_dir, caption, mask_list, box_list, label_list):
    value = 0  # 0 for background
    mask_list = torch.stack(mask_list)
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask_grassland.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'caption': caption,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] 
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label_grassland.json'), 'w') as f:
        json.dump(json_data, f)

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

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model with specific category set
    category_set = json.load(open('label_words.json'))
    category_texts = []
    for k in category_set:
        category_texts.append(category_set[k])
    # generate caption and tags for categories and run grounding dino model with captions
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    caption = generate_caption(image_pil)
    print(f"Caption: {caption}")
    tags = generate_tags(caption)
    for tag in tags['nouns']:
        if tag not in category_texts:
            category_texts.append(tag)

    total_boxes = []
    total_predphrases = []
    total_scores = []
    for category in category_texts:
        boxes_filt, pred_phrases, pred_scores = get_grounding_output(
            model, image, category, box_threshold, text_threshold, device=device
        )
        # fuse those overlap bbox with highest score of label
        if boxes_filt.shape[0] > 0:
            total_boxes.append(boxes_filt)
            total_predphrases.append(pred_phrases)
            total_scores.append(pred_scores)
    valid_boxes = []
    valid_phrases = []
    valid_scores = []
    # filter those overlapped boxes by nms and process boxes into image size (xyxy)
    for boxes_filt, pred_phrases, pred_scores in zip(total_boxes, total_predphrases, total_scores):
        for i in range(boxes_filt.shape[0]):
            valid_boxes.append(box_process(boxes_filt[i], image_pil))
            valid_phrases.append(pred_phrases[i])
            valid_scores.append(pred_scores[i])
    valid_boxes = torch.stack(valid_boxes)
    valid_scores = torch.stack(valid_scores)
    nms_idx = torchvision.ops.nms(valid_boxes, valid_scores, iou_threshold=0.5).numpy().tolist()
    valid_boxes = valid_boxes[nms_idx]
    valid_phrases = [valid_phrases[idx] for idx in nms_idx]
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    total_masks = []
    # mask with accurate bounding boxes
    for valid_box in valid_boxes:
        valid_box = valid_box.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(valid_box, image.shape[:2])

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        total_masks.append(masks)
    
    # visualization image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for masks in total_masks:
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # non-ambiguity annotation
    for valid_box, valid_phrase in zip(valid_boxes, valid_phrases):
        show_box(valid_box.numpy(), plt.gca(), valid_phrase, random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "annotation_edit_output_grassland.jpg"), bbox_inches="tight")

    # save for mask annotation data in json
    save_mask_data(output_dir, caption, total_masks, valid_boxes, valid_phrases)
