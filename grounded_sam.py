import argparse
import json
import os
import copy

import numpy as np
import torch
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

def box_process(box):
    processed_box = box.clone()
    processed_box[:2] = box[:2] - box[2:] / 2
    processed_box[2:] = box[:2] + box[2:] / 2
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

    # run grounding dino model
    category_set = json.load(open('/home/qifan/InstuctSGG/self_instruction/label_words.json'))
    category_texts = []
    prompt_template = ""
    for k in category_set:
        category_texts.append(prompt_template + category_set[k])
    total_boxes = []
    total_predphrases = []
    for category in category_texts:
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, category, box_threshold, text_threshold, device=device
        )
        # fuse those overlap bbox with highest score of label
        if boxes_filt.shape[0] > 0:
            total_boxes.append(boxes_filt)
            total_predphrases.append(pred_phrases)
    valid_boxes = []
    valid_phrases = []
    for boxes_filt, pred_phrases in zip(total_boxes, total_predphrases):
        for i in range(boxes_filt.shape[0]):
            box = boxes_filt[i]
            overlap = False
            for j in range(len(valid_boxes)):
                # process the bbox results from dino (x_c, y_c, w, h)
                processed_curbox = box_process(valid_boxes[j])
                processed_box = box_process(box)
                iou = IoU(processed_curbox.tolist(), processed_box.tolist())
                if iou > 0.9:
                    # consider as the same box
                    valid_phrases[j] = pred_phrases[i] if float(pred_phrases[i][-5:-1])>float(valid_phrases[j][-5:-1]) else valid_phrases[j]
                    overlap = True
                    break
            if not overlap:
                valid_boxes.append(box)
                valid_phrases.append(pred_phrases[i])
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    total_masks = []
    for boxes_filt in total_boxes:
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        total_masks.append(masks)
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for masks in total_masks:
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # overlapped annotation
    # for boxes_filt, pred_phrases in zip(total_boxes, total_predphrases):
    #     for box, label in zip(boxes_filt, pred_phrases):
    #         show_box(box.numpy(), plt.gca(), label, random_color=True)
    # non-ambiguity annotation
    for valid_box, valid_phrase in zip(valid_boxes, valid_phrases):
        show_box(valid_box.numpy(), plt.gca(), valid_phrase, random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "annotation_output.jpg"), bbox_inches="tight")
