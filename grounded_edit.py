import json
import omegaconf
import torch
import os
import argparse
import numpy as np
import yaml
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# stable diffusion
# from models.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# clip to filter ambuguite regions
import clip
# utils
from grounded_sam import load_image, load_model, show_box, generate_caption
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.ops import box_convert
from transformers import BlipProcessor, BlipForConditionalGeneration
# @torch.no_grad()
# def inpaint_img_with_lama_cleaner(
#         img: np.ndarray,
#         mask: np.ndarray,
#         config_p: str,
#         ckpt_p: str="./lama/configs/prediction/default.yaml",
#         mod = 8
# ):
#     assert len(mask.shape) == 2
#     img = torch.from_numpy(img).float().div(255.)
#     mask = torch.from_numpy(mask).float()
#     predict_config = omegaconf.load(config_p)
#     predict_config.model.path = ckpt_p
#     device = torch.device(predict_config.device)

#     train_config_path = os.path.join(
#         predict_config.model.path, 'config.yaml')

#     with open(train_config_path, 'r') as f:
#         train_config = omegaconf.OmegaConf.create(yaml.safe_load(f))

#     train_config.training_model.predict_only = True
#     train_config.visualizer.kind = 'noop'

#     checkpoint_path = os.path.join(
#         predict_config.model.path, 'models',
#         predict_config.model.checkpoint
#     )
#     model = load_checkpoint(
#         train_config, checkpoint_path, strict=False, map_location='cpu')
#     model.freeze()
#     if not predict_config.get('refine', False):
#         model.to(device)

#     batch = {}
#     batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
#     batch['mask'] = mask[None, None]
#     unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
#     batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
#     batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
#     batch = move_to_device(batch, device)
#     batch['mask'] = (batch['mask'] > 0) * 1

#     batch = model(batch)
#     cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
#     cur_res = cur_res.detach().cpu().numpy()

#     if unpad_to_size is not None:
#         orig_height, orig_width = unpad_to_size
#         cur_res = cur_res[:orig_height, :orig_width]

#     cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
#     return cur_res

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
# stable_pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to(device)
# stable_pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(stable_pix2pix.scheduler.config)

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
    parser.add_argument("--reverse", type=bool, default=False, required=False, help="whether reverse mask")
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
    reverse = args.reverse
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

    # judge which to edit (objects or background)


    # test for imagin
    # scene_image = {"size": [(2048, 1365)], "objects": [{"value": 1, "label": "dog", "logit": 0.48, "box": [550.0, 600.0, 1050.0, 950.0]}, {"value": 2, "label": "cat", "logit": 0.45, "box": [1100.0, 600.0, 1600.0, 950.0]}, {"value": 3, "label": "grass", "logit": 0.80, "box": [0.0, 1000.0, 2048.0, 1365.0]}, {"value": 4, "label": "sky", "logit": 0.85, "box": [0.0, 0.0, 2048.0, 500.0]}]}
    
    # if len(total_boxes) > 0:
    #     # print(image_pil.size)
    #     # initial_canvas = np.zeros(scene_image["size"][0]).T
    #     # np.expand_dims(initial_canvas, 2)
    #     # image_pil = Image.fromarray(np.uint8(initial_canvas))
    #     # print(image_pil.size)
    #     scene_image["objects"] = sorted(scene_image["objects"], key=lambda x: x["logit"], reverse=True)
    #     for object in scene_image["objects"]:
    #         image_mask = generate_masks_with_grounding(image_pil, np.array(object["box"]))
    #         phrase = object["label"]
    #         logit = object["logit"]
    #         mask_image = Image.fromarray(image_mask)  
    #         mask_image.save(os.path.join(output_dir, f'mask_{phrase}_{logit}.jpg'))
            
    #         image_source_for_inpaint = image_pil.resize((512, 512))
    #         image_mask_for_inpaint = mask_image.resize((512, 512))
    #         image_inpainting = ldm_stable_inpaint(prompt=object["label"], image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
    #         image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
    #         image_inpainting.save(os.path.join(output_dir, f'edit_result_{phrase}_{logit}.jpg'))
    #         image_pil = image_inpainting.copy()
    # ccc
    
    
    # objects mask with bounding box
    # if len(total_boxes) > 0:
    #     # Cut out all masks and select the most correct bounding box with clip
    #     cropped_boxes = []
    #     for box, pred_phrase in zip(total_boxes, total_predphrases):
    #         cropped_boxes.append(image_pil.crop(box.tolist()))
    #     idx = torch.argmax(retriev(cropped_boxes, text_prompt)).item()
    #     box = total_boxes[idx]
    #     image_mask = generate_masks_with_grounding(image_pil, box.numpy())
    #     mask_image = Image.fromarray(image_mask)  
    #     phrase, logit = pred_phrase.split('(')
    #     logit = logit[:-1]   
    #     mask_image.save(os.path.join(output_dir, f'mask_{phrase}_{logit}.jpg'))
    #     image_source_for_inpaint = image_pil.resize((512, 512))
    #     image_mask_for_inpaint = mask_image.resize((512, 512))
    #     image_inpainting = ldm_stable_inpaint(prompt=edit_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
    #     image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
    #     image_inpainting.save(os.path.join(output_dir, f'edit_result_{phrase}_{logit}.jpg'))
    
    # background edit with segmentation mask
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    caption = generate_caption(image_pil, processor, blip_model)
    print(caption)
    edit_prompts_background = ['river', 'Open field', 'Sunny meadow', 'Backyard', 'Nature reserve', 'Beach', 'Forest', 'City park', 'Hilltop', 'Countryside', 'Pasture']
    edit_prompts_object = ['Meadow flowers', 'Birds', 'Insects', 'Fence', 'Lake', 'Sand dunes', 'Buildings', 'Playground equipment', 'Hiking trail', 'Picnic table']
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
        if reverse:
            # need to reverse the mask to modift the background
            mask_image = Image.fromarray(~masks[0][0].cpu().numpy())
        else:
            mask_image = Image.fromarray(masks[0][0].cpu().numpy())
        phrase, logit = pred_phrase.split('(')
        logit = logit[:-1] 
        mask_image.save(os.path.join(output_dir, f'mask_{phrase}_{logit}_nonreverse.jpg'))
        image_source_for_inpaint = image_pil.resize((512, 512))
        image_mask_for_inpaint = mask_image.resize((512, 512))
        image_inpainting = ldm_stable_inpaint(prompt=edit_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
        image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
        image_inpainting.save(os.path.join(output_dir, f'edit_result_{phrase}_{logit}_nonreverse.jpg'))
        # instructPix2Pix for inpainting
        # for edit_prompt in edit_prompts_object:
            # image_inpainting_instruct = stable_pix2pix("change the background into "+edit_prompt, image=image_source_for_inpaint).images[0]
            # image_inpainting_instruct = stable_pix2pix("add "+edit_prompt+" into suitable position in the the image", image=image_source_for_inpaint).images[0]
            # lama-cleaner for removal objects

            # image_inpainting_instruct = image_inpainting_instruct.resize((image_pil.size[0], image_pil.size[1]))
            # image_inpainting_instruct.save(os.path.join(output_dir, f'edit_result_{phrase}_{logit}_{edit_prompt}.jpg'))
        

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)

    # non-ambiguity annotation
    for boxes_filt, pred_phrases in zip(total_boxes, total_predphrases):
    # for valid_box, valid_phrase in zip(boxes_filt, pred_phrases):
        show_box(boxes_filt.numpy(), plt.gca(), pred_phrases, random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "test.jpg"), bbox_inches="tight")

