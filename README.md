# Annotation-anything-pipeline
Annotation anything in one-pipeline

Using stable diffusion to generate and annotate bounding boxes and masks for object detection and segmentation just in one-pipeline! 

**LLM is a data specialist based on AIGC models.**  
1. ChatGPT acts as an educator to guide AIGC models to generate a variety of controllable images in various scenarios
2. Generally, given a raw image from the website or AIGC, SAM generated the masked region for source image and GroundingDINO generated the open-set detection results just in one step. Then, we filter overlap bounding boxes and obtain non-ambiguity annotations.
3. Mixture text prompt and clip model to select the region by similaity scores, which can be finally used to generate the target edited image with stable-diffusion-inpaint pipeline.

Use ```bash grounded_sam.sh``` to initialize the annotation pipeline. 

## Generated Cases





# Reference 

[1] https://chat.openai.com/

[2] https://github.com/huggingface/diffusers 

[3] https://github.com/facebookresearch/segment-anything

[4] https://github.com/IDEA-Research/Grounded-Segment-Anything/
