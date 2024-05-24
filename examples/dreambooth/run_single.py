import json, os, tqdm, torch

from JDiffusion.pipelines import StableDiffusionPipeline

tempid = 14
taskid = "{:0>2d}".format(tempid)
objects = ["Pizza"]
dataset_root = f"../../A/{taskid}"
style_path = f"./style/style_objects/style_{taskid}"
output_path = f"./output/output_objects/{taskid}"

with torch.no_grad():
    # load json
    with open(f"{dataset_root}/prompt_new.json", "r") as file:
        prompts = json.load(file)
    style_prompt = prompts["style"]
    for id, caption in prompts["caption"].items():
        if caption in objects:
            positive_prompt = f"Image of a {caption} in the style of {style_prompt}, limited color"
            print(positive_prompt)
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
            pipe.load_lora_weights(f"{style_path}/{caption}")
            image = pipe(positive_prompt, num_inference_steps=50, width=512, height=512, negative_prompt="low quality, blurry, unfinished").images[0]
            os.makedirs(f"{output_path}", exist_ok=True)
            image.save(f"{output_path}/{caption}.png")
