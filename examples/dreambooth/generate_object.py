import json, os, tqdm, torch

from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 15
output_num = 50
dataset_root = "../../A"
style_path = "./style/style_newprompt"

with torch.no_grad():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
    for style_id in tqdm.tqdm(range(0, max_num)):
        task_id = "{:0>2d}".format(style_id)
        output_path = f"{dataset_root}/{task_id}/object"
        os.makedirs(output_path, exist_ok=True)
        with open(f"{dataset_root}/{task_id}/prompt_new.json", "r") as file:
            prompts = json.load(file)
        for object_id, caption in prompts["caption"].items():
            os.makedirs(f"{output_path}/{caption}", exist_ok=True)
            for idx in tqdm.tqdm(range(0, output_num), desc=caption):
                image = pipe(f"Image of a {caption}, best quality, ultra-detailed", num_inference_steps=50, width=512, height=512, negative_prompt="low quality, blurry, unfinished").images[0]
                image.save(f"{output_path}/{caption}/{idx}.png")
