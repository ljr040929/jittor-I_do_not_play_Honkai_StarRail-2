import json, os, tqdm, torch

from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 15
dataset_root = "../../A"
style_path = "./style/style_objects"
output_path = "./output/output_objects"

with torch.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)

        # load json
        with open(f"{dataset_root}/{taskid}/prompt_new.json", "r") as file:
            prompts = json.load(file)

        style_prompt = prompts["style"]
        for id, caption in prompts["caption"].items():
            positive_prompt = f"Image of a {caption} in the style of {style_prompt}"
            print(positive_prompt)
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
            pipe.load_lora_weights(f"{style_path}/style_{taskid}/{caption}")
            image = pipe(positive_prompt, num_inference_steps=50, width=512, height=512, negative_prompt="low quality, blurry, unfinished").images[0]
            os.makedirs(f"{output_path}/{taskid}", exist_ok=True)
            image.save(f"{output_path}/{taskid}/{caption}.png")
