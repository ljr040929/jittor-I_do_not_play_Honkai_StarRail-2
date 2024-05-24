import json, os, tqdm, torch

from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 15
dataset_root = "../../A"
style_path = "./style/style_newprompt"
object_path = "./style/style_objects_separate"
output_path = "./output/output_objects_separate"

with torch.no_grad():
    for tempid in tqdm.tqdm(range(13, max_num)):
        taskid = "{:0>2d}".format(tempid)

        # load json
        with open(f"{dataset_root}/{taskid}/prompt_new.json", "r") as file:
            prompts = json.load(file)

        style_prompt = prompts["style"]
        for id, caption in prompts["caption"].items():
            print(caption)
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
            pipe.load_lora_weights(f"{style_path}/style_{taskid}")
            pipe.load_lora_weights(f"{object_path}/style_{taskid}/{caption}")
            image = pipe(f"Image of a {caption} in the style of {style_prompt}", num_inference_steps=50, width=512, height=512, negative_prompt="low quality, blurry, unfinished").images[0]
            os.makedirs(f"{output_path}/{taskid}", exist_ok=True)
            image.save(f"{output_path}/{taskid}/{caption}.png")
