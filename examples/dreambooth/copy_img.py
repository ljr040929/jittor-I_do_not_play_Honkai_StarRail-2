import glob
import shutil
import os
import tqdm
import json

max_num = 15
dataset_root = "../../A"
style_path = "./style/style_newprompt"

for style_id in tqdm.tqdm(range(0, max_num)):
    task_id = "{:0>2d}".format(style_id)
    output_path = f"{dataset_root}/{task_id}/object"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    with open(f"{dataset_root}/{task_id}/prompt_new.json", "r") as file:
        prompts = json.load(file)
    for object_id, caption in prompts["caption"].items():
        if not os.path.exists(f"{output_path}/{caption}"):
            os.makedirs(f"{output_path}/{caption}", exist_ok=True)
        filename = os.listdir(f"{dataset_root}/{task_id}/images")
        for file in tqdm.tqdm(filename, desc=f"Style {style_id} | {caption}"):
            shutil.copy(f"{dataset_root}/{task_id}/images/{file}",f"{output_path}/{caption}")


