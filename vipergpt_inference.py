import os
import json
import sys
sys.path.append("../../")

from tasks import get_task_data
import time
from PIL import Image

os.environ["CONFIG_DIR"] = "./configs"
os.environ["CONFIG_NAMES"] = "gpt4" # config file under: configs
# for name, value in os.environ.items():
#     print("{0}: {1}".format(name, value))
# current_file_directory = os.path.dirname(__file__)
# VIPERGPT_SRC= os.path.join(current_file_directory, "third_party", "viper")
# sys.path.append(VIPERGPT_SRC)
print(sys.path)
print(os.environ["OPENAI_API_KEY"])

from main_simple_lib import *
print(config)

def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def append_to_jsonl(path, data):
    with open(path, 'a') as file:
        json_str = json.dumps(data)
        file.write(json_str + '\n')


def vipergpt_inference(text, images, system_message):
    assert len(images) == 1 # only support single image for now
    image_path = images[0]
    im = load_image(image_path)
    # show_single_image(im)
    code = get_code(text)
    try:
        result = execute_code(code, im, show_intermediate_steps=False)
        print(result)
    except Exception as e:
        print(e)
        print("Error in executing code")
        result = None
    if code is None:
        return None
    return {"code_input":code[0], "code_output":result, "prompt":text}

def main(args):
    # set up openai api
    subset_size = args.subset_size
    print("subset_size:", subset_size)
    
    SLEEP_RATE = args.sleep_rate
    print("SLEEP_RATE set to:", SLEEP_RATE)

    if args.run_task is not None:
        outputs = []
        os.makedirs(args.output_dir, exist_ok=True)
        task_data = get_task_data(args.run_task, args.dataset_name, prompt_version=None)

        if args.subset_indices_json is not None:
            print("load subset index from json...")
            subset_idx = json.load(open(args.subset_indices_json, 'r'))
        else:
            if subset_size > len(task_data["images"]):
                print("randomly sample subset index...")
                import random
                random.seed(42)
                subset_idx = random.sample(range(len(task_data["images"])), subset_size)
            else:
                subset_idx = list(range(subset_size))
        print("subset_idx:", subset_idx)
        print("subset_idx len", len(subset_idx))
        
        assert len(subset_idx) == subset_size
        
        with open(os.path.join(args.output_dir, "dataset_config.json"), 'w') as file:
            json.dump({
                "task_name":args.run_task,
                "dataset_name":args.dataset_name,
                "subset_size":subset_size,
                "subset_idx":subset_idx
            }, file)

        output_path = os.path.join(args.output_dir, "response.jsonl")

        instance_already_processed = set()
        if os.path.exists(output_path):
            print("output exists, check existing ids...")
            existing_data = load_jsonl(output_path)
            for item in existing_data:
                if item != {} and item['api_response'] != []:
                    instance_already_processed.add(item["instance_data"]["id"])


        system_message = task_data["system_message"]
        if system_message is None:
                system_message = ""
        else:
            print("system_message:", system_message)
    
        # run inference on subset
        for idx in subset_idx:
            img_p = task_data["images"][idx]
            text = task_data["prompts"][idx]
            instance_info = task_data["info"][idx]

            if instance_info["id"] in instance_already_processed:
                print("instance already processed, skip:", instance_info["id"])
                continue

            print("running instance:", instance_info["id"])
            if img_p is not None and img_p != "" and os.path.exists(img_p):
                images = [img_p]
            else:
                images = []
            if args.additional_prompt_suffix is not None:
                text += " " + args.additional_prompt_suffix

            ret = vipergpt_inference(text, images, system_message)

            output = {"api_response":ret, "instance_data":instance_info} 
            outputs.append(output)

            if os.path.exists(output_path):
                append_to_jsonl(output_path, output)
            else:
                with open(output_path, 'w') as file:
                    json_str = json.dumps(output)
                    file.write(json_str + '\n')
            # import pdb; pdb.set_trace()
            time.sleep(SLEEP_RATE)
    else:        
        # run single inference
        if args.image_file is not None and args.image_file != "" and os.path.exists(args.image_file):
            images = [args.image_file]
        else:
            images = []
        text = args.text_prompt
        if args.additional_prompt_suffix is not None:
            text += " " + args.additional_prompt_suffix
        print(images, text)
        ret = vipergpt_inference(text, images, system_message)
        output = {"api_response":ret, "instance_data":{"text":text, "images":images}}
        print(output)

import argparse
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional_prompt_suffix", type=str, default=None)
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--subset_indices_json", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=False, default=None)
    parser.add_argument("--run-task", type=str, required=False, default=None)
    parser.add_argument("--dataset-name", type=str, required=False, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sleep_rate", type=int, default=15)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    main(args)