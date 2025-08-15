"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse
import torch
import numpy as np
from PIL import Image

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
import traceback

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

def extract_points(molmo_output, image_w, image_h):
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return np.array(all_points)  # Ensure it's always a NumPy array

def show_points(coords, labels, ax, marker_size=375):
    # Only plot if there are points
    if len(coords) == 0:
        return
    
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def create_image_with_points(image, coords, labels, marker_size=375):
    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    image_format = image.format.lower()
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'

    # Only show points if there are any valid coordinates
    show_points(coords, labels, ax, marker_size)

    plt.axis('off')  # Turn off axis

    # Convert the figure to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

class MolmoToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = "/mnt/petrelfs/share_data/suzhaochen/models/Molmo-7B-D-0924", 
                 model_base = "", 
                 model_name = "Point",
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 max_length = 2048,
                 args = None,
                 ):
        self.max_length = max_length
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            args=args
            )

        
    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
        quant_config = None
        print(f"load8bitarg:{self.args.load_8bit}\n load4bitarg:{self.args.load_4bit}")
        self.load_4bit = self.args.load_4bit 
        self.load_8bit = self.args.load_8bit 
        
        print(f"load8bit:{self.load_8bit}\n load4bit:{self.load_4bit}")
        if self.load_4bit or self.load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=self.load_8bit,
                load_in_4bit=self.load_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            logger.info(f"Using quantization config: {quant_config}")

        # load the processor
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map={"": 0},
            quantization_config=quant_config
        )

        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        image =  base64_to_pil(image) #  PIL image
        # breakpoint()
        text_prompt = "Point to the {} in the scene.".format(generate_param)
        
        ret = {"text": "", "error_code": 0}
        try:
            with torch.no_grad():
                inputs = self.processor.process(
                    images=[image],
                    text=text_prompt,
                )
                inputs["images"] = inputs["images"].to(torch.bfloat16)

                inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                        output = self.model.generate_from_batch(
                            inputs,
                            GenerationConfig(max_new_tokens=self.max_length, stop_strings="<|endoftext|>"),
                            tokenizer=self.processor.tokenizer
                        )

                        # only get generated tokens; decode them to text
                        generated_tokens = output[0,inputs['input_ids'].size(1):]
                        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        if response is not None:
                            ret["text"] = response
 
                
        except Exception as e:
            logger.error(f"Error when using molmo to point: {e}")
            logger.error(traceback.format_exc()) 
            ret["text"] = f"Error when using molmo to point: {e}"
            # ret['edited_image'] = None
        
        return ret
    
    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "point",
                "description": "Identify a point in the image based on a natural language description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier or path of the image in which to locate the point, e.g., 'img_1'."
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the point of interest, e.g., 'the dog’s nose', 'center of the clock', 'the tallest tree'."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }
        return instruction 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20027)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/share_data/suzhaochen/models/Molmo-7B-D-0924")
    parser.add_argument("--gpu_usage_limit",type=float, default=None)
    parser.add_argument("--load-8bit", type=str2bool, default=False, help="Use 8-bit quantization")
    parser.add_argument("--load-4bit",  type=str2bool, default=False, help="Use 4-bit quantization (nf4)")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = MolmoToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path = args.model_path,
        args = args,
    )
    worker.run()