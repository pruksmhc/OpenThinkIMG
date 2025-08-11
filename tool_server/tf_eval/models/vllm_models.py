from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List
import torch
from vllm import LLM, SamplingParams


from .template_instruct import *
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem

from ..utils.log_utils import get_logger
inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("vllm_models",)

class VllmModels(tp_model):
    def __init__(
      self,  
      pretrained : str = None,
      tensor_parallel: str = "1",
      limit_mm_per_prompt: str = "1"
    ):
        tensor_parallel = eval(tensor_parallel)
        self.model = LLM(
            model=pretrained,
            tensor_parallel_size=tensor_parallel,
            limit_mm_per_prompt={"image": int(limit_mm_per_prompt)}
        )

    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):  
        text = "Question: " + text
        
        image = pil_to_base64(image)
        offline_prompt = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
    - **crop**: Crops an image. Example: `{"name": "crop", "arguments": {"image": "img_1", "param": "[100, 100, 300, 300]"}}`
    To solve the problem, Select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
    Your output should be in a strict JSON format as follows:
    {"thought": "the reasoning process", "actions": [{"name": "action", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
    """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": offline_prompt,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                ],
            }
        ]

        return messages
    
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role
    ):
        if image:
            new_messages=[
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        },
                    ],
                }
            ]
        else:
            new_messages=[
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }
            ]
        
        conversation.extend(new_messages)

        return conversation
    
    
    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        if len(batch) == 0:
            return None
        messages = []
        for item in batch:
            messages.append(item.conversation)
        return messages
    
    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6)
        
        inputs = self.form_input_from_dynamic_batch(batch)
        # breakpoint()
        response = self.model.chat(inputs, sampling_params)


        for item, output_item in zip(batch, response):
            output_text = output_item.outputs[0].text
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self
            
        
        
