#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-1 VQA smoke test for your Qwen2VL + BaseToolInferencer stack.

Example:
  python test_qwen2vl_vqa.py \
    --image ./demo.jpg \
    --question "What is the main object in this photo?" \
    --model Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import argparse
from PIL import Image

# Disable any breakpoint() in the provided source.
#os.environ["PYTHONBREAKPOINT"] = "0"

# ---- Project imports (paths assume this script sits at repo root) ----
from tool_server.tf_eval.models.vllm_models import VllmModels
from tool_server.tf_eval.tool_inferencer.tool_inferencer import BaseToolInferencer

# --------------------- Minimal single-item dataset ---------------------
class SingleVQADataset:
    """Yields one sample with keys idx, text, image. Collects results via store_results()."""
    def __init__(self, image_path: str, question: str):
        self.image = Image.open(image_path).convert("RGB")
        self.question = question
        self._results = {}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return {
            "idx": 0,
            "text": self.question,
            "image": self.image,  # DynamicBatchManager expects PIL here
        }

    # BaseToolInferencer calls this; we capture whatever it gives us.
    def store_results(self, payload: dict):
        # payload like: {"idx": 0, "results": { ... "model_response": [...], "final_answer": ...}}
        self._results[payload["idx"]] = payload["results"]

    @property
    def results(self):
        return self._results.get(0, None)

# ------------------------------- Main ---------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HF model id or local path for Qwen2-VL Instruct.")
    p.add_argument("--image", required=True, help="Path to an input image.")
    p.add_argument("--question", required=True, help="VQA question about the image.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()

    # Instantiate your tp_model wrapper
    qwen = VllmModels(pretrained=args.model, limit_mm_per_prompt="4")

    # Set a simple generation config since Qwen2VL.generate expects it
    qwen.generation_config = {"max_new_tokens": args.max_new_tokens}

    # Build the one-sample dataset
    dataset = SingleVQADataset(args.image, args.question)

    # Inferencer with batch_size=1 and max_rounds=0 so we finish after 1 response
    infer = BaseToolInferencer(
        tp_model=qwen,
        batch_size=1,
        model_mode="general",
        max_rounds=0,            # important: finish immediately; no tool loop
        stop_token="<stop>",
        controller_addr="http://127.0.0.1:20001",  # unused here, but required by ctor
    )

    # Run the full loop once
    infer.batch_inference(dataset)

    # Fetch and print results
    res = dataset.results
    if not res:
        print("No results captured.")
        return

    # Raw model output(s)
    responses = res.get("model_response", [])
    print("\n=== Raw Model Responses ===")
    for i, r in enumerate(responses):
        print(f"[{i}] {r}")

    # If your manager parsed a 'final_answer', show it (may be None if model didn't emit Terminate)
    final_answer = res.get("final_answer", None)
    if final_answer:
        print("\n=== Parsed Final Answer ===")
        print(final_answer)

if __name__ == "__main__":
    main()
