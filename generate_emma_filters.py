import os
import json
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
import base64

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"

async def call_model(messages, temperature=0.7):
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()
import base64
from io import BytesIO

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def evaluate_problem(item, logs_path):
    # 1. Ask for plan (deterministic)
    encoded_image = pil_to_base64(item["image_1"])
    plan_prompt = [
        {"role": "system", "content": "You are an expert visual assistant, who is an expert at answering questions that involve images."},
        {"role": "user", "content": [{"type": "text", 
                                      "text": f"Given the below image and question, generate a concise plan to help a text-only LLM solve the problem. Please enumerate the steps, but do not use bullet points. Problem: {item['question']}\nOptions: {item.get('options')}\nProvide a plan to solve this problem."},
                                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]}
    ]
    plan = await call_model(plan_prompt, temperature=0.0)
    # Save the plan in a temp directory for later inspection
    os.makedirs("out/emma_filtering/", exist_ok=True)
    plan_path = os.path.join("out/temp", f"{item.get('pid', 'unknown')}.txt")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan)

    # 2. Create 10 sampled calls in parallel
    answer_prompts = [
        [
            {"role": "system", "content": "You are an expert visual assistant, who is an expert at answering questions that involve images. Use the following plan to answer the problem."},
            {"role": "user", "content": f"Question: {item['question']}\nOptions: {item.get('options')}. Here's a plan to help you: Plan: {plan}\n Think step by step, then output with Final Answer:"}
        ]
        for _ in range(10)
    ]
    tasks = [call_model(msg, temperature=0.8) for msg in answer_prompts]
    answers = await asyncio.gather(*tasks)
    plan_path = os.path.join(logs_path, f"{item.get('pid', 'unknown')}.txt")
    

    # 3. Count correct
    correct = sum(1 for ans in answers if ans.split()[0].lower() == item["answer"].lower())
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(f"Question: {item['question']}\n")
        f.write(f"Answer: {item['answer']}\n")
        f.write(f"Plan: {plan}")
        for idx, ans in enumerate(answers):
            f.write(f"Raw prediction {idx}: {ans}")
        if correct <= 5:
            f.write(f"Filtered out")
        else:
            f.write(f"Kept")
    return correct <= 5, correct

async def main(logs_path):
    ds = load_dataset("luckychao/EMMA", "Math", split="test")
    filtered_items = []
    import tqdm
    for item in tqdm.tqdm(ds):
        keep, correct = await evaluate_problem(item, logs_path)
        if keep:
            item["correct_count"] = correct
            filtered_items.append(item)
        else:
            print(f"Filtered out {item['pid']} (correct {correct}/10)")
    print(f"After filtering, we have {len(filtered_items)} datasets")
    # Save filtered dataset
    with open("emma_math_filtered.jsonl", "w") as f:
        for it in filtered_items:
            f.write(json.dumps(it) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_path", type=str, default="out/emma_filtering/logs", help="Path to save logs")
    args = parser.parse_args()
    logs_path = args.logs_path
    asyncio.run(main(logs_path))
