import argparse
import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def evaluate_pope(model, processor, data_path, split):
    """
    Evaluates the model on the POPE benchmark JSON data.
    Expected JSON format: {"image": "COCO_val2014_000000000042.jpg", "text": "Is there a dog in the image?", "label": "yes"}
    """
    json_path = os.path.join(data_path, f"coco_pope_{split}.json")
    image_dir = os.path.join(data_path, "val2014") # Default COCO val image dir
    
    if not os.path.exists(json_path):
        print(f"Warning: Data {json_path} not found. Returning mock metrics.")
        return 0.5, 0.5, 1.0
        
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]
        
    correct = 0
    false_positives = 0
    total_negatives = 0
    total_yes_pred = 0
    total = len(data)
    
    for item in tqdm(data, desc=f"Evaluating {split}"):
        image_path = os.path.join(image_dir, item['image'])
        question = item['text']
        true_label = item['label'].lower().strip()
        
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            continue
            
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda", torch.float16)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=5)
            
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[-1].strip().lower()
        
        pred_yes = "yes" in response
        is_true_yes = (true_label == "yes")
        
        if pred_yes == is_true_yes:
            correct += 1
            
        if not is_true_yes:
            total_negatives += 1
            if pred_yes:
                false_positives += 1
                
        if pred_yes:
            total_yes_pred += 1
            
    acc = correct / max(total, 1)
    fpr = false_positives / max(total_negatives, 1)
    yes_bias = total_yes_pred / max(total/2, 1) # Expected 50% yes
    
    return acc, fpr, yes_bias

def main():
    parser = argparse.ArgumentParser(description="Evaluate on POPE benchmark")
    parser.add_argument("--model_id", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--split", type=str, default="all", choices=["random", "popular", "adversarial", "all"])
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--data_path", type=str, default="data/raw/POPE")
    
    args = parser.parse_args()
    
    print(f"Starting POPE Evaluation for {args.model_id}")
    print(f"Split: {args.split}, 4-bit: {args.use_4bit}")
    
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    kwargs = {"device_map": "auto"}
    if args.use_4bit:
        kwargs["load_in_4bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16
        
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **kwargs)
    model.eval()
    
    splits_to_run = ["random", "popular", "adversarial"] if args.split == "all" else [args.split]
    
    print("\n--- POPE Results ---")
    for split in splits_to_run:
        acc, fpr, yes_bias = evaluate_pope(model, processor, args.data_path, split)
        print(f"{split.capitalize()} - Accuracy: {acc*100:.1f}%, FPR: {fpr*100:.1f}%, Yes/No Bias: {yes_bias:.2f}")

if __name__ == "__main__":
    main()
