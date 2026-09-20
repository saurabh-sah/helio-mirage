import argparse
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_data(example, processor):
    # This is a generic formatting function, assuming 'image' and 'text' features
    # Adjust according to the exact schema of the fine-tuning dataset
    prompt = f"USER: <image>\n{example['text']}\nASSISTANT: {example['label']}"
    
    # Process image if available, else skip
    try:
        inputs = processor(text=prompt, images=example['image'].convert("RGB"), return_tensors="pt")
    except Exception:
        # Fallback for text-only formatting if images are missing
        inputs = processor(text=prompt, return_tensors="pt")
        
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    # For CausalLM, labels are typically the input_ids
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Task B")
    parser.add_argument("--config-name", type=str, default="lora_config.yaml")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, default="data/raw/llava_instruct")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    print(f"Starting LoRA fine-tuning with config: {args.config_name}")
    
    # Load config
    config_dict = {}
    if os.path.exists(args.config_name):
        with open(args.config_name, 'r') as f:
            config_dict = yaml.safe_load(f)
            
    r = config_dict.get('r', 16)
    lora_alpha = config_dict.get('lora_alpha', 32)
    lora_dropout = config_dict.get('lora_dropout', 0.05)
    target_modules = config_dict.get('target_modules', ["q_proj", "v_proj"])

    print("Loading processor and base model...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    print("Initializing PEFT configuration...")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading contrastive dataset...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset path {args.data_path} not found. Please ensure the instruction dataset is downloaded and the path is correct.")
    # Assuming the data is stored in Parquet or JSONL
    try:
        dataset = load_dataset("json", data_files=os.path.join(args.data_path, "*.json"))
        train_dataset = dataset["train"]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Map dataset
    train_dataset = train_dataset.map(lambda x: format_data(x, processor), remove_columns=train_dataset.column_names)

    print("Starting training loop...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: {k: torch.stack([item[k] for item in x]) for k in x[0].keys()}
    )
    
    trainer.train()
    
    print(f"Training complete! Model saved to {args.output_dir}")
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
