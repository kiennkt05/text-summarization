import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
from unsloth import FastLanguageModel
from transformers import TextStreamer
import PTM.model.config as config
from PTM.evaluate.rouge_eval import compute_rouge_and_bleu
from PTM.evaluate.bertscore import compute_bertscore

def generate_summary(model, tokenizer, text, stream=True):
    """
    Generates a summary for a given text using the fine-tuned model.
    """
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tóm tắt văn bản sau đây.

### Input:
{}

### Response:
{}"""

    prompt = alpaca_prompt.format(text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    if stream:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            streamer=text_streamer, 
            max_new_tokens=128, 
            pad_token_id=tokenizer.eos_token_id
        )
        return ""
    else:
        outputs = model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=128, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        # Skip the prompt in the output
        output_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return output_text

def generate_summary_batch(model, tokenizer, texts):
    """
    Generates summaries for a batch of texts using the fine-tuned model.
    """
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tóm tắt văn bản sau đây.

### Input:
{}

### Response:
{}"""

    prompts = [alpaca_prompt.format(text, "") for text in texts]
    
    # Left padding is required for batched generation in causal LMs
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        max_new_tokens=128, 
        use_cache=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the generated part by slicing past the prompt length
    prompt_lengths = inputs.input_ids.shape[1]
    output_texts = tokenizer.batch_decode(outputs[:, prompt_lengths:], skip_special_tokens=True)
    return output_texts

def evaluate_dataset(model, tokenizer, test_path, batch_size=8):
    print(f"Loading test dataset from {test_path}...")
    df = pd.read_parquet(test_path)
    df = df.dropna(subset=['article', 'summary'])
    
    predictions = []
    references = df['summary'].tolist()
    articles = df['article'].tolist()
    
    print("Generating predictions in batches...")
    for i in tqdm(range(0, len(articles), batch_size)):
        batch_articles = articles[i:i+batch_size]
        batch_preds = generate_summary_batch(model, tokenizer, batch_articles)
        predictions.extend(batch_preds)
        
    print("\nComputing metrics...")
    rouge_bleu_res = compute_rouge_and_bleu(predictions, references)
    bert_res = compute_bertscore(predictions, references)
    
    print("\nEvaluation Results")
    print(f"| ROUGE-1        | {rouge_bleu_res['rouge']['rouge1']*100:.2f} |")
    print(f"| ROUGE-2        | {rouge_bleu_res['rouge']['rouge2']*100:.2f} |")
    print(f"| ROUGE-L        | {rouge_bleu_res['rouge']['rougeL']*100:.2f} |")
    print(f"| BLEU           | {rouge_bleu_res['bleu']['bleu']*100:.2f} |")
    print(f"| BERTScore (F1) | {bert_res['bertscore']['f1']*100:.2f} |")
    
    # Save results
    os.makedirs("outputs_ptm", exist_ok=True)
    with open("outputs_ptm/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": predictions, 
            "references": references
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PTM or Generate Summary")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved LoRA model. If not provided, uses the original pretrained model.")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test parquet dataset for evaluation")
    parser.add_argument("--text", type=str, default=None, help="Text to summarize (single inference)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataset evaluation")
    args = parser.parse_args()
    
    if args.test_path is None and args.text is None:
        raise ValueError("Must provide either --test_path for evaluation or --text for single inference.")
    
    model_name_to_load = args.model_path if args.model_path and os.path.exists(args.model_path) else config.MODEL_NAME
    print(f"Loading model: {model_name_to_load}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_to_load,
        max_seq_length = config.MAX_SEQ_LENGTH,
        dtype = config.DTYPE,
        load_in_4bit = config.LOAD_IN_4BIT,
    )
    
    if args.test_path:
        evaluate_dataset(model, tokenizer, args.test_path, batch_size=args.batch_size)
    elif args.text:
        print("\n--- Summary ---")
        generate_summary(model, tokenizer, args.text, stream=True)
        print("\n---------------")
