import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel

import PTM.model.config as config
from PTM.evaluate.rouge_eval import compute_rouge_and_bleu
from PTM.evaluate.bertscore import compute_bertscore

def generate_summary(model, tokenizer, text, oneshot_enabled, cot_enabled, max_new_tokens=128, stream=True):
    """
    Generates a summary for a given text using the fine-tuned model.
    """
    instruction = """Nhiệm vụ của bạn là đọc kỹ bài báo được cung cấp và viết một đoạn tóm tắt ngắn gọn, súc tích nhưng vẫn giữ được các thông điệp cốt lõi.

Hãy áp dụng phương pháp tư duy từng bước (Chain-of-Thought) theo hướng dẫn sau:

1. Phân tích ngữ cảnh: Đọc toàn bộ bài viết và xác định chủ đề chính.
2. Trích xuất thông tin cốt lõi (5W1H):
   - Ai/Cơ quan nào? (Who)
   - Sự kiện/Vấn đề gì đang diễn ra? (What)
   - Thời gian nào? (When)
   - Ở đâu? (Where)
   - Tại sao/Mục đích là gì? (Why/Goal)
3. Lọc bỏ chi tiết phụ: Loại bỏ các số liệu quá chi tiết, trích dẫn trực tiếp dài dòng, hoặc các ví dụ nhỏ lẻ không làm thay đổi nội dung tổng thể.
4. Tổng hợp và Viết tóm tắt: Kết nối các thông tin cốt lõi ở bước 2 thành một đoạn văn hoàn chỉnh, logic, dễ đọc (khoảng 2-3 câu).
""" if cot_enabled else "Tóm tắt văn bản sau đây."

    oneshot = """
## Example
### Instruction:
Tóm tắt văn bản sau đây.

### Input:
Gần 20 sự kiện được tổ chức trên toàn thành phố, kéo dài từ 19/4 đến 10/5. Theo Sở Du lịch Hà Nội, ngoài thu hút du khách, loạt sự kiện cũng là các gợi ý dành cho người dân thủ đô không đi chơi xa và muốn tham gia các hoạt động trong ngày. Một số hoạt động tiêu biểu gồm Lễ hội Du lịch Hà Nội 2024 với chủ đề 'Thăng Long - Hà Nội, Thủ đô quyến rũ'; Triển lãm Ngô Quyền - Anh hùng dân tộc kiệt xuất và Thăng Long hội tụ; Việt Nam - những chiến thắng làm thay đổi dòng chảy lịch sử thế giới hay Tái hiện lễ hội Cầu mưa dân tộc Lô Lô, tỉnh Cao Bằng, Lễ hội Tình yêu năm 2024. Bên cạnh đó, nhân dịp kỷ niệm ngày Giải phóng miền Nam, thống nhất đất nước 30/4 - Quốc tế Lao động 1/5 và Ngày sinh Chủ tịch Hồ Chí Minh 19/5, Sở phối hợp Ban Quản lý Lăng Chủ tịch hỗ trợ nước, sữa và bánh miễn phí phục vụ nhân dân, du khách đến viếng Lăng Bác. Hà Nội cũng có rất nhiều sản phẩm du lịch xanh được ra mắt và đẩy mạnh trong thời gian qua như trải nghiệm xe điện trong lòng phố cổ, tour xe đạp, các sản phẩm du lịch sinh thái và nghỉ dưỡng ở ngoại thành. Nghỉ lễ kéo dài 5 ngày kéo theo nhu cầu tham quan, di chuyển của người dân dự kiến tăng cao. Giám đốc Sở Du lịch Hà Nội, Đặng Hương Giang, cho biết đây là 'cơ hội lớn' cho ngành tăng sức hút với du khách. Sở đã chỉ đạo các bên liên quan rà soát, nâng cao chất lượng sản phẩm, dịch vụ du lịch để đáp ứng nhu cầu của du khách. Hà Nội kỳ vọng với nhiều hoạt động trong kỳ nghỉ lễ nêu trên, ngành du lịch sẽ phục vụ chuyên nghiệp, chu đáo, đáp ứng tốt nhất nhu cầu tham quan, giải trí, nghỉ ngơi của nhân dân và thu hút nhiều khách quốc tế ghé thăm. Phương Anh

### Response:
Hà Nội tổ chức gần 20 sự kiện từ 19/4 đến 10/5, bao gồm Lễ hội Du lịch Hà Nội 2024, các triển lãm và lễ hội văn hóa, nhằm thu hút cả du khách và người dân.  Song song đó, thành phố cũng đẩy mạnh các sản phẩm du lịch xanh và chuẩn bị chu đáo để đáp ứng nhu cầu tăng cao trong kỳ nghỉ lễ 5 ngày, kỳ vọng thu hút nhiều khách du lịch trong và ngoài nước.
    """ if oneshot_enabled else ""

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
{}
### Instruction:
{}

### Input:
{}

### Response:
{}"""

    prompt = alpaca_prompt.format(oneshot, instruction, text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    if stream:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            streamer=text_streamer, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.eos_token_id
        )
        return ""
    else:
        outputs = model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        # Skip the prompt in the output
        output_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return output_text

def generate_summary_batch(model, tokenizer, texts, oneshot_enabled, cot_enabled, max_new_tokens=128):
    """
    Generates summaries for a batch of texts using the fine-tuned model.
    """
    instruction = """Nhiệm vụ của bạn là đọc kỹ bài báo được cung cấp và viết một đoạn tóm tắt ngắn gọn, súc tích nhưng vẫn giữ được các thông điệp cốt lõi.

Hãy áp dụng phương pháp tư duy từng bước (Chain-of-Thought) theo hướng dẫn sau:

1. Phân tích ngữ cảnh: Đọc toàn bộ bài viết và xác định chủ đề chính.
2. Trích xuất thông tin cốt lõi (5W1H):
   - Ai/Cơ quan nào? (Who)
   - Sự kiện/Vấn đề gì đang diễn ra? (What)
   - Thời gian nào? (When)
   - Ở đâu? (Where)
   - Tại sao/Mục đích là gì? (Why/Goal)
3. Lọc bỏ chi tiết phụ: Loại bỏ các số liệu quá chi tiết, trích dẫn trực tiếp dài dòng, hoặc các ví dụ nhỏ lẻ không làm thay đổi nội dung tổng thể.
4. Tổng hợp và Viết tóm tắt: Kết nối các thông tin cốt lõi ở bước 2 thành một đoạn văn hoàn chỉnh, logic, dễ đọc (khoảng 2-3 câu).
""" if cot_enabled else "Tóm tắt văn bản sau đây."

    oneshot = """
## Example
### Instruction:
Tóm tắt văn bản sau đây.

### Input:
Gần 20 sự kiện được tổ chức trên toàn thành phố, kéo dài từ 19/4 đến 10/5. Theo Sở Du lịch Hà Nội, ngoài thu hút du khách, loạt sự kiện cũng là các gợi ý dành cho người dân thủ đô không đi chơi xa và muốn tham gia các hoạt động trong ngày. Một số hoạt động tiêu biểu gồm Lễ hội Du lịch Hà Nội 2024 với chủ đề 'Thăng Long - Hà Nội, Thủ đô quyến rũ'; Triển lãm Ngô Quyền - Anh hùng dân tộc kiệt xuất và Thăng Long hội tụ; Việt Nam - những chiến thắng làm thay đổi dòng chảy lịch sử thế giới hay Tái hiện lễ hội Cầu mưa dân tộc Lô Lô, tỉnh Cao Bằng, Lễ hội Tình yêu năm 2024. Bên cạnh đó, nhân dịp kỷ niệm ngày Giải phóng miền Nam, thống nhất đất nước 30/4 - Quốc tế Lao động 1/5 và Ngày sinh Chủ tịch Hồ Chí Minh 19/5, Sở phối hợp Ban Quản lý Lăng Chủ tịch hỗ trợ nước, sữa và bánh miễn phí phục vụ nhân dân, du khách đến viếng Lăng Bác. Hà Nội cũng có rất nhiều sản phẩm du lịch xanh được ra mắt và đẩy mạnh trong thời gian qua như trải nghiệm xe điện trong lòng phố cổ, tour xe đạp, các sản phẩm du lịch sinh thái và nghỉ dưỡng ở ngoại thành. Nghỉ lễ kéo dài 5 ngày kéo theo nhu cầu tham quan, di chuyển của người dân dự kiến tăng cao. Giám đốc Sở Du lịch Hà Nội, Đặng Hương Giang, cho biết đây là 'cơ hội lớn' cho ngành tăng sức hút với du khách. Sở đã chỉ đạo các bên liên quan rà soát, nâng cao chất lượng sản phẩm, dịch vụ du lịch để đáp ứng nhu cầu của du khách. Hà Nội kỳ vọng với nhiều hoạt động trong kỳ nghỉ lễ nêu trên, ngành du lịch sẽ phục vụ chuyên nghiệp, chu đáo, đáp ứng tốt nhất nhu cầu tham quan, giải trí, nghỉ ngơi của nhân dân và thu hút nhiều khách quốc tế ghé thăm. Phương Anh

### Response:
Hà Nội tổ chức gần 20 sự kiện từ 19/4 đến 10/5, bao gồm Lễ hội Du lịch Hà Nội 2024, các triển lãm và lễ hội văn hóa, nhằm thu hút cả du khách và người dân.  Song song đó, thành phố cũng đẩy mạnh các sản phẩm du lịch xanh và chuẩn bị chu đáo để đáp ứng nhu cầu tăng cao trong kỳ nghỉ lễ 5 ngày, kỳ vọng thu hút nhiều khách du lịch trong và ngoài nước.
    """ if oneshot_enabled else ""

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
{}
### Instruction:
{}

### Input:
{}

### Response:
{}"""

    prompts = [alpaca_prompt.format(oneshot, instruction, text, "") for text in texts]
    
    # Left padding is required for batched generation in causal LMs
    tokenizer.padding_side = "left"
    
    # Ensure pad token is set (some tokenizers like LLaMA don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        input_ids=inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens, 
        use_cache=True, 
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode only the generated part by slicing past the prompt length
    prompt_lengths = inputs.input_ids.shape[1]
    output_texts = tokenizer.batch_decode(outputs[:, prompt_lengths:], skip_special_tokens=True)
    return output_texts

def evaluate_dataset(model, tokenizer, args):
    print(f"Loading test dataset from {args.test_path}...")
    df = pd.read_parquet(args.test_path)
    df = df.dropna(subset=['article', 'summary'])
    
    predictions = []
    references = df['summary'].tolist()
    articles = df['article'].tolist()
    
    print("Generating predictions in batches...")
    for i in tqdm(range(0, len(articles), args.batch_size)):
        batch_articles = articles[i:i+args.batch_size]
        batch_preds = generate_summary_batch(model, tokenizer, batch_articles, args.oneshot_enabled, args.cot_enabled, args.max_new_tokens)
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
    os.makedirs(f"{args.result_dir}", exist_ok=True)
    with open(f"{args.result_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path if os.path.exists(args.model_path) else args.model_name,
            "rouge_bleu_res": rouge_bleu_res,
            "bert_res": bert_res,
            "predictions": predictions, 
            "references": references
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PTM or Generate Summary")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="Name of the base model")
    parser.add_argument("--oneshot_enabled", action="store_true", default=False, help="Enable oneshot")
    parser.add_argument("--cot_enabled", action="store_true", default=False, help="Enable cot")
    parser.add_argument("--max_seq_length", type=int, default=config.MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS, help="Max new tokens")
    parser.add_argument("--dtype", type=str, default=config.DTYPE, help="Data type (e.g., float16, bfloat16)")
    parser.add_argument("--load_in_4bit", type=bool, default=config.LOAD_IN_4BIT, help="Load in 4bit")
    parser.add_argument("--model_path", type=str, default=f"{config.OUTPUT_DIR}/lora_model", help="Path to saved LoRA model. If not provided, uses the original pretrained model.")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test parquet dataset for evaluation")
    parser.add_argument("--text", type=str, default=None, help="Text to summarize (single inference)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataset evaluation")
    parser.add_argument("--result_dir", type=str, default="outputs", help="Directory to save results")
    args = parser.parse_args()
    
    if args.test_path is None and args.text is None:
        raise ValueError("Must provide either --test_path for evaluation or --text for single inference.")
    
    # Determine correct torch dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Set up quantization if load_in_4bit is requested
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4", # Standard optimal format for 4-bit
            bnb_4bit_use_double_quant=True
        )

    print(f"Loading base model: {args.model_name}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # 3. Apply LoRA weights if model_path exists
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading LoRA weights from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print("Running base model (No LoRA adapter found).")
        
    # Set to evaluation mode
    model.eval()

    if args.test_path:
        evaluate_dataset(model, tokenizer, args)
    elif args.text:
        print("\n--- Summary ---")
        generate_summary(model, tokenizer, args.text, args.oneshot_enabled, args.cot_enabled, args.max_new_tokens, stream=True)
        print("\n---------------")