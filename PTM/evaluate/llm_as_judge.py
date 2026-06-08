import argparse
import os
import re
import pandas as pd
import json
import time
from dotenv import load_dotenv
from google.genai import types
from google import genai
from pydantic import BaseModel, ValidationError

EVALUATION_PROMPT_TEMPLATE = """
Bạn sẽ được cung cấp một bài báo gốc và 4 bản tóm tắt của nó. Nhiệm vụ của bạn là đánh giá 4 bản tóm tắt này dựa trên 4 tiêu chí khác nhau.
Vui lòng đọc và hiểu kỹ các hướng dẫn này.

Tiêu chí và Các bước đánh giá:

1. Relevance (Sự liên quan).
Tiêu chí: Bản tóm tắt chỉ nên bao gồm những thông tin quan trọng từ tài liệu gốc. Hãy trừ điểm đối với các bản tóm tắt chứa thông tin dư thừa, lặp lại và không cần thiết.
Các bước đánh giá:
1. Đọc kỹ bản tóm tắt và tài liệu gốc.
2. So sánh bản tóm tắt với tài liệu gốc và xác định các ý chính của bài báo.
3. Đánh giá mức độ bản tóm tắt bao quát các ý chính của bài báo, và mức độ chứa thông tin dư thừa hoặc không liên quan.
4. Chấm điểm sự liên quan bằng một số nguyên từ 0 đến 100.

2. Coherence (Tính mạch lạc).
Tiêu chí: Bản tóm tắt phải được cấu trúc và tổ chức tốt. Bản tóm tắt không nên chỉ là một tập hợp các thông tin rời rạc, mà phải được liên kết từ câu này sang câu khác thành một khối thông tin mạch lạc về chủ đề.
Các bước đánh giá:
1. Đọc kỹ bài báo và xác định chủ đề chính cùng các ý chính.
2. Đọc bản tóm tắt và so sánh với bài báo. Kiểm tra xem bản tóm tắt có bao quát chủ đề chính và các ý chính của bài báo hay không, và liệu nó có trình bày chúng theo một trình tự rõ ràng và hợp lý không.
3. Chấm điểm tính mạch lạc theo thang điểm từ 0 đến 100, trong đó 0 là thấp nhất và 100 là cao nhất dựa trên Tiêu chí đánh giá.

3. Consistency (Tính nhất quán).
Tiêu chí: Một bản tóm tắt nhất quán về mặt thực tế chỉ chứa các câu khẳng định được hỗ trợ và suy ra từ tài liệu gốc. Hãy trừ điểm nặng đối với các bản tóm tắt chứa thông tin bịa đặt, sai lệch, hoặc không có trong bài gốc.
Các bước đánh giá:
1. Đọc kỹ bài báo và xác định các sự kiện và chi tiết chính mà nó trình bày.
2. Đọc bản tóm tắt và so sánh với bài báo. Kiểm tra xem bản tóm tắt có chứa bất kỳ lỗi sai sự thật hoặc thông tin nào không được bài báo hỗ trợ hay không.
3. Chấm điểm tính nhất quán từ 0 đến 100 dựa trên Tiêu chí đánh giá.

4. Fluency (Tính trôi chảy).
Tiêu chí: Chất lượng của bản tóm tắt về mặt ngữ pháp, chính tả, dấu câu, cách chọn từ và cấu trúc câu.
- 0-33: Kém. Bản tóm tắt có nhiều lỗi khiến văn bản khó hiểu hoặc đọc nghe rất thiếu tự nhiên.
- 34-66: Khá. Bản tóm tắt có một số lỗi ảnh hưởng đến sự rõ ràng hoặc trôi chảy của văn bản, nhưng các ý chính vẫn có thể hiểu được.
- 67-100: Tốt. Bản tóm tắt có rất ít hoặc không có lỗi, hành văn tự nhiên, dễ đọc và dễ theo dõi.
Các bước đánh giá: Đọc bản tóm tắt và đánh giá tính trôi chảy của nó dựa trên các tiêu chí đã cho. Chấm điểm tính trôi chảy bằng một số nguyên từ 0 đến 100.

Dữ liệu:

Văn bản gốc:

{document}

Bản tóm tắt 1:

{summary1}

Bản tóm tắt 2:

{summary2}

Bản tóm tắt 3:

{summary3}

Bản tóm tắt 4:

{summary4}

Đánh giá:
Chỉ cung cấp ĐIỂM SỐ là các số nguyên từ 0 đến 100 cho mỗi tiêu chí, tương ứng với thứ tự của các bản tóm tắt.
Định dạng đầu ra bắt buộc phải là một đối tượng JSON hợp lệ có chứa 4 mảng điểm số cho 'Relevance', 'Coherence', 'Consistency', và 'Fluency'.
Ví dụ định dạng:
{{
  "Relevance": [score1, score2, score3, score4],
  "Coherence": [score1, score2, score3, score4],
  "Consistency": [score1, score2, score3, score4],
  "Fluency": [score1, score2, score3, score4]
}}
Không xuất ra gạch đầu dòng, khối mã markdown, hoặc bất kỳ lời giải thích nào khác. Chỉ xuất đúng định dạng JSON được yêu cầu.
"""

class EvaluationScores(BaseModel):
    Relevance: list[int]
    Coherence: list[int]
    Consistency: list[int]
    Fluency: list[int]

def get_geval_scores(client, model_name, document: str, summary1: str, summary2: str, summary3: str, summary4: str):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        document=document,
        summary1=summary1,
        summary2=summary2,
        summary3=summary3,
        summary4=summary4,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
            top_p=1.0,
            response_mime_type="application/json",
            response_schema=EvaluationScores,
        )
    )

    return response.text

def process_evaluation_response(response_text: str):
    try:
        parsed_data = EvaluationScores.model_validate_json(response_text)

        # Verify all metrics have exactly 4 scores
        metrics_dict = parsed_data.model_dump()
        for metric, scores in metrics_dict.items():
            if len(scores) != 4:
                print(f"Cảnh báo: Mong đợi 4 điểm số cho {metric}, nhận được {len(scores)}.")
                return None

        print("✅ Đánh giá thành công 4 tiêu chí.")
        return metrics_dict # Returns a dict: {"Relevance": [...], "Coherence": [...], ...}

    except ValidationError as e:
        print(f"❌ Lỗi cấu trúc dữ liệu.\nChi tiết: {e}")
        return None
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return None

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="LLM as a Judge Evaluation using Gemini API")
    parser.add_argument("--api_key", type=str, default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key")
    parser.add_argument("--input_path", type=str, default="/content/valid-sampled-predictions.parquet", help="Path to predictions parquet")
    parser.add_argument("--output_path", type=str, default="llm_judge_results.csv", help="Path to save evaluation results pivot table")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Gemini model name")
    
    parser.add_argument("--requests_per_minute_limit", type=int, default=15, help="Rate limit requests per minute")
    parser.add_argument("--window_duration_seconds", type=int, default=60, help="Rate limit window in seconds")
    
    # Columns setup
    parser.add_argument("--article_col", type=str, default="article", help="Column name for article")
    parser.add_argument("--base_pred_col", type=str, default="base_pred", help="Column name for summary 1 (Base Predictions)")
    parser.add_argument("--lora_pred_col", type=str, default="lora_pred", help="Column name for summary 2 (Lora Predictions)")
    parser.add_argument("--dpo_b_pred_col", type=str, default="dpo_b_pred", help="Column name for summary 3 (DPO bottom set)")
    parser.add_argument("--dpo_f_pred_col", type=str, default="dpo_f_pred", help="Column name for summary 4 (DPO full set)")
    
    args = parser.parse_args()

    # Khởi tạo client
    client = genai.Client(api_key=args.api_key)

    sampled_df = pd.read_parquet(args.input_path)
    articles = sampled_df[args.article_col].to_list()
    summary_1 = sampled_df[args.base_pred_col].to_list()
    summary_2 = sampled_df[args.lora_pred_col].to_list()
    summary_3 = sampled_df[args.dpo_b_pred_col].to_list()
    summary_4 = sampled_df[args.dpo_f_pred_col].to_list()

    model_names = [
        "Base Predictions",
        "Lora Predictions",
        "DPO (bottom set) Predictions",
        "DPO (full set) Predictions"
    ]
    metric_names = ["Relevance", "Coherence", "Consistency", "Fluency"]

    # Khởi tạo accumulator cho toàn bộ dữ liệu
    accumulators = {
        metric: {name: [] for name in model_names} for metric in metric_names
    }

    requests_made = 0
    window_start_time = time.time()

    num_samples = len(articles) # Chạy toàn bộ dữ liệu
    print(f"🚀 Bắt đầu đánh giá G-Eval trên {num_samples} mẫu văn bản...")
    print(f"⏱️ Đã kích hoạt giới hạn: {args.requests_per_minute_limit} requests / {args.window_duration_seconds} giây.")

    for idx in range(num_samples):
        print(f"\n⚡ Đang đánh giá bài báo thứ {idx + 1}/{num_samples}...")

        # -- Logic kiểm tra Rate Limit trước khi gọi API --
        current_time = time.time()
        elapsed_time = current_time - window_start_time

        # Nếu đã qua thời gian window (vd: 60s), reset lại cửa sổ thời gian và bộ đếm
        if elapsed_time >= args.window_duration_seconds:
            requests_made = 0
            window_start_time = current_time
        # Nếu chưa qua window nhưng đã chạm ngưỡng requests limit
        elif requests_made >= args.requests_per_minute_limit:
            sleep_time = args.window_duration_seconds - elapsed_time
            print(f"   [⏳ RATE LIMIT] Đã đạt {args.requests_per_minute_limit} requests. Tạm dừng {sleep_time:.2f} giây...")
            time.sleep(sleep_time)
            # Reset lại sau khi thức dậy
            requests_made = 0
            window_start_time = time.time()

        # Tăng bộ đếm request
        requests_made += 1
        # ------------------------------------------------

        try:
            # Gọi API tính cả 4 tiêu chí cho 4 model
            result_text = get_geval_scores(
                client=client,
                model_name=args.model,
                document=articles[idx],
                summary1=summary_1[idx],
                summary2=summary_2[idx],
                summary3=summary_3[idx],
                summary4=summary_4[idx]
            )

            scores_dict = process_evaluation_response(result_text)

            if scores_dict:
                # Nếu thành công, đưa điểm vào accumulator
                for metric in metric_names:
                    for i, model_name in enumerate(model_names):
                        accumulators[metric][model_name].append(scores_dict[metric][i])

        except Exception as e:
            print(f"❌ Lỗi trong quá trình xử lý bài báo {idx + 1}: {e}")

    # Chuẩn bị Data format chuẩn để Pivot
    data = {"Evaluation Type": [], "Summary Type": [], "Score": []}

    for metric in metric_names:
        for model_name in model_names:
            # Tính trung bình cộng
            score_list = accumulators[metric][model_name]
            avg_score = sum(score_list) / len(score_list) if score_list else 0

            data["Evaluation Type"].append(metric)
            data["Summary Type"].append(model_name)
            data["Score"].append(round(avg_score, 2))

    # Tiến hành Pivot table phẳng hàng
    pivot_df = pd.DataFrame(data).pivot(
        index="Evaluation Type", columns="Summary Type", values="Score"
    )

    # Keep the order
    pivot_df = pivot_df[model_names]

    # Display the results
    print("\n--- KẾT QUẢ ĐÁNH GIÁ LLM AS A JUDGE ---")
    print(pivot_df)

    # Save to CSV
    pivot_df.to_csv(args.output_path)
    print(f"\n✅ Đã lưu kết quả đánh giá (pivot table) vào {args.output_path}")

if __name__ == "__main__":
    main()
