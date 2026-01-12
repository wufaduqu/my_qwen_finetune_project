import re

def generate_response(prompt_text, model, tokenizer, max_new_tokens=256):
    # 格式化輸入為 Qwen 預期的對話格式
    system_prompt = "你是一位名為萊拉的女僕，個性溫和且開朗，平時會幫忙打掃、做家事"
    formatted_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tokenizer.encode(formatted_input, return_tensors="pt")

    if torch.cuda.is_available():
        input_ids = input_ids.cuda() # 將輸入移到 GPU 上

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id, # 使用之前設定的 pad_token_id
            do_sample=True, # 啟用抽樣生成，增加多樣性
            temperature=0.7, # 調整生成溫度
            top_p=0.9,       # Top-p 抽樣
            eos_token_id=tokenizer.eos_token_id
        )

    # 解碼生成的 token
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取助手的回應部分
    assistant_start_tag = '<|im_start|>assistant\n'
    assistant_end_tag = '<|im_end|>'

    assistant_start_index = decoded_output.find(assistant_start_tag)
    if assistant_start_index != -1:
        raw_assistant_response = decoded_output[assistant_start_index + len(assistant_start_tag):]

        # 移除 <|im_end|> 標籤及其之後的內容
        assistant_end_index = raw_assistant_response.find(assistant_end_tag)
        if assistant_end_index != -1:
            raw_assistant_response = raw_assistant_response[:assistant_end_index].strip()
        else:
            raw_assistant_response = raw_assistant_response.strip()

        # --- 精簡的後處理邏輯：直接移除 <think> 和 </think> 標籤及其內容 --- #
        # 使用正則表達式匹配 <think>...</think> 及其內容，並將其替換為空字串
        # re.DOTALL 讓 . 也能匹配換行符號
        cleaned_response = re.sub(r'<think>.*?<\/think>', '', raw_assistant_response, flags=re.DOTALL).strip()

        # 再次清理，確保沒有多餘的換行和空白
        assistant_response = re.sub(r'\n+', '\n', cleaned_response).strip()
        assistant_response = re.sub(r'\s{2,}', ' ', assistant_response).strip() # Replace multiple spaces with single space

    else:
        assistant_response = "無法解析助手回應。"

    return assistant_response

# 測試範例
print("請輸入您的測試對話（輸入 'exit' 結束）：")
while True:
    user_input = input("您：")
    if user_input.lower() == 'exit':
        break

    response = generate_response(user_input, model_to_test, test_tokenizer)
    print(f"模型：{response}")