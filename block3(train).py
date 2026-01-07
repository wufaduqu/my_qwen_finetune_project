from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset # 確保這裡導入了 Dataset

# 定義模型名稱
model_name = "Qwen/Qwen3-0.6B"

# 載入 Tokenizer 並修正錯誤
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Qwen models don't have a pad token by default, add one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # 使用 add_special_tokens 而不是直接賦值

# Helper function to format a single conversation entry for Qwen
def format_single_conversation(conversation_list):
    formatted_string = ""
    for turn in conversation_list:
        role = turn["from"]
        content = turn["value"]
        formatted_string += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted_string

# 準備資料
# 我們需要將對話資料轉換成模型可以理解的格式 (token IDs)
def format_dataset(examples):
    # 將 `conversations` 欄位中的每個對話列表轉換為模型可訓練的單一字串
    formatted_texts = []
    for conv_list in examples['conversations']:
        formatted_texts.append(format_single_conversation(conv_list))
    return {"text": formatted_texts}

# 將資料集轉換為 Hugging Face Datasets 格式
# 使用之前生成的 converted_dataset 變數
# converted_dataset 應該是一個列表 of dictionaries，每個字典包含 'id' 和 'conversations' 鍵
hf_dataset = Dataset.from_list(converted_dataset)

# 應用格式化函數
hf_dataset = hf_dataset.map(format_dataset, batched=True)

# Tokenize 資料集
def tokenize_function(examples):
    # 使用 tokenizer 對文本進行編碼
    # padding='max_length' 會將序列填充到最大長度
    # truncation=True 會截斷超過最大長度的序列
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy() # For causal language modeling, labels are the input_ids
    return tokenized_inputs

# 移除原始的 'id' 和 'conversations' 欄位，保留 'text' 欄位給後續處理
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["id", "conversations", "text"])

# 分割訓練集和驗證集 (可選)
# 如果您的資料集較大，可以分割一部分用於驗證
# dataset = tokenized_dataset.train_test_split(test_size=0.1)
# train_dataset = dataset['train']
# eval_dataset = dataset['test']
train_dataset = tokenized_dataset # 如果不分割，就使用整個資料集作為訓練集


# 載入模型 (使用 4-bit 量化以節省記憶體)
# 需要確保 CUDA 可用才能使用 4-bit 量化
bnb_config = None
if torch.cuda.is_available():
    #from transformers import BitsAndBytesConfig # Move this import to the top
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("將使用 4-bit 量化載入模型")
else:
    print("CUDA 不可用，將使用標準方式載入模型 (可能需要更多記憶體)")


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" if torch.cuda.is_available() else None # 自動分配到可用的裝置 (GPU 或 CPU)
)

# 設置 PEFT (LoRA)
lora_config = LoraConfig(
    r=8, # LoRA 的秩
    lora_alpha=16, # LoRA 的縮放因子
    target_modules=["q_proj", "v_proj"], # 應用 LoRA 的層 (Qwen 模型可能需要調整)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./qwen_finetuned", # 輸出目錄
    per_device_train_batch_size=2, # 每個裝置的訓練批量大小 (根據您的 GPU 記憶體調整)
    gradient_accumulation_steps=4, # 梯度累計步數
    learning_rate=2e-4, # 學習率
    num_train_epochs=3, # 訓練輪數
    logging_dir="./logs", # 日誌目錄
    logging_steps=10, # 每隔多少步記錄一次日誌
    save_steps=50, # 每隔多少步保存一次模型
    save_total_limit=2, # 最多保存的模型數量
    push_to_hub=False, # 是否推送到 Hugging Face Hub
    report_to="none" # 不報告到任何平台
)

# 創建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset # 如果有驗證集
)

# 開始訓練
print("開始微調模型...")
trainer.train()

print("模型微調完成！")