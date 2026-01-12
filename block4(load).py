from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch
import os

# 定義模型名稱 (應與訓練時使用的基礎模型名稱一致)
base_model_name = "Qwen/Qwen3-0.6B"

# 微調後的模型輸出路徑 (應與 TrainingArguments 中的 output_dir 一致)
# 修正：LoRA 權重會儲存在 output_dir 下的特定子資料夾中。
# 根據訓練程式碼，最終 LoRA adapter 儲存為 'final_qwen_lora_adapter'。
finetuned_model_path = "./qwen_finetuned/final_qwen_lora_adapter" # <-- 這裡已修正路徑

# 載入 Tokenizer
test_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if test_tokenizer.pad_token is None:
    test_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

# 載入基礎模型
# 由於我們只保存了 LoRA 的權重，需要先載入基礎模型
# 為了保持一致性，如果訓練時使用了 4-bit 量化，這裡也應該使用
bnb_config = None
if torch.cuda.is_available():
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("將使用 4-bit 量化載入基礎模型")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto" if torch.cuda.is_available() else None # 自動分配到可用的裝置 (GPU 或 CPU)
)

# 載入 PEFT (LoRA) 權重
# 注意：finetuned_model_path 應該是指向包含 adapter_model.safetensors 等文件的資料夾
# 如果您的 checkpoint-XXX 資料夾是 LoRA 權重，則直接載入該資料夾

# 檢查是否有 LoRA 權重文件 (例如 adapter_model.safetensors) 在微調路徑下
# 如果 TrainingArguments 的 output_dir 是 './qwen_finetuned'，那麼權重應該在該資料夾中
if os.path.exists(os.path.join(finetuned_model_path, "adapter_model.safetensors")) or \
   os.path.exists(os.path.join(finetuned_model_path, "adapter_model.bin")):
    print(f"從 '{finetuned_model_path}' 載入 LoRA 權重...")
    model_to_test = PeftModel.from_pretrained(base_model, finetuned_model_path)
    print("LoRA 權重載入成功。")
else:
    # 如果找不到 LoRA 權重，這表示路徑可能有誤，或者 LoRA 權重沒有被保存
    # 在此範例中，應該要能找到 LoRA 權重，所以這個情況應該表示路徑有問題
    print(f"錯誤：在 '{finetuned_model_path}' 中找不到 LoRA 權重。請確認路徑是否正確，或模型是否已成功微調並保存。")
    print("將使用基礎模型進行推理 (未微調)。")
    model_to_test = base_model

model_to_test.eval()

print("模型載入完成並設定為評估模式。")
