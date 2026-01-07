# 這裡假設您的新資料檔案已經是目標格式 (例如：由 previous `convert_to_conversation_format` 函數產生的格式)
# 請將 'your_new_formatted_data.jsonl' 替換為您的實際檔案路徑和名稱

import json
import os
from datasets import Dataset # 確保這裡導入了 Dataset

new_data_file_path = 'output_data.jsonl' # 請修改為您的新資料檔案路徑你有注意到

print(f"正在載入新資料檔案 '{new_data_file_path}'...")

try:
    with open(new_data_file_path, 'r', encoding='utf-8') as f:
        converted_dataset = [json.loads(line) for line in f]
    print(f"成功載入 {len(converted_dataset)} 條新格式資料。")

    # 印出轉換後的前幾個樣本來檢查格式
    print("載入的新資料集範例：")
    for i, sample in enumerate(converted_dataset[:5]):
        print(f"樣本 {i+1}: {sample}")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{new_data_file_path}'。請確認檔案路徑是否正確。")
    converted_dataset = [] # 設定為空列表以避免後續錯誤
except json.JSONDecodeError as e:
    print(f"錯誤：無法解析您的新資料檔案 '{new_data_file_path}'。請確認檔案是有效的 JSON Lines 格式。")
    print(f"錯誤訊息：{e}")
    converted_dataset = [] # 設定為空列表以避免後續錯誤
except Exception as e:
    print(f"載入新資料集時發生其他錯誤：{e}")
    converted_dataset = [] # 設定為空列表以避免後續錯誤

# 如果您希望將載入的資料集保存為 Hugging Face Dataset 格式，可以取消註解以下程式碼
# hf_dataset = Dataset.from_list(converted_dataset)
# print("新資料已載入到 'converted_dataset' 變數中，可供後續模型訓練使用。")