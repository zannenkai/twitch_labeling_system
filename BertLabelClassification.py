from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import torch

import json

class BertLabelClassification:
    def __init__(self, model_path, label_map):
        # ラベルのマッピング辞書
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        
        # 日本語の事前学習モデル
        self.MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        
        # モデルのロード
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # トークナイザのロード
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.MODEL_NAME)

    def classify_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        outputs = self.model(input_ids, attention_mask)
        predictions = torch.softmax(outputs.logits, dim=1).tolist()[0]

        # 上位3位のクラスと確信度を取得
        top_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:3]

        # 結果を連想配列で整形
        result = [
            {"label": self.reverse_label_map.get(index, "Unknown"), "rate": round(score,4)}
            for index, score in top_predictions
        ]
        
        return result

# 以下は使用例
# ラベルのマッピング辞書
# JSONファイルからラベルマップを読み込む
# with open(r"C:\pd3\projects\labels.json", 'r', encoding='utf-8') as f:
#     label_map = json.load(f)

# # モデルのパス
# model_path = r'C:\pd3\projects\1124_1445\comment_classification_bert_1124_1445_61'

# # BertLabelClassificationのインスタンスを作成
# classifier = BertLabelClassification(model_path, label_map)

# while True:
#     # 推論
#     text = input("text:")
#     result = classifier.classify_text(text)
#     print(result)
