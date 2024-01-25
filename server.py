import asyncio
import websockets
import json
import os
from datetime import datetime

import BertLabelClassification

# ログファイルのパス
log_dir = "labeled_logs"
log_file_path = None

# WebSocket接続を処理するコルーチン
async def handle_socket(websocket, path):
    global log_file_path

    # WebSocket接続が確立されたときの処理
    print("WebSocket connection established.")

    # 起動時に一回だけログファイルを用意
    if log_file_path is None:
        log_file_path = prepare_log_file()

    # メッセージを受信するループ
    async for message in websocket:
        data = json.loads(message)
        print("Received message from JavaScript: ", message)
        result = classifyLabel(data["data"])
        predictions = [
            {"label": result[i]["label"], "rate": result[i]["rate"]}
            for i in range(3)  # 上位3位までを送信
        ]
        data["predictions"] = predictions

        # 結果を送信する
        await websocket.send(json.dumps(data))
        print("Sent response to JavaScript: ", data)

        # その他以外の場合、ログに追記
        if predictions[0]["label"] != "その他":
            log_result(data)

def classifyLabel(comment):
    result = classifier.classify_text(comment)
    return result

def prepare_log_file():
    # ログディレクトリが存在しない場合は作成
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ログファイルのパスを生成
    now = datetime.now().strftime("%m%d_%H%M")
    log_path = os.path.join(log_dir, f"{now}.json")

    return log_path

def log_result(data):
    # 結果をログファイルに追記
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        json.dump(data, log_file, ensure_ascii=False)
        log_file.write("\n")

# WebSocketサーバーを起動する
start_server = websockets.serve(handle_socket, "localhost", 8000)

# ラベルのマッピング辞書
# JSONファイルからラベルマップを読み込む
with open(r"C:\pd3\projects\labels.json", 'r', encoding='utf-8') as f:
    label_map = json.load(f)

# モデルのパス
model_path = 'comment_classification_bert_0109_1601_74'

# BertLabelClassificationのインスタンスを作成
classifier = BertLabelClassification.BertLabelClassification(model_path, label_map)

# イベントループを実行する
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
