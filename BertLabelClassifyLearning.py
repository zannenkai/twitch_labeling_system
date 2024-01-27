import json
import random
import glob
from tqdm import tqdm
import os
import datetime
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger  # TensorBoardLoggerをインポート

# 現在の日付を取得
now = datetime.datetime.now()
# 月と日を取得（1桁の場合は0埋め）
month = str(now.month).zfill(2)
day = str(now.day).zfill(2)
hour = str(now.hour).zfill(2)
minute = str(now.minute).zfill(2)

# 月と日を結合して指定の形式で表示
date_str = month + day + "_" + hour + minute

# プロジェクトのディレクトリが存在しない場合、作成する
if not os.path.exists(date_str):
    os.makedirs(date_str)

# カレントディレクトリの変更
os.chdir(date_str)

# TensorBoardログディレクトリのパス
log_dir = 'logs'  # 任意のディレクトリを指定

# ログディレクトリが存在しない場合、作成する
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# TensorBoardLoggerを設定
tensorboard_logger = TensorBoardLogger(save_dir=log_dir, name='my_model')
# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# トークナイザのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

max_length = 128
dataset_for_loader = []

# data.jsonファイルのパス
data_file = input("データファイルのパスを入力してください: ")

weight_decay = 0.01
max_epochs = 10
train_batch_size = 16
val_batch_size = 32
test_batch_size = 32
dropout_ratio = 0.5
learning_rate = 1e-5

# data.jsonファイルの読み込み
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)
# ラベルのマッピング辞書
# JSONファイルからラベルマップを読み込む
# with open(r"../labels.json", 'r', encoding='utf-8') as f:
with open(r"../labels.json", 'r', encoding='utf-8') as f:
    label_map = json.load(f)

# 各labelごとのコメントの数を数える
label_counts = Counter(item['label'] for item in data)

# data.jsonの各データを整形
for item in data:
    text = item["comment"]
    label = item["label"]

    label_id = label_map[label]

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    encoding["labels"] = torch.tensor(label_id)  # ラベルを整数に変換
    encoding = {k: torch.tensor(v) for k, v in encoding.items()}
    dataset_for_loader.append(encoding)
# 6-13
# データセットの分割
random.shuffle(dataset_for_loader) # ランダムにシャッフル
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train] # 学習データ
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

# データセットからデータローダを作成
# 学習データはshuffle=Trueにする。
dataloader_train = DataLoader(
    dataset_train, batch_size=train_batch_size, shuffle=True
)
dataloader_val = DataLoader(dataset_val, batch_size=val_batch_size)
dataloader_test = DataLoader(dataset_test, batch_size=test_batch_size)

# パラメータをparameters.txtに保存する
with open("parameters.txt", "w") as f:
    f.write(f"weight_decay={weight_decay}\n")
    f.write(f"max_epochs={max_epochs}\n")
    f.write(f"train_batch_size={train_batch_size}\n")
    f.write(f"val_batch_size={val_batch_size}\n")
    f.write(f"test_batch_size={test_batch_size}\n")
    f.write(f"dropout_ratio={dropout_ratio}\n")
    f.write(f"learning_rate={learning_rate}\n")
    f.write(f"dataset_num={len(dataset_for_loader)}\n")
    # 結果を表示
    for label, count in label_counts.items():
        f.write(f'{label}: {count}\n')
# 6-14
class BertForSequenceClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()

        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters()

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        # 追加：ドロップアウトと分類器の初期化
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.classifier = torch.nn.Linear(self.bert_sc.config.hidden_size, num_labels)

        self.val_labels_correct = [0] * num_labels
        self.val_labels_total = [0] * num_labels

        self.test_labels_correct = [0] * num_labels
        self.test_labels_total = [0] * num_labels

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        labels = batch["labels"]
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)  # 正答率
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss

        labels = batch["labels"]
        labels_predicted = output.logits.argmax(-1)

        # ラベルごとの正答数とサンプル数を更新
        for i in range(self.hparams.num_labels):
            self.val_labels_correct[i] += ((labels_predicted == labels) & (labels == i)).sum().item()
            self.val_labels_total[i] += (labels == i).sum().item()

        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)  # 正答率
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
    # 検証データのエポックが終了した後に呼ばれるメソッド
    def on_validation_epoch_end(self):
        # ラベルごとの正答率を計算してTensorBoardにログを記録
        for i in range(len(self.val_labels_correct)):
            label_accuracy = self.val_labels_correct[i] / max(1, self.val_labels_total[i])
            label_name = f'val_accuracy_label_{i}'
            self.log(label_name, label_accuracy, prog_bar=True)
            self.log(f'val_num_accuracy_label_{i}',  torch.tensor(self.val_labels_total[i], dtype=torch.float32), prog_bar=True)

        # ラベルごとの正答数とサンプル数をリセット
        self.val_labels_correct = [0] * len(self.val_labels_correct)
        self.val_labels_total = [0] * len(self.val_labels_total)


    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)

        # ラベルごとの正答数とサンプル数を更新
        for i in range(self.hparams.num_labels):
            self.test_labels_correct[i] += ((labels_predicted == labels) & (labels == i)).sum().item()
            self.test_labels_total[i] += (labels == i).sum().item()

        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct / labels.size(0)  # 精度
        # ロスの計算
        loss = F.cross_entropy(output.logits, labels)

        self.log('accuracy', accuracy)  # 精度を'accuracy'の名前でログをとる。
        self.log('loss', loss)  # ロスを'loss'の名前でログをとる。

    # テストデータのエポックが終了した後に呼ばれるメソッド
    def on_test_epoch_end(self):
        # ラベルごとの正答率を計算してTensorBoardにログを記録
        for i in range(len(self.test_labels_correct)):
            label_accuracy = self.test_labels_correct[i] / max(1, self.test_labels_total[i])
            label_name = f'test_accuracy_label_{i}'
            self.log(label_name, label_accuracy, prog_bar=True)

        # ラベルごとの正答数とサンプル数をリセット
        self.test_labels_correct = [0] * len(self.test_labels_correct)
        self.test_labels_total = [0] * len(self.test_labels_total)
    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,weight_decay=weight_decay)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_sc(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
# 6-15
# 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_accuracy',
    mode='max',
    # monitor='val_loss',
    # mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

# 学習の方法を指定
trainer = pl.Trainer(
    accelerator='gpu',
    max_epochs=max_epochs,
    callbacks = [checkpoint],
    logger=tensorboard_logger
)
# 6-16
# PyTorch Lightningモデルのロード
model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=len(label_map), lr=learning_rate
)

# ファインチューニングを行う。
trainer.fit(model, dataloader_train, dataloader_val)
# 6-17
best_model_path = checkpoint.best_model_path # ベストモデルのファイル
print('ベストモデルのファイル: ', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)
# 6-19
test = trainer.test(dataloaders=dataloader_test)
print(f'Accuracy: {test[0]["accuracy"]:.2f}')
# 6-20
# PyTorch Lightningモデルのロード
model = BertForSequenceClassification_pl.load_from_checkpoint(

    best_model_path
)


accuracy = int(test[0]["accuracy"] * 100)

model.bert_sc.save_pretrained(f"./comment_classification_bert_{date_str}_{accuracy}")


# bert_sc = BertForSequenceClassification.from_pretrained(
#     r'hoge\モデル\comment_classification_bert'
# )

