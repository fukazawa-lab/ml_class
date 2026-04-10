from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import japanize_matplotlib
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
from datasets import Dataset, ClassLabel
from datasets import load_dataset
import numpy as np
import pandas as pd


from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset

plt.rcParams["font.size"] = 12  # 文字サイズを大きくする

def visualize_text_length(dataset: Dataset):
    # モデル名を指定してトークナイザを読み込む
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """データセット中のテキストのトークン数の分布をグラフとして描画"""
    # データセット中のテキストの長さを数える
    length_counter = Counter()
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["sentence"]))
        length_counter[length] += 1
    # length_counterの値から棒グラフを描画する
    # グラフのサイズを設定
    plt.figure(figsize=(4, 3))  # 幅 8 インチ、高さ 6 インチ

    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("事例数")
    plt.show()



import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def visualize_labels(file_name, column_name):
    """指定されたファイルのカラム内のラベル分布をグラフとして描画"""
    # ファイルを読み込む
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"エラー: ファイル {file_name} が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: ファイル {file_name} を読み込む際にエラーが発生しました: {e}")
        return

    # 指定されたカラムが存在するかを確認
    if column_name not in df.columns:
        print(f"エラー: ファイル {file_name} に {column_name} カラムが含まれていません。")
        return

    # データセット中のラベルの数を数える
    label_counter = Counter(df[column_name])

    # ラベルIDをラベル名に変換するための辞書（任意で変更可能）
    label_id_to_name = {label_id: f"ラベル{label_id}" for label_id in label_counter.keys()}

    # グラフのサイズを設定
    plt.figure(figsize=(len(label_counter) * 1.5, 3))  # 幅がラベルの数に応じて調整される

    # label_counterを棒グラフとして描画する
    label_names = [label_id_to_name[label_id] for label_id in label_counter.keys()]
    label_counts = list(label_counter.values())
    plt.bar(label_names, label_counts, width=0.6)
    plt.xlabel("ラベル")
    plt.ylabel("事例数")
    plt.show()




# 新しく追加する関数
def check_class_difference(train_csv, validation_csv, target_column):
    # CSVファイルを読み込み
    train_df = pd.read_csv(train_csv)
    validation_df = pd.read_csv(validation_csv)

    # 各データセットのtargetカラムのユニークなクラスを取得
    train_classes = set(train_df[target_column].unique())
    validation_classes = set(validation_df[target_column].unique())

    # どちらかにしかないクラスを取得
    train_only_classes = train_classes - validation_classes
    validation_only_classes = validation_classes - train_classes

    # 結果を表示
    if train_only_classes or validation_only_classes:
        if train_only_classes:
            print(f"Trainデータセットにのみ存在するクラス: {train_only_classes}")
        if validation_only_classes:
            print(f"Validationデータセットにのみ存在するクラス: {validation_only_classes}")
    else:
        print("両方のデータセットでクラスが一致しています。")


# 新しく追加する関数
def check_class_difference(train_csv, validation_csv, target_column):
    # CSVファイルを読み込み
    train_df = pd.read_csv(train_csv)
    validation_df = pd.read_csv(validation_csv)

    # 各データセットのtargetカラムのユニークなクラスを取得
    train_classes = set(train_df[target_column].unique())
    validation_classes = set(validation_df[target_column].unique())

    # どちらかにしかないクラスを取得
    train_only_classes = train_classes - validation_classes
    validation_only_classes = validation_classes - train_classes

    # 結果を表示
    if train_only_classes or validation_only_classes:
        if train_only_classes:
            print(f"エラー：Trainデータセットにのみ存在するクラス: {train_only_classes}")
        if validation_only_classes:
            print(f"エラー：Validationデータセットにのみ存在するクラス: {validation_only_classes}")
    else:
        print("両方のデータセットでクラスが一致しています。")


import pandas as pd

def validate_target_column(file_name):
    # ファイルを読み込む
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"エラー: ファイル {file_name} が見つかりません。")
        return False
    except Exception as e:
        print(f"エラー: ファイル {file_name} を読み込む際にエラーが発生しました: {e}")
        return False

    # targetカラムが存在するかを確認
    if 'label' not in df.columns:
        print(f"エラー: ファイル {file_name} に target カラムが含まれていません。")
        return False

    target_series = df['label']

    # 1. 整数以外が入っていないことを確認
    if not pd.api.types.is_integer_dtype(target_series):
        print(f"エラー: ファイル {file_name} の target カラムに整数以外の値が含まれています")
        return False

    # 2. 最小値が0であることを確認
    min_value = target_series.min()
    if min_value != 0:
        print(f"エラー: ファイル {file_name} の target カラムの最小値は {min_value} です。0から始めてください。")
        return False

    # 3. 最小値と最大値の間に漏れている整数がないことを確認
    max_value = target_series.max()
    missing_classes = set(range(min_value, max_value + 1)) - set(target_series.unique())
    if missing_classes:
        print(f"エラー: ファイル {file_name} にクラスが漏れています。以下のクラスが存在しません: {missing_classes}")
        return False

    print(f"{file_name}のデータセットのtargetカラムが正しくフォーマットされています。")

    return True
