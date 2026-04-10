
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
from sklearn.utils import resample

import shap
import matplotlib.pyplot as plt

def train_and_evaluate_model(folder, train_path, valid_path):

    # データのロードと前処理
    train_df = pd.read_csv(folder+"/"+train_path)
    valid_df = pd.read_csv(folder+"/"+valid_path)
    SEED = 42 
    # label_map = {3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
    # label_map = {"low": 0, "medium": 1, "high": 2}
    # train_df["label"] = train_df["label"]
    # valid_df["label"] = valid_df["label"]

    # resampled_dfs = []
    # target_count = 300
    # for label, group in train_df.groupby("label"):
    #     if len(group) > target_count:
    #         resampled = resample(group, replace=False, n_samples=target_count, random_state=SEED)
    #     else:
    #         resampled = resample(group, replace=True, n_samples=target_count, random_state=SEED)
    #     resampled_dfs.append(resampled)
    # train_df = pd.concat(resampled_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_valid = valid_df.drop(columns=['label'])
    y_valid = valid_df['label']

    all_labels = train_df['label'].tolist() + valid_df['label'].tolist()
    unique_labels = np.unique(all_labels)
    
    # ラベルエンコーディングの設定
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    
    # モデルのトレーニングとSHAP解析
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    # 予測と評価
    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(model.predict(X_valid)),
        **valid_df.drop(columns=['label']).to_dict('series')
    })
    predictions_df.to_csv('ml_class/results/result_num_rf.csv', index=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
                                  columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
                                  index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])))
    conf_matrix_df.to_csv('ml_class/results/confusion_matrix_num_rf.csv')
    
    # メトリクスの計算と表示
    print("Accuracy:", accuracy_score(predictions_df['label'], predictions_df['predicted_label']))
    print("Precision:", precision_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))
    print("Recall:", recall_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))

    
    # SHAP解析とプロット
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_valid)
    
    # SHAP値の形状を確認
    print(f"SHAP values shape: {shap_values.shape}")

    feature_names=X_train.columns
    
    # 3クラス以上の場合のSHAPプロット
     # # 2クラスのとき
    if len(unique_labels)==2:
        shap.summary_plot(shap_values[:, :, 1], X_valid, feature_names=feature_names)
    # # 1クラスのとき
    # elif len(unique_labels)==1:
    #     shap.summary_plot(shap_values, X_valid, feature_names=feature_names)
    else:
        # 3クラス以上のとき
        for i in range(len(unique_labels)):
            print("Class:" + str(i))  # 修正: iを文字列に変換して出力
            shap.summary_plot(shap_values[:, :, i], X_valid, feature_names=feature_names)
        
    plt.show()

    # SHAP 値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_valid)
    
    # SHAP値をnumpy配列に変換
    shap_values = np.array(shap_values)
    print(shap_values.shape)  # (94, 13, 2)
    
    # 2クラスの場合は正例クラス（クラス1）の値を全サンプル分取得
    if len(unique_labels) == 2:
        shap_to_plot = shap_values[:, :, 1]  # shape: (94, 13)
    else:
        shap_to_plot = shap_values[:, :, 2]  # shape: (94, 13)
    
    # DataFrameに変換
    shap_df = pd.DataFrame(shap_to_plot, columns=X_valid.columns, index=X_valid.index)
    print(shap_df.shape)  # (94, 13)
    
    shap_df.to_csv("ml_class/results/shap_value.csv")


    # 各特徴量の SHAP 値の平均を計算
    shap_means = shap_df.mean()
    
    # 平均が 0 ではない特徴量をフィルタリング
    positive_shap_features = shap_means[shap_means != 0].abs().sort_values(ascending=False).index


    # # ヒストグラムの描画
    # plt.figure(figsize=(12, 20))
    
    # for i, feature in enumerate(positive_shap_features):
    #     plt.subplot(len(positive_shap_features) // 3 + 1, 3, i + 1)  # 3列でレイアウト
        
    #     # データポイントの数をカウント
    #     num_positive = (shap_df[feature] > 0).sum()
    #     num_negative = (shap_df[feature] < 0).sum()
        
    #     # ヒストグラムの描画
    #     plt.hist(shap_df[feature], bins=20, alpha=0.7, color="blue", edgecolor="black")
        
    #     # タイトルと軸ラベル
    #     plt.title(feature)
    #     plt.xlabel("SHAP Value")
    #     plt.ylabel("Frequency")
    
    #     # 凡例を追加
    #     plt.legend([f"> 0: {num_positive}\n< 0: {num_negative}"], loc="upper right", fontsize=10)
    
    # plt.tight_layout()
    # plt.show()



    top_20_features = shap_df.abs().mean().nlargest(30).index
    
    plt.figure(figsize=(15, 30))
    for i, feature in enumerate(top_20_features):
        plt.subplot(11, 3, i + 1)
        sc = plt.scatter(X_valid[feature], shap_df[feature], c=X_valid[feature], cmap="coolwarm", alpha=0.6, edgecolors="k")
        plt.colorbar(sc, label="Feature Value")
        plt.title(feature)
        plt.xlabel("Feature Value")
        plt.ylabel("SHAP Value")
    plt.tight_layout()
    plt.show()


    #         # TreeExplainer でモデルとデータ
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_valid)
    
    # # 交互作用値を取得
    # interaction_values = explainer.shap_interaction_values(X_valid)
    
    # # 形状を確認
    # if isinstance(interaction_values, list):
    #     num_classes = len(interaction_values)
    # else:
    #     # 4次元: (num_samples, num_features, num_features, num_classes)
    #     num_classes = interaction_values.shape[3]
    
    # num_samples = X_valid.shape[0]
    # feature_names = X_valid.columns
    
    # # クラスごとに CSV 出力
    # for class_idx in range(num_classes):
    #     if isinstance(interaction_values, list):
    #         inter_vals_class = np.array(interaction_values[class_idx])  # shape (num_samples, num_features, num_features)
    #     else:
    #         inter_vals_class = interaction_values[:, :, :, class_idx]
    
    #     # reshape -> (num_samples, num_features*num_features)
    #     df_inter = pd.DataFrame(
    #         inter_vals_class.reshape(num_samples, -1),
    #         columns=[f"{f1}__{f2}" for f1 in feature_names for f2 in feature_names],
    #         index=X_valid.index
    #     )
    
    #     # CSV出力
    #     df_inter.to_csv(f"shap_interaction_class{class_idx}.csv")
    #     print(f"Saved shap interaction CSV for class {class_idx}")
