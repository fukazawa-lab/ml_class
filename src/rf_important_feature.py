
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

    # ===== 重要特徴量（Feature Importance） =====
    importances = model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # 上位20件をプロット
    top_n = 20
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.show()
