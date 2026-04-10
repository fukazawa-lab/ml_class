import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import os


def objective(trial, X_train, y_train, class_weights):
    # KNNのハイパーパラメータ候補
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    # class_weightはKNNにはないので使いません

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()


def train_and_evaluate_model(folder, train_path, valid_path,
                             use_smote=False, use_class_weight=False,  # use_class_weightはKNNでは無効
                             do_tuning=False,
                             output_filename="result_num_knn.csv"):

    # データのロード
    train_df = pd.read_csv(os.path.join(folder, train_path))
    valid_df = pd.read_csv(os.path.join(folder, valid_path))

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_valid = valid_df.drop(columns=['label'])
    y_valid = valid_df['label']

    # ラベルエンコーディング
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)

    # SMOTE適用（オプション）
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train_encoded = smote.fit_resample(X_train, y_train_encoded)
        print(f"SMOTE applied: {np.bincount(y_train_encoded)}")

    # KNNはclass_weightパラメータがないので無視
    class_weights = None


    best_params = {'n_neighbors': 1, 'weights': 'distance'}

    model = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        n_jobs=-1
    )

    model.fit(X_train, y_train_encoded)

    # 予測
    y_pred = model.predict(X_valid)
    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(y_pred),
        **X_valid.reset_index(drop=True).to_dict('series')
    })

    # 出力ディレクトリ
    output_dir = 'ml_class/results'
    os.makedirs(output_dir, exist_ok=True)

    # 予測結果の保存
    output_path = os.path.join(output_dir, output_filename)
    predictions_df.to_csv(output_path, index=False)

    # 混同行列の保存
    confusion_filename = "confusion_matrix_knn.csv"
    confusion_path = os.path.join(output_dir, confusion_filename)
    conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])
    conf_matrix_df = pd.DataFrame(conf_matrix,
                                  columns=sorted(label_encoder.classes_),
                                  index=sorted(label_encoder.classes_))
    conf_matrix_df.to_csv(confusion_path)

    # メトリクスの表示
    # print("Accuracy:", accuracy_score(predictions_df['label'], predictions_df['predicted_label']))
    # print("Precision:", precision_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))
    # print("Recall:", recall_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))

    return predictions_df
