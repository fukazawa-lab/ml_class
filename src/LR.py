import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import os
import optuna

def objective(trial, X_train, y_train, class_weights):
    # LogisticRegressionのハイパーパラメータ
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)

    model = LogisticRegression(
        C=C,
        class_weight=class_weights,
        solver='liblinear',   # 安定＆小規模データ向き
        max_iter=1000,
        random_state=42
    )

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()


def simple_logistic_with_tuning(X_train, y_train_encoded, class_weights, n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train_encoded, class_weights), n_trials=n_trials)

    print("Best parameters found:", study.best_params)

    best_model = LogisticRegression(
        C=study.best_params['C'],
        class_weight=class_weights,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    best_model.fit(X_train, y_train_encoded)
    return best_model


def train_and_evaluate_model(folder, train_path, valid_path,
                             use_smote=False, use_class_weight=False,
                             do_tuning=False,
                             output_filename="result_num_logistic.csv"):

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

    # クラス重み
    class_weights = "balanced" if use_class_weight else None

    # モデル
    if do_tuning:
        model = simple_logistic_with_tuning(X_train, y_train_encoded, class_weights)
    else:
        best_params = {
            'C': 1.0
        }

        model = LogisticRegression(
            C=best_params['C'],
            class_weight=class_weights,
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train_encoded)

    # 予測
    y_pred = model.predict(X_valid)

    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(y_pred),
        **X_valid.reset_index(drop=True).to_dict('series')
    })

    # 保存
    output_dir = 'ml_class/results/classification'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    predictions_df.to_csv(output_path, index=False)

    # 混同行列
    confusion_filename = "confusion_matrix_lr.csv"
    confusion_path = os.path.join(output_dir, confusion_filename)

    conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])
    conf_matrix_df = pd.DataFrame(conf_matrix,
                                  columns=sorted(label_encoder.classes_),
                                  index=sorted(label_encoder.classes_))
    conf_matrix_df.to_csv(confusion_path)

    # 評価
    # print("Accuracy:", accuracy_score(predictions_df['label'], predictions_df['predicted_label']))
    # print("Precision:", precision_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))
    # print("Recall:", recall_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))

    return predictions_df