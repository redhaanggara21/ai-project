# utils/kfold_utils.py

from sklearn.model_selection import KFold
import numpy as np

def evaluate_kfold(model, X, y, k=5, predict_fn=None, scoring_fn=None, random_state=42, verbose=False):
    """
    Evaluasi model menggunakan K-Fold Cross Validation.

    Parameters
    ----------
    model : object
        Model dengan metode .fit() dan .predict(), atau gunakan predict_fn untuk custom prediksi.
    X : array-like
        Fitur input (numpy array atau dataframe).
    y : array-like
        Label target.
    k : int, optional
        Jumlah fold untuk KFold (default=5).
    predict_fn : callable, optional
        Fungsi custom untuk menghasilkan prediksi, format: predict_fn(model, X_test).
    scoring_fn : callable, optional
        Fungsi custom untuk menghitung skor, format: scoring_fn(y_true, y_pred).
        Default: akurasi sederhana (np.mean(y_pred == y_true)).
    random_state : int, optional
        Seed untuk pembagian data (default=42).
    verbose : bool, optional
        Jika True, menampilkan hasil tiap fold.

    Returns
    -------
    dict
        {
            'fold_accuracies': list akurasi per fold,
            'avg_accuracy': rata-rata akurasi
        }
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Latih model
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)

        # Prediksi
        y_pred = predict_fn(model, X_test) if predict_fn else model.predict(X_test)

        # Hitung skor
        if scoring_fn:
            score = scoring_fn(y_test, y_pred)
        else:
            score = np.mean(y_pred == y_test)

        scores.append(score)

        if verbose:
            print(f"Fold {fold} Accuracy: {score:.4f}")

    return {
        'fold_accuracies': scores,
        'avg_accuracy': np.mean(scores)
    }
