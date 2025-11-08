import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib




def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_wavelengths(arg: str) -> List[int]:
    if arg == 'all':
        return list(range(700, 901, 20))
    return [int(x) for x in arg.split(',') if x.strip()]


def list_patients(features_dir: str) -> List[str]:
    return [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]


def extract_pressure(name: str) -> int:
    m = re.search(r'pressure(\d+)', name)
    return int(m.group(1)) if m else None


def aggregate_patient_wavelength(features_dir: str, patient: str, wl: int, agg: str) -> pd.Series:
    fpath = os.path.join(features_dir, patient, f"{wl}nm.csv")
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return None
    if agg == 'mean':
        s = num.mean()
    else:
        s = num.median()
    return s


def build_unified_row(features_dir: str, patient: str, wavelengths: List[int], agg: str, normal_low: int, normal_high: int) -> Dict:
    pressure = extract_pressure(patient)
    if pressure is None:
        return None
    row = {
        'patient': patient,
        'pressure': pressure,
        'is_normal_gt': int(normal_low <= pressure <= normal_high)
    }
    for wl in wavelengths:
        s = aggregate_patient_wavelength(features_dir, patient, wl, agg)
        if s is None:
            return None  # require all wavelengths
        for k, v in s.items():
            row[f"{wl}nm_{k}"] = v
    return row


def normalize_roi_key(name: str) -> str:
    bn = os.path.basename(str(name))
    # strip extension
    if bn.endswith('.mat'):
        bn = bn[:-4]
    # drop suffix after '__'
    base = bn.split('__', 1)[0]
    m = re.match(r'(\d+)', base)
    if m:
        digits = m.group(1)
        pref3 = digits[:3] if len(digits) >= 3 else digits
        return pref3.zfill(3)
    return base


def wl_remainder(wl: int) -> int:
    try:
        return ((wl - 780) // 20) % 11
    except Exception:
        print(f"[WARN] unexpected wavelength value in wl_remainder: {wl}")
        return 0


def compute_roi_group(name: str, wl: int):
    bn = os.path.basename(str(name))
    if bn.endswith('.mat'):
        bn = bn[:-4]
    base = bn.split('__', 1)[0]
    m = re.match(r'(\d+)', base)
    if not m:
        return None
    try:
        base_id = int(m.group(1))
    except Exception:
        return None
    r = wl_remainder(wl)
    return (base_id - r) // 11


def build_unified_rows_for_patient(features_dir: str, patient: str, wavelengths: List[int], normal_low: int, normal_high: int) -> List[Dict]:
    dfs = []
    for wl in wavelengths:
        fpath = os.path.join(features_dir, patient, f"{wl}nm.csv")
        if not os.path.isfile(fpath):
            return []
        df = pd.read_csv(fpath)
        if 'file' not in df.columns:
            return []
        # compute numeric roi_group stable across wavelengths
        df['roi_group'] = df['file'].apply(lambda nm: compute_roi_group(nm, wl))
        df = df[df['roi_group'].notna()]
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return []
        num_cols = [c for c in num.columns if c != 'roi_group']
        df_wl = df[['roi_group'] + num_cols].copy()
        df_wl.rename(columns={c: f"{wl}nm_{c}" for c in num_cols}, inplace=True)
        dfs.append(df_wl)
    df_join = dfs[0]
    for k in range(1, len(dfs)):
        df_join = pd.merge(df_join, dfs[k], on='roi_group', how='inner')
    if df_join.empty:
        return []
    pressure = extract_pressure(patient)
    is_normal_gt = int(normal_low <= pressure <= normal_high)
    rows = []
    for _, r in df_join.iterrows():
        new_row = {'patient': patient, 'pressure': pressure, 'is_normal_gt': is_normal_gt}
        new_row['roi_group'] = int(r['roi_group'])
        new_row['file'] = str(int(r['roi_group']))
        for c in df_join.columns:
            if c == 'roi_group':
                continue
            new_row[c] = r[c]
        rows.append(new_row)
    return rows



def load_unified_df(features_dir: str, patients: List[str], wavelengths: List[int], agg: str, normal_low: int, normal_high: int, require_all: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    for p in patients:
        rs = build_unified_rows_for_patient(features_dir, p, wavelengths, normal_low, normal_high)
        if not rs:
            print(f"[WARN] build_unified_rows_for_patient returned empty for patient: {p}; skipping")
            continue
        rows.extend(rs)
    if not rows:
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ['patient', 'pressure', 'is_normal_gt', 'file', 'roi_key', 'roi_group']]
    df = df.dropna(subset=feature_cols)
    return df, feature_cols


def plot_pca(X: np.ndarray, y: np.ndarray, pred_labels: List[str], out_path: str, title: str = None):
    try:
        pca = PCA(n_components=2, random_state=42)
        XY = pca.fit_transform(X)
        plt.figure(figsize=(6, 5), dpi=120)
        colors = np.array(['tab:red', 'tab:blue'])  # 0=abnormal, 1=normal
        markers = {'normal': 'o', 'abnormal': '^'}
        for i in range(XY.shape[0]):
            c = colors[int(y[i])]
            m = markers[pred_labels[i]]
            plt.scatter(XY[i, 0], XY[i, 1], c=c, marker=m, s=45, edgecolors='k', linewidths=0.3)
        ax = plt.gca()
        gt_handles = [
            Line2D([0], [0], marker='o', color='w', label='GT: normal',
                   markerfacecolor='tab:blue', markeredgecolor='k', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='GT: abnormal',
                   markerfacecolor='tab:red', markeredgecolor='k', markersize=8),
        ]
        pred_handles = [
            Line2D([0], [0], marker='o', color='k', label='Pred: normal',
                   markerfacecolor='none', markersize=8),
            Line2D([0], [0], marker='^', color='k', label='Pred: abnormal',
                   markerfacecolor='none', markersize=8),
        ]
        leg1 = ax.legend(handles=gt_handles, loc='upper right', title='Ground Truth')
        ax.add_artist(leg1)
        ax.legend(handles=pred_handles, loc='lower left', title='Prediction')
        plt.title(title or 'Unified PCA: color=GT, marker=Pred')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    #except Exception:
    #    pass
    except Exception as e:
        import traceback
        print(f"[ERROR] plot_pca failed: {e}")
        traceback.print_exc()



def train_unified(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: List[str], wavelengths: List[int], out_dir: str, models_dir: str, agg: str, normal_low: int, normal_high: int, cv_folds: int, rs: int):
    ensure_dir(out_dir)
    ensure_dir(models_dir)

    X_train = df_train[feature_cols].values
    y_train = df_train['is_normal_gt'].values.astype(int)
    X_test = df_test[feature_cols].values if df_test is not None and not df_test.empty else None
    y_test = df_test['is_normal_gt'].values.astype(int) if X_test is not None else None

    supervised = {
        'lr': LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear', random_state=rs),
        'svm': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=rs),
        #'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=rs),
        #'gb': GradientBoostingClassifier(random_state=rs),
        'rf_xgb': XGBClassifier(
            n_estimators=200,
            tree_method='gpu_hist',   # 关键：使用GPU加速
            predictor='gpu_predictor',
            random_state=rs,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    sup_metrics = {}
    best_name = None
    best_f1 = -1.0
    best_joblib = None
    best_pred_labels_test = None

    # Train supervised models
    for name, clf in supervised.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
        acc_mean = float(np.mean(cv_res['test_accuracy']))
        f1_mean = float(np.mean(cv_res['test_f1_macro']))
        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        acc_train_full = float(accuracy_score(y_train, y_train_pred))
        f1_train_full = float(f1_score(y_train, y_train_pred, average='macro'))
        cm_train_full = confusion_matrix(y_train, y_train_pred).tolist()

        test_metrics = None
        pred_labels_test = None
        preds_detail = None
        if X_test is not None:
            y_test_pred = pipe.predict(X_test)
            acc_test = float(accuracy_score(y_test, y_test_pred))
            f1_test = float(f1_score(y_test, y_test_pred, average='macro'))
            cm_test = confusion_matrix(y_test, y_test_pred).tolist()
            try:
                proba = pipe.predict_proba(X_test)
                # get classes from classifier inside pipeline
                try:
                    clf_inside = pipe.named_steps['clf']
                    classes = list(clf_inside.classes_)
                except Exception:
                    # fallback
                    classes = [0, 1]
                if 1 in classes and 0 in classes:
                    idx_normal = classes.index(1)
                    idx_abnormal = classes.index(0)
                elif 1 in classes:
                    idx_normal = classes.index(1)
                    idx_abnormal = 0 if 0 in classes else (1 - idx_normal)
                else:
                    # fallback default indices
                    idx_normal = 1
                    idx_abnormal = 0
                ...
            except Exception:
                proba = None

            pred_labels_test = ['normal' if p == 1 else 'abnormal' for p in y_test_pred]

            test_metrics = {
                'accuracy': acc_test,
                'f1_macro': f1_test,
                'confusion_matrix': cm_test,
                'n_test': int(len(y_test))
            }

        sup_metrics[name] = {
            'cv_accuracy_mean': acc_mean,
            'cv_f1_macro_mean': f1_mean,
            'train_accuracy_full': acc_train_full,
            'train_f1_macro_full': f1_train_full,
            'train_confusion_matrix_full': cm_train_full,
            'test': test_metrics
        }

        model_path = os.path.join(models_dir, f"unified_{name}.joblib")
        joblib.dump({
            'pipeline': pipe,
            'feature_cols': feature_cols,
            'wavelengths': wavelengths,
            'aggregate': agg,
            'normal_range': [normal_low, normal_high],
            'model_name': name,
            'cv_metrics': {'accuracy_mean': acc_mean, 'f1_macro_mean': f1_mean},
            'train_metrics': {'accuracy_full': acc_train_full, 'f1_macro_full': f1_train_full, 'confusion_matrix_full': cm_train_full},
            'test_metrics': test_metrics,
        }, model_path)

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_name = name
            best_joblib = model_path
            best_pred_labels_test = pred_labels_test

    best_alias = os.path.join(models_dir, f"unified_best.joblib")
    if best_joblib:
        obj = joblib.load(best_joblib)
        joblib.dump(obj, best_alias)

    # Aggregates CSV with set column
    df_out = pd.concat([
        df_train.assign(set='train'),
        df_test.assign(set='test') if df_test is not None and not df_test.empty else pd.DataFrame(columns=df_train.columns.tolist() + ['set'])
    ], ignore_index=True)
    agg_path = os.path.join(out_dir, f"unified_aggregates.csv")
    df_out.to_csv(agg_path, index=False)

    try:
        X_all = df_out[feature_cols].values
        y_all = df_out['is_normal_gt'].values.astype(int)
        if best_alias:
            best_obj = joblib.load(best_alias)
            pipe_all = best_obj['pipeline']
            y_all_pred = pipe_all.predict(X_all)
            labels_plot = ['normal' if p == 1 else 'abnormal' for p in y_all_pred]
        else:
            labels_plot = ['normal' if yi == 1 else 'abnormal' for yi in y_all]
        plot_path = os.path.join(out_dir, f"plot_unified.png")
        plot_pca(X_all, y_all, labels_plot, plot_path, title=f"Unified PCA (Train+Test set): color=GT, marker=Pred")
        print(f"[DONE] PCA (train): {plot_path} (dataset=train+test, n_train={df_train.shape[0]}, n_test={df_test.shape[0] if df_test is not None else 0})")
        # Train-only PCA plot
        plot_train_path = os.path.join(out_dir, f"plot_unified_train.png")
        if best_alias:
            best_obj = joblib.load(best_alias)
            pipe_train = best_obj['pipeline']
            y_train_pred = pipe_train.predict(X_train)
            labels_train_plot = ['normal' if p == 1 else 'abnormal' for p in y_train_pred]
        else:
            labels_train_plot = ['normal' if yi == 1 else 'abnormal' for yi in y_train]
        plot_pca(X_train, y_train, labels_train_plot, plot_train_path, title=f"Unified PCA (Train set): color=GT, marker=Pred")
        print(f"[DONE] PCA (train-only): {plot_train_path} (dataset=train, n_train={df_train.shape[0]})")
    except Exception:
        pass

    preds_csv_path = None
    if best_pred_labels_test is not None:
        best_obj = joblib.load(best_alias)
        pipe = best_obj['pipeline']
        try:
            proba = pipe.predict_proba(X_test) if X_test is not None else None
        except Exception:
            proba = None
        y_test_pred = pipe.predict(X_test) if X_test is not None else None
        rows = []
        if X_test is not None:
            #classes = list(pipe.classes_) if hasattr(pipe, 'classes_') else [0, 1]
            try:
                clf_inside = pipe.named_steps['clf']
                classes = list(clf_inside.classes_)
            except Exception:
                classes = [0, 1]

            idx_normal = classes.index(1) if 1 in classes else 1
            idx_abnormal = classes.index(0) if 0 in classes else 0
            for i in range(len(y_test_pred)):
                rows.append({
                    'patient': df_test.iloc[i]['patient'],
                    'pressure': int(df_test.iloc[i]['pressure']),
                    'gt': 'normal' if y_test[i] == 1 else 'abnormal',
                    'pred': 'normal' if y_test_pred[i] == 1 else 'abnormal',
                    'prob_normal': float(proba[i, idx_normal]) if proba is not None else None,
                    'prob_abnormal': float(proba[i, idx_abnormal]) if proba is not None else None,
                    'correct': int(y_test_pred[i] == y_test[i])
                })
        preds_df = pd.DataFrame(rows)
        preds_csv_path = os.path.join(out_dir, f"unified_predictions_test.csv")
        preds_df.to_csv(preds_csv_path, index=False)

    metrics = {
        'wavelengths': wavelengths,
        'n_train': int(df_train.shape[0]),
        'n_test': int(df_test.shape[0]) if df_test is not None else 0,
        'aggregate': agg,
        'normal_range': [normal_low, normal_high],
        'supervised_models': sup_metrics,
        'best_model_name': best_name,
        'best_model_joblib': best_alias if best_joblib else None,
        'output_files': {
            'aggregates': agg_path,
            'plot': os.path.join(out_dir, f"plot_unified.png"),
            'plot_train': os.path.join(out_dir, f"plot_unified_train.png"),
            'predictions_test': preds_csv_path
        }
    }
    with open(os.path.join(out_dir, f"unified_metrics.json"), 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description='Unified multi-wavelength training with explicit train/test split')
    ap.add_argument('--config', default='')
    ap.add_argument('--features-dir', default='')
    ap.add_argument('--output-dir', default='')
    ap.add_argument('--models-dir', default='')
    ap.add_argument('--wavelengths', default='all')
    ap.add_argument('--aggregate', default='median', choices=['median', 'mean'])
    ap.add_argument('--normal-range', default='140,190')
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--split-file', default='')
    ap.add_argument('--train-patients', default='')
    ap.add_argument('--test-patients', default='')
    ap.add_argument('--test-size', type=float, default=0.3)
    ap.add_argument('--roi-range', default='')
    args = ap.parse_args()

    cfg = {}
    if args.config and os.path.isfile(args.config):
        try:
            with open(args.config, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    if not cfg:
        try:
            default_cfg_path = os.path.join(Path(__file__).resolve().parent, 'unified_config.json')
            if os.path.isfile(default_cfg_path):
                with open(default_cfg_path, 'r') as f:
                    cfg = json.load(f)
        except Exception:
            pass

    features_dir = args.features_dir or cfg.get('features_dir', '')
    output_dir = args.output_dir or cfg.get('output_dir', '')
    models_dir = args.models_dir or cfg.get('models_dir', '')
    wavelengths_arg = args.wavelengths if args.wavelengths else cfg.get('wavelengths', 'all')
    aggregate = args.aggregate if args.aggregate else cfg.get('aggregate', 'median')
    normal_range_arg = args.normal_range if args.normal_range else cfg.get('normal_range', '140,190')
    cv_folds = args.cv_folds if args.cv_folds else int(cfg.get('cv_folds', 5))
    random_state = args.random_state if args.random_state else int(cfg.get('random_state', 42))
    split_file = args.split_file or cfg.get('split_file', '')
    train_patients_arg = args.train_patients or cfg.get('train_patients', [])
    test_patients_arg = args.test_patients or cfg.get('test_patients', [])
    test_size = args.test_size if args.test_size else float(cfg.get('test_size', 0.3))
    roi_range = args.roi_range or cfg.get('roi_range', '')

    if isinstance(wavelengths_arg, str):
        wavelengths = parse_wavelengths(wavelengths_arg)
    else:
        wavelengths = [int(x) for x in wavelengths_arg]
    if isinstance(normal_range_arg, str):
        normal_low, normal_high = [int(x) for x in normal_range_arg.split(',')]
    elif isinstance(normal_range_arg, (list, tuple)) and len(normal_range_arg) >= 2:
        normal_low, normal_high = int(normal_range_arg[0]), int(normal_range_arg[1])
    else:
        normal_low, normal_high = 140, 190

    if not features_dir:
        print('[ERROR] --features-dir is required (or set in config).')
        return
    if not output_dir:
        print('[ERROR] --output-dir is required (or set in config).')
        return
    if not models_dir:
        print('[ERROR] --models-dir is required (or set in config).')
        return

    ensure_dir(output_dir)
    ensure_dir(models_dir)

    train_patients = []
    test_patients = []
    if split_file and os.path.isfile(split_file):
        with open(split_file, 'r') as f:
            sp = json.load(f)
        train_patients = sp.get('train', [])
        test_patients = sp.get('test', [])
    else:
        if train_patients_arg:
            if isinstance(train_patients_arg, str):
                train_patients = [s for s in train_patients_arg.split(',') if s.strip()]
            else:
                train_patients = list(train_patients_arg)
        if test_patients_arg:
            if isinstance(test_patients_arg, str):
                test_patients = [s for s in test_patients_arg.split(',') if s.strip()]
            else:
                test_patients = list(test_patients_arg)

    if not train_patients or not test_patients:
        all_patients = []
        for p in list_patients(features_dir):
            # require all wavelengths to be present
            ok = True
            for wl in wavelengths:
                if not os.path.isfile(os.path.join(features_dir, p, f"{wl}nm.csv")):
                    ok = False
                    break
            if ok:
                all_patients.append(p)
        labels = []
        for p in all_patients:
            pr = extract_pressure(p)
            labels.append(int(normal_low <= pr <= normal_high))
        train_patients, test_patients = train_test_split(all_patients, test_size=test_size, stratify=labels, random_state=random_state)

    df_train, feature_cols = load_unified_df(features_dir, train_patients, wavelengths, aggregate, normal_low, normal_high, require_all=True)
    df_test, _ = load_unified_df(features_dir, test_patients, wavelengths, aggregate, normal_low, normal_high, require_all=True)
    print(f"[INFO] df_train pre-filter shape: {df_train.shape}")
    if 'roi_group' in df_train.columns:
        try:
            vc = df_train['roi_group'].value_counts().sort_index()
            print(f"[INFO] roi_group counts (train) head: {dict(list(vc.items())[:10])}")
        except Exception:
            pass
    if roi_range:
        def parse_digits(tok: str):
            m = re.search(r'(\d+)', tok or '')
            return int(m.group(1)) if m else None
        try:
            start_tok, end_tok = [t.strip() for t in roi_range.split('-', 1)]
        except Exception:
            start_tok, end_tok = '', ''
        s0 = parse_digits(start_tok)
        s1 = parse_digits(end_tok)
        if (s0 is not None) and (s1 is not None) and 'roi_group' in df_train.columns:
            df_train = df_train[df_train['roi_group'].apply(lambda v: s0 <= int(v) <= s1)]
            df_test = df_test[df_test['roi_group'].apply(lambda v: s0 <= int(v) <= s1)] if df_test is not None and not df_test.empty else df_test
        else:
            def in_roi_range_fn(x: str) -> bool:
                bn = os.path.basename(str(x))
                m = re.search(r'(\d+)', bn)
                if m and s0 is not None and s1 is not None:
                    idx = int(m.group(1))
                    return s0 <= idx <= s1
                return (bn >= start_tok) and (bn <= end_tok) if start_tok and end_tok else True
            if 'file' in df_train.columns:
                df_train = df_train[df_train['file'].apply(in_roi_range_fn)]
            if df_test is not None and not df_test.empty and 'file' in df_test.columns:
                df_test = df_test[df_test['file'].apply(in_roi_range_fn)]
    print(f"[INFO] df_train post-filter shape: {df_train.shape}")
    if df_test is not None:
        print(f"[INFO] df_test post-filter shape: {df_test.shape}")

    # --- Defensive checks and then call train_unified ---
    if df_train.empty:
        print("[ERROR] df_train is empty after loading/filtering. Aborting.")
        return
    if not feature_cols:
        print("[ERROR] No feature columns detected. Aborting.")
        return

    # If df_test is empty, set to None so train_unified knows there's no test set
    df_test_for_train = None if (df_test is None or df_test.empty) else df_test

    print("[INFO] Starting training...")
    try:
        train_unified(df_train, df_test_for_train, feature_cols, wavelengths, output_dir, models_dir, aggregate, normal_low, normal_high, cv_folds, random_state)
        print("[INFO] Training finished.")
    except Exception as e:
        import traceback
        print("[ERROR] Exception during train_unified:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()