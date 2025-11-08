import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass




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
            return None  
        for k, v in s.items():
            row[f"{wl}nm_{k}"] = v
    return row


def read_patients_arg(features_dir: str, patients_arg: str, patients_file: str, wavelengths: List[int]) -> List[str]:
    patients = []
    if patients_file and os.path.isfile(patients_file):
        try:
            with open(patients_file, 'r') as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                if 'patients' in obj and isinstance(obj['patients'], list):
                    patients = obj['patients']
                elif 'test' in obj and isinstance(obj['test'], list):
                    patients = obj['test']
        except Exception:
            pass
    if not patients and patients_arg:
        patients = [s for s in patients_arg.split(',') if s.strip()]
    if not patients:
        for p in list_patients(features_dir):
            ok = True
            for wl in wavelengths:
                if not os.path.isfile(os.path.join(features_dir, p, f"{wl}nm.csv")):
                    ok = False
                    break
            if ok:
                patients.append(p)
    return patients


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
    except Exception:
        # best-effort plotting; ignore failures
        pass




def main():
    ap = argparse.ArgumentParser(description='Predict unified multi-wavelength model for a group of patients')
    ap.add_argument('--config', default='')
    ap.add_argument('--features-dir', default='')
    ap.add_argument('--models-dir', default='')
    ap.add_argument('--output-dir', default='')
    ap.add_argument('--wavelengths', default='all')
    ap.add_argument('--aggregate', default='median', choices=['median', 'mean'])
    ap.add_argument('--normal-range', default='140,190')
    ap.add_argument('--patients', default='')
    ap.add_argument('--patients-file', default='')
    ap.add_argument('--model-name', default='best', choices=['best', 'lr', 'svm', 'rf', 'gb'])
    ap.add_argument('--roi-range', default='')
    ap.add_argument('--no-plot', action='store_true', help='Skip PCA concatenation and plotting to reduce memory usage')
    ap.add_argument('--limit-patients', type=int, default=0, help='Only process the first N patients (0 means all)')
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
    models_dir = args.models_dir or cfg.get('models_dir', '')
    output_dir = args.output_dir or cfg.get('output_dir', '')
    wavelengths_arg = args.wavelengths if args.wavelengths else cfg.get('wavelengths', 'all')
    aggregate = args.aggregate if args.aggregate else cfg.get('aggregate', 'median')
    normal_range_arg = args.normal_range if args.normal_range else cfg.get('normal_range', '140,190')
    patients_arg = args.patients or cfg.get('patients', [])
    patients_file = args.patients_file or cfg.get('patients_file', '')
    model_name = args.model_name if args.model_name else cfg.get('model_name', 'best')
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
    if not models_dir:
        print('[ERROR] --models-dir is required (or set in config).')
        return
    if not output_dir:
        print('[ERROR] --output-dir is required (or set in config).')
        return

    ensure_dir(output_dir)
    


    if model_name == 'best':
        model_path = os.path.join(models_dir, 'unified_best.joblib')
    else:
        model_path = os.path.join(models_dir, f'unified_{model_name}.joblib')
    
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    obj = joblib.load(model_path)
    pipe = obj['pipeline']
    feature_cols = obj['feature_cols']
    model_wavelengths = obj.get('wavelengths', wavelengths)
    agg_model = obj.get('aggregate', aggregate)
    


    subset = 'all'
    patients = []
    if patients_file and os.path.isfile(patients_file):
        try:
            with open(patients_file, 'r') as f:
                sp = json.load(f)
            if isinstance(sp, dict):
                if 'train' in sp and isinstance(sp['train'], list) and sp['train']:
                    patients = sp['train']
                    subset = 'train'
                elif 'patients' in sp and isinstance(sp['patients'], list) and sp['patients']:
                    patients = sp['patients']
                    subset = 'custom'
                elif 'test' in sp and isinstance(sp['test'], list) and sp['test']:
                    patients = sp['test']
                    subset = 'test'
        except Exception:
            patients = []
    if not patients:
        split_path = os.path.join(output_dir, 'unified_split.json')
        if os.path.isfile(split_path):
            try:
                with open(split_path, 'r') as f:
                    sp = json.load(f)
                if isinstance(sp, dict):
                    if 'test' in sp and isinstance(sp['test'], list) and sp['test']:
                        patients = sp['test']
                        subset = 'test'
                    elif 'train' in sp and isinstance(sp['train'], list) and sp['train']:
                        patients = sp['train']
                        subset = 'train'
            except Exception:
                pass
    if not patients:
        patients = read_patients_arg(features_dir, patients_arg, '', model_wavelengths)
        subset = 'all' if patients else subset

    

    summary_rows = []
    df_for_pca = []
    preds_labels_for_plot = []
    processed = 0
    for patient in patients:
        if args.limit_patients and processed >= int(args.limit_patients):
            break
        
        rows = build_unified_rows_for_patient(features_dir, patient, model_wavelengths, normal_low, normal_high)
        
        if not rows:
            print(f"[SKIP] Missing ROI or wavelengths for patient: {patient}")
            continue
        dfp = pd.DataFrame(rows)
        before_roi_len = len(dfp)
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
            if (s0 is not None) and (s1 is not None) and 'roi_group' in dfp.columns:
                dfp = dfp[dfp['roi_group'].apply(lambda v: s0 <= int(v) <= s1)]
            else:
                def in_roi_range_fn(x: str) -> bool:
                    bn = os.path.basename(str(x))
                    m = re.search(r'(\d+)', bn)
                    if m and s0 is not None and s1 is not None:
                        idx = int(m.group(1))
                        return s0 <= idx <= s1
                    return (bn >= start_tok) and (bn <= end_tok) if start_tok and end_tok else True
                if 'file' in dfp.columns:
                    dfp = dfp[dfp['file'].apply(in_roi_range_fn)]
        
        if dfp.empty:
            print(f"[SKIP] No ROI rows in range for patient: {patient}")
            continue
        X = dfp[feature_cols].values
        y_gt_each = dfp['is_normal_gt'].astype(int).values
        
        y_pred_each = pipe.predict(X)
        try:
            proba = pipe.predict_proba(X)
            classes = list(pipe.classes_) if hasattr(pipe, 'classes_') else [0, 1]
            idx_normal = classes.index(1) if 1 in classes else 1
            idx_abnormal = classes.index(0) if 0 in classes else 0
            p_normal_each = [float(proba[i, idx_normal]) for i in range(proba.shape[0])]
            p_abnormal_each = [float(proba[i, idx_abnormal]) for i in range(proba.shape[0])]
        except Exception:
            p_normal_each = [None] * len(y_pred_each)
            p_abnormal_each = [None] * len(y_pred_each)
        rois_rows = []
        for i in range(len(y_pred_each)):
            pred_label = 'normal' if int(y_pred_each[i]) == 1 else 'abnormal'
            gt_label = 'normal' if int(y_gt_each[i]) == 1 else 'abnormal'
            preds_labels_for_plot.append(pred_label)
            rois_rows.append({
                'patient': patient,
                'file': str(dfp.iloc[i]['file']) if 'file' in dfp.columns else '',
                'pressure': int(dfp.iloc[i]['pressure']),
                'gt': gt_label,
                'pred': pred_label,
                'prob_normal': p_normal_each[i],
                'prob_abnormal': p_abnormal_each[i],
                'correct': int(int(y_pred_each[i]) == int(y_gt_each[i]))
            })
        csv_path = os.path.join(output_dir, f"pred_unified_{patient}.csv")
        pd.DataFrame(rois_rows).to_csv(csv_path, index=False)
        pred_labels = [r['pred'] for r in rois_rows]
        overall_pred = max(set(pred_labels), key=pred_labels.count)
        n_norm = sum(1 for lbl in pred_labels if lbl == 'normal')
        n_abn = len(pred_labels) - n_norm
        
        avg_pn = None
        avg_pa = None
        if all(p is not None for p in p_normal_each):
            avg_pn = float(np.mean([p for p in p_normal_each]))
        if all(p is not None for p in p_abnormal_each):
            avg_pa = float(np.mean([p for p in p_abnormal_each]))
        gt_label_patient = 'normal' if int(dfp.iloc[0]['is_normal_gt']) == 1 else 'abnormal'
        correct_patient = int((1 if overall_pred == 'normal' else 0) == int(dfp.iloc[0]['is_normal_gt']))
        json_path = os.path.join(output_dir, f"pred_unified_{patient}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'patient': patient,
                'pressure': int(dfp.iloc[0]['pressure']),
                'overall_label': overall_pred,
                'prob_normal': avg_pn,
                'prob_abnormal': avg_pa,
                'wavelengths': model_wavelengths,
                'details_csv': csv_path
            }, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Patient={patient}, pred={overall_pred}, prob_normal={avg_pn}, out={json_path}")
        summary_rows.append({
            'patient': patient,
            'pressure': int(dfp.iloc[0]['pressure']),
            'gt': gt_label_patient,
            'pred': overall_pred,
            'prob_normal': avg_pn,
            'prob_abnormal': avg_pa,
            'correct': correct_patient
        })

        if not args.no_plot:
            df_for_pca.append(dfp)
            preds_labels_for_plot.extend([r['pred'] for r in rois_rows])

        # proactive memory cleanup per patient
        try:
            import gc
            del X, y_gt_each, y_pred_each, p_normal_each, p_abnormal_each, rois_rows, dfp
            gc.collect()
        except Exception:
            pass
        processed += 1

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_csv_name = 'unified_predictions_summary.csv'
        if subset == 'train':
            out_csv_name = 'unified_predictions_train.csv'
        elif subset == 'test':
            out_csv_name = 'unified_predictions_test.csv'
        elif subset == 'custom':
            out_csv_name = 'unified_predictions_custom.csv'
        out_csv = os.path.join(output_dir, out_csv_name)
        summary_df.to_csv(out_csv, index=False)
        
        y_true = [1 if r['gt'] == 'normal' else 0 for r in summary_rows]
        y_pred = [1 if r['pred'] == 'normal' else 0 for r in summary_rows]
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average='macro'))
        cm = confusion_matrix(y_true, y_pred).tolist()
        with open(os.path.join(output_dir, 'unified_overall_metrics.json'), 'w') as f:
            json.dump({'accuracy': acc, 'f1_macro': f1m, 'confusion_matrix': cm, 'n_samples': len(y_true)}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Overall: accuracy={acc:.4f}, f1_macro={f1m:.4f}, n_patients={len(y_true)}")

        if not args.no_plot:
            try:
                df_pca = pd.concat(df_for_pca, ignore_index=True)
                X_all = df_pca[feature_cols].values
                y_all = df_pca['is_normal_gt'].astype(int).values
                pred_labels_plot = preds_labels_for_plot

                if subset == 'train':
                    plot_path = os.path.join(output_dir, 'plot_unified_predict_train.png')
                    title_str = 'Unified PCA (Train subset): color=GT, marker=Pred'
                    ds_label = 'train subset'
                elif subset == 'test':
                    plot_path = os.path.join(output_dir, 'plot_unified_predict.png')
                    title_str = 'Unified PCA (Test subset): color=GT, marker=Pred'
                    ds_label = 'test subset'
                else:
                    plot_path = os.path.join(output_dir, 'plot_unified_predict_custom.png')
                    title_str = 'Unified PCA (Custom subset): color=GT, marker=Pred'
                    ds_label = 'custom subset'
                plot_pca(X_all, y_all, pred_labels_plot, plot_path, title=title_str)
                print(f"[DONE] PCA (predict): {plot_path} (dataset={ds_label}, n={len(df_pca)})")
            except Exception:
                pass
        print(f"[DONE] Summary: {out_csv}")
    else:
        print('[WARN] No predictions generated.')



def build_unified_rows_for_patient(features_dir: str, patient: str, wavelengths: List[int], normal_low: int, normal_high: int) -> List[Dict]:
    dfs = []
    
    def wl_remainder(wl: int) -> int:
        return ((wl - 780) // 20) % 11
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
    for wl in wavelengths:
        fpath = os.path.join(features_dir, patient, f"{wl}nm.csv")
        if not os.path.isfile(fpath):
            return []
        df = pd.read_csv(fpath)
        if 'file' not in df.columns:
            return []
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
    else:
        pass
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


#if __name__ == '__main__':
#    main()

if __name__ == "__main__":
    k_list = [80, 100, 120, 150]
    all_results = []

    for k in k_list:
        print(f"\n===== Running with k = {k} =====")
        args.k_best = k

        # 重新选择特征
        selector = SelectKBest(f_classif, k=k)
        X_train_k = selector.fit_transform(X_train, y_train)
        X_test_k = selector.transform(X_test)

        # 训练 SVM
        model = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=args.random_state)
        model.fit(X_train_k, y_train)

        # 验证
        y_pred_test = model.predict(X_test_k)
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average="weighted")

        print(f"[k={k}] Test Accuracy={acc:.4f}, F1={f1:.4f}")
        all_results.append({"k": k, "acc": acc, "f1": f1})

    # 结果汇总
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results/kbest_comparison.csv", index=False)
    print("\n=== Summary ===")
    print(df_results)
    best_k = df_results.loc[df_results["f1"].idxmax(), "k"]
    print(f"\n✅ 最优特征数: k = {best_k}")
    main()