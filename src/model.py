from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_MODELS_DIR = Path(__file__).parent / 'models'

# Display name → bundle file
MODEL_CATALOG = {
    "Core model":                                    _MODELS_DIR / 'no_REVEL_no_phyloP100way_DT_bundle.joblib',
    "Core + missense predictor (REVEL)":             _MODELS_DIR / 'no_phyloP100way_DT_bundle.joblib',
    "Core + missense predictor + conservation (phyloP)": _MODELS_DIR / 'complete_DT_bundle.joblib',
}

RF_CATALOG = {
    name: Path(str(path).replace('_DT_', '_RF_'))
    for name, path in MODEL_CATALOG.items()
}

# Columns with fixed domain defaults (not derived from imputer statistics)
HARDCODED_FILLS = {
    'log10_AF_popmax': -6.0,
    'spliceai': 0.0,
}


def _imputer_and_numeric_cols(bundle):
    """Normalise key names — the two bundles use different conventions."""
    if 'imputer' in bundle:
        return bundle['imputer'], bundle['numeric_columns']
    return bundle['feature_imputer'], bundle['feature_numeric_columns']


def load_bundle(model_name=None):
    """Load the DT bundle by display name (from MODEL_CATALOG) or by file path."""
    if model_name is None:
        path = next(iter(MODEL_CATALOG.values()))
    elif model_name in MODEL_CATALOG:
        path = MODEL_CATALOG[model_name]
    else:
        path = Path(model_name)
    return joblib.load(path)


def load_rf_bundle(model_name):
    """Load the RF bundle corresponding to the given model name."""
    return joblib.load(RF_CATALOG[model_name])


def get_imputer_defaults(bundle):
    """Return {col: fill_value} for feature columns derived from imputer.statistics_.
    Columns in HARDCODED_FILLS are excluded — their defaults are fixed constants."""
    imputer, numeric_cols = _imputer_and_numeric_cols(bundle)
    feature_cols = set(bundle.get('feature_columns', []))
    if imputer is not None and hasattr(imputer, 'statistics_'):
        return {
            col: float(val)
            for col, val in zip(numeric_cols, imputer.statistics_)
            if col in feature_cols and col not in HARDCODED_FILLS
        }
    return {}


def _preprocess(bundle, feature_values: dict) -> pd.DataFrame:
    """Apply full preprocessing pipeline to a single feature dict.

    Steps:
      1. Derive has_aa_pos from aa_pos if not supplied (mirrors dataset behaviour).
      2. Apply hardcoded fills for log10_AF_popmax and spliceai.
      3. Apply imputer (median) for any remaining NaN in numeric feature columns.
    """
    feat_cols = bundle['feature_columns']
    imp, num_cols = _imputer_and_numeric_cols(bundle)
    drop_cols = set(bundle.get('feature_drop_cols', []))

    fv = dict(feature_values)

    # Step 1: derive has_aa_pos if absent
    if 'has_aa_pos' not in fv or fv['has_aa_pos'] is None:
        fv['has_aa_pos'] = 0.0 if pd.isna(fv.get('aa_pos', np.nan)) else 1.0

    X = pd.DataFrame([fv])
    for col in feat_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feat_cols]

    # Step 2: hardcoded fills
    for col, val in HARDCODED_FILLS.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)

    # Step 3: imputer fills for remaining NaN values
    for i, col in enumerate(num_cols):
        if col in drop_cols or col not in X.columns:
            continue
        X[col] = X[col].fillna(float(imp.statistics_[i]))

    return X


def predict_dataset(bundle, df: pd.DataFrame) -> np.ndarray:
    """Return predicted classes for every row in df (vectorised, no decision path)."""
    feat_cols = bundle['feature_columns']
    imp, num_cols = _imputer_and_numeric_cols(bundle)
    drop_cols = set(bundle.get('feature_drop_cols', []))

    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        if col == 'has_aa_pos':
            X[col] = (~df['aa_pos'].isna()).astype(float)
        elif col in df.columns:
            X[col] = df[col].values
        else:
            X[col] = np.nan

    for col, val in HARDCODED_FILLS.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)

    for i, col in enumerate(num_cols):
        if col in drop_cols or col not in X.columns:
            continue
        X[col] = X[col].fillna(float(imp.statistics_[i]))

    return bundle['model'].predict(X).astype(int)


def predict(bundle, feature_values: dict):
    clf = bundle['model']
    feat_cols = bundle['feature_columns']

    X = _preprocess(bundle, feature_values)

    pred_class = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    pred_proba = float(proba[pred_class])

    # Decision path only available for single-tree models (not RF ensembles)
    path = _extract_decision_path(clf, X, feat_cols) if hasattr(clf, 'tree_') else []
    return pred_class, pred_proba, path


def predict_from_row(bundle, row: pd.Series):
    """Convenience wrapper for a raw dataset row (e.g. from concepts_withVariantKey.csv)."""
    feat_cols = bundle['feature_columns']
    fv = {
        c: (row[c] if c in row.index else np.nan)
        for c in feat_cols
        if c != 'has_aa_pos'
    }
    # has_aa_pos is not in the dataset — always derive it
    fv['has_aa_pos'] = None
    return predict(bundle, fv)


def _extract_decision_path(clf, X, feature_cols):
    tree = clf.tree_
    X_arr = X.values
    node_indicator = clf.decision_path(X)
    leaf_id = clf.apply(X)[0]

    node_ids = node_indicator.indices[
        node_indicator.indptr[0]: node_indicator.indptr[1]
    ]

    steps = []
    for node_id in node_ids:
        if node_id == leaf_id:
            node_values = tree.value[node_id][0]
            total = node_values.sum()
            steps.append({
                'type': 'leaf',
                'samples': int(tree.n_node_samples[node_id]),
                'class': int(np.argmax(node_values)),
                'benign_frac': float(node_values[0] / total) if total > 0 else 0.0,
                'pathogenic_frac': float(node_values[1] / total) if total > 0 else 0.0,
            })
        else:
            feat_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feat_val = float(X_arr[0, feat_idx])
            goes_left = feat_val <= threshold
            steps.append({
                'type': 'split',
                'feature': feature_cols[feat_idx],
                'threshold': round(float(threshold), 4),
                'value': round(feat_val, 4),
                'direction': 'left' if goes_left else 'right',
                'condition': f'<= {round(float(threshold), 4)}',
                'met': goes_left,
            })

    return steps


def get_feature_importances(bundle):
    clf = bundle['model']
    feature_cols = bundle['feature_columns']
    pairs = list(zip(feature_cols, clf.feature_importances_))
    return sorted(pairs, key=lambda x: x[1], reverse=True)
