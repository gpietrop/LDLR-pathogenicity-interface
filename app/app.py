import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.features import (
    BINARY_FEATURES, FEATURE_DEFAULTS, FEATURE_GROUPS, FEATURE_LABELS,
)
from src.model import HARDCODED_FILLS, MODEL_CATALOG, RF_CATALOG, get_imputer_defaults, load_bundle, load_rf_bundle, predict, predict_dataset, predict_from_row

DATA_PATH         = Path(__file__).parent.parent / 'data' / 'concepts_withVariantKey.csv'
INSTRUCTIONS_PATH = Path(__file__).parent.parent / 'INSTRUCTIONS.txt'

st.set_page_config(
    page_title="LDLR Variant Pathogenicity Classifier",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 270px; max-width: 270px; }
    .stCheckbox { margin-bottom: -0.3rem !important; }
    .stCheckbox label { padding-top: 0 !important; padding-bottom: 0 !important; }
    hr { margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; }
    h4 { margin-top: 0.3rem !important; margin-bottom: 0.2rem !important; }
    .block-container { padding-left: 1.5rem !important; padding-right: 1.5rem !important; max-width: 100% !important; }
    [data-testid="stExpander"] summary { font-size: 1.1rem !important; font-weight: 600 !important; background-color: #dbeafe !important; border-radius: 6px !important; padding: 8px 12px !important; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_bundle(model_name: str):
    return load_bundle(model_name)

@st.cache_resource(show_spinner=False)
def get_rf_bundle(model_name: str):
    return load_rf_bundle(model_name)

@st.cache_data(show_spinner=False)
def load_dataset(_mtime: float = 0):
    return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner="Computing predictions…")
def get_error_sets(model_name: str, _dataset_fingerprint: float):
    """Return sets of #Uploaded_variation values where DT, RF, or both are wrong.
    Keyed by unique HGVS name — duplicate variant_keys cannot bleed across rows."""
    df  = load_dataset(_mtime=_dataset_fingerprint)
    dt  = get_bundle(model_name)
    rf  = get_rf_bundle(model_name)
    true_labels = df['VariantClassification'].astype(int).values
    dt_preds    = predict_dataset(dt, df)
    rf_preds    = predict_dataset(rf, df)
    keys = df['#Uploaded_variation'].values
    dt_wrong   = set(keys[(dt_preds != true_labels)])
    rf_wrong   = set(keys[(rf_preds != true_labels)])
    both_wrong = dt_wrong & rf_wrong
    return dt_wrong, rf_wrong, both_wrong

# ── Session state defaults ────────────────────────────────────────────────────

if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

# ── Welcome / model selection ─────────────────────────────────────────────────

st.title("LDLR Variant Pathogenicity Classifier")

if st.session_state['selected_model'] is None:
    st.markdown("### Select a model to get started")
    st.write("")

    chosen = st.radio(
        "Model",
        list(MODEL_CATALOG.keys()),
        label_visibility="collapsed",
    )
    st.write("")
    if st.button("Continue", type="primary"):
        st.session_state['selected_model'] = chosen
        st.rerun()
    st.stop()

# ── Model loaded ──────────────────────────────────────────────────────────────

selected_model = st.session_state['selected_model']
bundle    = get_bundle(selected_model)
rf_bundle = get_rf_bundle(selected_model)
imputer_defaults = get_imputer_defaults(bundle)
feat_cols_set = set(bundle['feature_columns'])
has_revel  = 'REVEL' in feat_cols_set
has_phylop = 'phyloP100way' in feat_cols_set

# ── Sidebar ───────────────────────────────────────────────────────────────────

@st.dialog("User Guide", width="large")
def show_instructions():
    st.markdown(
        f"```\n{INSTRUCTIONS_PATH.read_text()}\n```"
    )

st.sidebar.markdown(f"**Model:** {selected_model}")
col_model, col_info = st.sidebar.columns([3, 1])
with col_model:
    if st.button("Change model", use_container_width=True):
        st.session_state['selected_model'] = None
        for key in ('manual_result',):
            st.session_state.pop(key, None)
        st.rerun()
with col_info:
    if st.button("ℹ", use_container_width=True, help="How to use"):
        show_instructions()

st.sidebar.divider()

mode = st.sidebar.radio(
    "Input mode",
    ["Manual input", "Select from dataset"],
)
st.sidebar.divider()

# ── Shared rendering helpers ──────────────────────────────────────────────────

def _imputed_display(feat, imputer_defaults):
    """Return the fill value used for a missing feature, formatted for display."""
    # Availability flags always mean "not available" when missing → No
    if feat.startswith('has_'):
        return "No"
    if feat == 'log10_AF_popmax':
        return f"{10 ** HARDCODED_FILLS['log10_AF_popmax']:.4g}"
    if feat in HARDCODED_FILLS:
        return f"{HARDCODED_FILLS[feat]:.4g}"
    if feat in imputer_defaults:
        val = imputer_defaults[feat]
        return ("Yes" if val >= 0.5 else "No") if feat in BINARY_FEATURES else f"{val:.4g}"
    return None


# Maps each optional feature to the availability flag that marks it as measured
_AVAILABILITY_FLAGS = {
    'log10_AF_popmax': 'has_AF_popmax',
    'spliceai':        'has_spliceai',
    'REVEL':           'has_REVEL',
    'phyloP100way':    'has_phyloP100way',
    'functional_score':'has_functional_score',
    'functional_se':   'has_functional_score',
    'functional_sd':   'has_functional_score',
    'aa_pos':          'has_aa_pos',
}

FEATURE_HELP = {
    'nmd_aa_below_830': (
        "Flagged when ALL three conditions are true:"
        "variant is nonsense (stop gained),"
        "variant is frameshift, and"
        "amino acid position < 830"
    ),
}


def _validate_inputs(inputs: dict):
    """Return list of (level, message) for logical inconsistencies in variant inputs.
    level is 'error' (blocks prediction) or 'warning' (informational)."""
    issues = []
    e117   = bool(inputs.get('is_exon_1_17', 0))
    e18    = bool(inputs.get('is_exon18', 0))
    e4     = bool(inputs.get('is_exon4', 0))
    n      = int(inputs.get('exon_number', 0) or 0)

    # Exon flag conflicts
    if e117 and e18:
        issues.append(('error', 'Exon 1–17 and Exon 18 cannot both be selected.'))
    if e18 and n != 18:
        issues.append(('warning', f'Exon 18 is flagged but Exon Number is {n}.'))
    if e4 and n != 4:
        issues.append(('warning', f'Exon 4 is flagged but Exon Number is {n}.'))
    if e117 and n > 17:
        issues.append(('warning', f'Exon 1–17 is flagged but Exon Number is {n}.'))

    # NMD flag: should be 1 iff aa_pos < 830 AND is_nonsense AND is_frameshift
    has_aa      = bool(inputs.get('has_aa_pos', 0))
    aa_pos      = inputs.get('aa_pos')
    nmd         = bool(inputs.get('nmd_aa_below_830', 0))
    is_nonsense = bool(inputs.get('is_nonsense', 0))
    is_fs       = bool(inputs.get('is_frameshift', 0))

    if has_aa and aa_pos is not None and not (isinstance(aa_pos, float) and np.isnan(aa_pos)):
        aa_pos = float(aa_pos)
        nmd_conditions_met = aa_pos < 830 and is_nonsense and is_fs
        if nmd_conditions_met and not nmd:
            issues.append(('warning',
                f'AA position {int(aa_pos)} < 830, variant is nonsense and frameshift — '
                f'"AA < 830" (NMD trigger) should be selected.'))
        elif nmd and not nmd_conditions_met:
            reasons = []
            if aa_pos >= 830:
                reasons.append(f'AA position {int(aa_pos)} ≥ 830')
            if not is_nonsense:
                reasons.append('variant is not nonsense')
            if not is_fs:
                reasons.append('variant is not frameshift')
            issues.append(('warning',
                f'"AA < 830" (NMD trigger) is selected but: {", ".join(reasons)}.'))

    return issues


def _imputed_features(inputs: dict) -> set:
    """Return the set of feature names whose values were default-filled (not measured)."""
    return {
        feat for feat, flag in _AVAILABILITY_FLAGS.items()
        if not inputs.get(flag, 1)
    }


_IMAGES_DIR = Path(__file__).parent.parent / 'src' / 'images'

_TREE_IMAGES = {
    "Core model":                                        _IMAGES_DIR / 'model.png',
    "Core + missense predictor (REVEL)":                 _IMAGES_DIR / 'model_revel.png',
    "Core + missense predictor + conservation (phyloP)": _IMAGES_DIR / 'model_revel_phylo.png',
}

def _pred_card_small_html(pred_class, pred_proba, title="Predicted"):
    label = "PATHOGENIC" if pred_class == 1 else "BENIGN"
    conf  = round(pred_proba * 100, 1)
    if conf < 75:
        border_color = "#b8b400"
        label_color  = "#b8b400"
        border_width = "2px"
    else:
        border_color = "#c0392b" if pred_class == 1 else "#27ae60"
        label_color  = border_color
        border_width = "2px"
    return f"""
    <div style="font-size:1rem; font-weight:normal; color:#333;
                margin-bottom:4px; text-align:center;">{title}</div>
    <div style="border:{border_width} solid {border_color}; border-radius:6px;
                padding:6px 8px; text-align:center;">
        <div style="font-size:1.1rem; font-weight:700; color:{label_color};">{label}</div>
        <div style="font-size:0.75rem; color:#777; margin-top:2px;">Confidence: {conf}%</div>
    </div>"""


def render_tree(model_name: str):
    img_path = _TREE_IMAGES.get(model_name)
    if img_path and img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning("Tree image not found.")


def _pred_card_html(pred_class, pred_proba, title="Predicted"):
    label  = "PATHOGENIC" if pred_class == 1 else "BENIGN"
    color  = "#c0392b" if pred_class == 1 else "#27ae60"
    conf   = round(pred_proba * 100, 1)
    if conf < 75:
        border_color = "#b8b400"
        border_width = "3px"
        label_color  = "#b8b400"
    else:
        border_color = color
        border_width = "2px"
        label_color  = color
    return f"""
    <div style="font-size:1.15rem; font-weight:normal; color:#333;
                margin-bottom:6px; text-align:center;">{title}</div>
    <div style="border:{border_width} solid {border_color}; border-radius:8px;
                padding:12px; text-align:center;">
        <div style="font-size:1.6rem; font-weight:700; color:{label_color};">{label}</div>
        <div style="font-size:0.85rem; color:#555; margin-top:4px;">Confidence: {conf}%</div>
    </div>"""


def render_prediction_card(pred_class, pred_proba, missing):
    label_text = "PATHOGENIC" if pred_class == 1 else "BENIGN"
    label_color = "#c0392b" if pred_class == 1 else "#27ae60"
    conf_pct = round(pred_proba * 100, 1)

    col_pred, col_conf = st.columns([1, 2])
    with col_pred:
        st.markdown(
            f"""
            <div style="border:2px solid {label_color}; border-radius:8px;
                        padding:20px; text-align:center;">
                <div style="font-size:0.85rem; color:#888; margin-bottom:4px;">
                    Predicted classification
                </div>
                <div style="font-size:2rem; font-weight:700; color:{label_color};">
                    {label_text}
                </div>
                <div style="font-size:0.9rem; color:#555; margin-top:6px;">
                    Confidence: {conf_pct}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_conf:
        if missing:
            st.warning(
                f"Missing optional data: {', '.join(missing)}. "
                "Default values were used. Results may be less precise."
            )
        else:
            st.success("All feature data provided.")
        st.progress(pred_proba, text=f"{conf_pct}% confidence in {label_text.lower()}")


def render_decision_path(path, imputed: set = None):
    if not path:
        return
    if imputed is None:
        imputed = set()
    st.subheader("Decision Path")
    split_steps = [s for s in path if s['type'] == 'split']

    rows_html = ""
    for i, step in enumerate(split_steps, start=1):
        feat = step['feature']
        feat_label = FEATURE_LABELS.get(feat, feat)
        met = step['met']
        val = step['value']
        threshold = step['threshold']

        if feat in BINARY_FEATURES:
            requires  = "No"
            value_str = "No" if val == 0 else "Yes"
        elif feat == 'log10_AF_popmax':
            requires  = f"≤ {10 ** threshold:.4g}"
            value_str = f"{10 ** val:.4g}"
        else:
            requires  = f"≤ {threshold}"
            value_str = str(val)

        if feat in imputed:
            value_str += " <span style='font-size:0.75rem; color:#aaa; font-style:italic;'>(default)</span>"

        result_color = "#27ae60" if met else "#e67e22"
        result_text  = "✓" if met else "✗"

        rows_html += f"""
        <tr style="border-bottom:1px solid #e8e8e8;">
          <td style="padding:10px 14px; font-weight:600; color:#888;">{i}</td>
          <td style="padding:10px 14px; font-weight:600;">{feat_label}</td>
          <td style="padding:10px 14px; font-family:monospace;">{value_str}</td>
          <td style="padding:10px 14px; font-family:monospace; color:#888;">{requires}</td>
          <td style="padding:10px 14px; font-weight:700; color:{result_color};">{result_text}</td>
        </tr>"""

    st.markdown(
        f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.92rem;
                      border:1px solid #e0e0e0; border-radius:6px; overflow:hidden;">
          <thead>
            <tr style="background:#f5f5f5; border-bottom:2px solid #d0d0d0;">
              <th style="padding:10px 14px; text-align:left; color:#555;">#</th>
              <th style="padding:10px 14px; text-align:left; color:#555;">Feature</th>
              <th style="padding:10px 14px; text-align:left; color:#555;">Your value</th>
              <th style="padding:10px 14px; text-align:left; color:#555;">Requires</th>
              <th style="padding:10px 14px; text-align:left; color:#555;"></th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Mode A: Manual input
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Manual input":

    inputs = {}

    # ── Variant Type ──────────────────────────────────────────────────────────
    st.markdown("#### Variant Type")
    vt_feats = FEATURE_GROUPS['Variant Type']
    for row_start in range(0, len(vt_feats), 4):
        cols = st.columns(4)
        for col, feat in zip(cols, vt_feats[row_start:row_start + 4]):
            with col:
                inputs[feat] = int(
                    st.checkbox(FEATURE_LABELS[feat],
                                value=bool(FEATURE_DEFAULTS[feat]), key=feat,
                                help=FEATURE_HELP.get(feat))
                )

    st.divider()

    # ── Exon & Position ───────────────────────────────────────────────────────
    st.markdown("#### Exon & Position")
    c1, c2, c3, c4, _ = st.columns(5)
    with c1:
        inputs['exon_number'] = float(
            st.number_input(FEATURE_LABELS['exon_number'], min_value=1, max_value=18,
                            value=int(FEATURE_DEFAULTS['exon_number']), step=1, key='exon_number')
        )
    with c2:
        inputs['codon_change_length'] = float(
            st.number_input(FEATURE_LABELS['codon_change_length'], min_value=-50, max_value=50,
                            value=int(FEATURE_DEFAULTS['codon_change_length']),
                            step=1, key='codon_change_length')
        )
    with c3:
        inputs['has_aa_pos'] = int(
            st.checkbox(FEATURE_LABELS['has_aa_pos'],
                        value=bool(FEATURE_DEFAULTS['has_aa_pos']), key='has_aa_pos')
        )
    with c4:
        aa_val = st.number_input(
            FEATURE_LABELS['aa_pos'], min_value=0.0,
            value=imputer_defaults.get('aa_pos', 0.0), step=1.0, key='aa_pos',
            disabled=not inputs['has_aa_pos'],
        )
        inputs['aa_pos'] = float(aa_val) if inputs['has_aa_pos'] else np.nan

    ep_bool = [f for f in FEATURE_GROUPS['Exon & Position']
               if f not in ('exon_number', 'codon_change_length', 'has_aa_pos', 'aa_pos')]
    for row_start in range(0, len(ep_bool), 4):
        cols = st.columns(4)
        for col, feat in zip(cols, ep_bool[row_start:row_start + 4]):
            with col:
                inputs[feat] = int(
                    st.checkbox(FEATURE_LABELS[feat],
                                value=bool(FEATURE_DEFAULTS[feat]), key=feat,
                                help=FEATURE_HELP.get(feat))
                )

    exon_issues   = _validate_inputs(inputs)
    exon_conflict = any(lvl == 'error' for lvl, _ in exon_issues)
    for level, msg in exon_issues:
        (st.error if level == 'error' else st.warning)(msg)

    st.divider()

    # ── Scores ────────────────────────────────────────────────────────────────
    st.markdown("#### Scores")
    n_score_cols = 3 + int(has_revel) + int(has_phylop)
    score_cols = st.columns(n_score_cols)
    ci = 0

    with score_cols[ci]:
        st.markdown("**Allele Frequency**")
        inputs['has_AF_popmax'] = int(
            st.checkbox(FEATURE_LABELS['has_AF_popmax'], value=False, key='has_AF_popmax')
        )
        af_raw = st.number_input(
            "AF popmax", value=0.001, min_value=0.0, max_value=1.0,
            step=0.0001, format="%.6f", key='af_popmax_raw',
            disabled=not inputs['has_AF_popmax'],
        )
        inputs['log10_AF_popmax'] = (-6.0 if af_raw == 0.0 else float(np.log10(af_raw))) if inputs['has_AF_popmax'] else -6.0
    ci += 1

    with score_cols[ci]:
        st.markdown("**SpliceAI**")
        inputs['has_spliceai'] = int(
            st.checkbox(FEATURE_LABELS['has_spliceai'], value=False, key='has_spliceai')
        )
        sp_val = st.number_input(
            FEATURE_LABELS['spliceai'], value=0.0, min_value=0.0, max_value=1.0,
            step=0.01, key='spliceai', disabled=not inputs['has_spliceai'],
        )
        inputs['spliceai'] = float(sp_val) if inputs['has_spliceai'] else 0.0
    ci += 1

    if has_revel:
        with score_cols[ci]:
            st.markdown("**REVEL**")
            inputs['has_REVEL'] = int(
                st.checkbox(FEATURE_LABELS['has_REVEL'], value=False, key='has_REVEL')
            )
            rv_val = st.number_input(
                FEATURE_LABELS['REVEL'], value=imputer_defaults.get('REVEL', 0.5),
                min_value=0.0, max_value=1.0, step=0.01, key='REVEL',
                disabled=not inputs['has_REVEL'],
            )
            inputs['REVEL'] = float(rv_val) if inputs['has_REVEL'] else np.nan
        ci += 1

    if has_phylop:
        with score_cols[ci]:
            st.markdown("**phyloP100way**")
            inputs['has_phyloP100way'] = int(
                st.checkbox(FEATURE_LABELS['has_phyloP100way'], value=False, key='has_phyloP100way')
            )
            ph_val = st.number_input(
                FEATURE_LABELS['phyloP100way'], value=imputer_defaults.get('phyloP100way', 0.0),
                min_value=-20.0, max_value=10.0, step=0.01, key='phyloP100way',
                disabled=not inputs['has_phyloP100way'],
            )
            inputs['phyloP100way'] = float(ph_val) if inputs['has_phyloP100way'] else np.nan
        ci += 1

    with score_cols[ci]:
        st.markdown("**Functional Score**")
        inputs['has_functional_score'] = int(
            st.checkbox(FEATURE_LABELS['has_functional_score'], value=False,
                        key='has_functional_score')
        )
        fs_val = st.number_input(
            FEATURE_LABELS['functional_score'],
            value=imputer_defaults.get('functional_score', 0.0), key='functional_score',
            disabled=not inputs['has_functional_score'],
        )
        se_val = st.number_input(
            FEATURE_LABELS['functional_se'], value=0.0, min_value=0.0, key='functional_se',
            disabled=not inputs['has_functional_score'],
        )
        sd_val = st.number_input(
            FEATURE_LABELS['functional_sd'], value=0.0, min_value=0.0, key='functional_sd',
            disabled=not inputs['has_functional_score'],
        )
        if inputs['has_functional_score']:
            inputs['functional_score'] = float(fs_val)
            inputs['functional_se']    = float(se_val)
            inputs['functional_sd']    = float(sd_val)
        else:
            inputs['functional_score'] = inputs['functional_se'] = inputs['functional_sd'] = np.nan

    st.divider()
    classify_btn = st.button("Classify Variant", type="primary",
                             use_container_width=True, disabled=exon_conflict)

    # ── Results ───────────────────────────────────────────────────────────────

    if classify_btn:
        dt_pred, dt_proba, dt_path = predict(bundle, inputs)
        rf_pred, rf_proba, _       = predict(rf_bundle, inputs)
        st.session_state['manual_result'] = (dt_pred, dt_proba, dt_path, rf_pred, rf_proba, dict(inputs))

    if 'manual_result' not in st.session_state:
        st.stop()

    dt_pred, dt_proba, dt_path, rf_pred, rf_proba, used_inputs = st.session_state['manual_result']

    missing = []
    if not used_inputs.get('has_spliceai'):
        missing.append('SpliceAI')
    if has_revel and not used_inputs.get('has_REVEL'):
        missing.append('REVEL')
    if has_phylop and not used_inputs.get('has_phyloP100way'):
        missing.append('phyloP100way')
    if not used_inputs.get('has_functional_score'):
        missing.append('Functional score')
    if not used_inputs.get('has_AF_popmax'):
        missing.append('Population AF')
    if not used_inputs.get('has_aa_pos'):
        missing.append('AA position')

    st.divider()
    col_dt, col_rf, col_warn = st.columns([1, 1, 2])
    with col_dt:
        st.markdown(_pred_card_html(dt_pred, dt_proba, "Decision Tree"), unsafe_allow_html=True)
    with col_rf:
        st.markdown(_pred_card_html(rf_pred, rf_proba, "Random Forest"), unsafe_allow_html=True)
    with col_warn:
        if missing:
            st.warning(f"Missing optional data: {', '.join(missing)}. Default values used.")
        else:
            st.success("All feature data provided.")

    st.divider()
    render_decision_path(dt_path, _imputed_features(used_inputs))
    with st.expander("Decision Tree Structure", expanded=True):
        render_tree(selected_model)


# ══════════════════════════════════════════════════════════════════════════════
# Mode B: Select from dataset
# ══════════════════════════════════════════════════════════════════════════════

else:
    try:
        df = load_dataset(_mtime=os.path.getmtime(DATA_PATH))
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    def clean_key(k):
        return str(k).replace('_<NA>', '').replace('<NA>_', '').replace('<NA>', '').strip('_')

    def protein_change(k):
        p = str(k).split('_')
        if len(p) == 6 and p[3] != '<NA>' and p[4] != '<NA>' and p[5] != '<NA>':
            aa_old, aa_pos, aa_new = p[3], p[4], p[5]
            if aa_new == aa_old:   return f"p.{aa_old}{aa_pos}="
            elif aa_new == 'Ter':  return f"p.{aa_old}{aa_pos}*"
            elif aa_new == 'fs':   return f"p.{aa_old}{aa_pos}fs"
            return f"p.{aa_old}{aa_pos}{aa_new}"
        return ""

    # Build parallel lists — #Uploaded_variation is unique so safe as display key
    all_hgvs     = df['#Uploaded_variation'].tolist()
    all_vk       = df['variant_key'].tolist()
    all_proteins = [protein_change(k) for k in all_vk]

    # Error filter
    _df_fp = os.path.getmtime(DATA_PATH)
    dt_wrong, rf_wrong, both_wrong = get_error_sets(selected_model, _df_fp)
    error_filter = st.sidebar.radio(
        "Filter by prediction",
        ["All variants", "DT misclassified", "RF misclassified", "Both misclassified"],
    )
    err_set = {'DT misclassified': dt_wrong, 'RF misclassified': rf_wrong,
               'Both misclassified': both_wrong}.get(error_filter)

    # Variant type filter
    TYPE_FEATS = [
        'is_missense', 'is_synonymous', 'is_frameshift', 'is_nonsense',
        'is_intron', 'is_utr', 'is_duplication', 'is_deletion',
        'is_insertion', 'is_inframe_ins_or_del',
    ]
    type_labels = {f: FEATURE_LABELS[f] for f in TYPE_FEATS}
    selected_types = st.sidebar.multiselect(
        "Variant type",
        options=list(type_labels.keys()),
        format_func=lambda f: type_labels[f],
        default=[],
        placeholder="All types",
    )
    type_allowed = None
    if selected_types:
        type_allowed = set(df[df[selected_types].eq(1).any(axis=1)]['#Uploaded_variation'])

    # Apply both filters to the parallel lists simultaneously
    rows_iter = zip(all_hgvs, all_vk, all_proteins)
    filtered_rows = [
        (h, vk, pc) for h, vk, pc in rows_iter
        if (err_set is None or h in err_set)
        and (type_allowed is None or h in type_allowed)
    ]
    display_keys = [r[0] for r in filtered_rows]
    vk_keys      = [r[1] for r in filtered_rows]
    protein_keys = [r[2] for r in filtered_rows]

    st.sidebar.divider()
    search = st.sidebar.text_input("Search", placeholder="key, position or p.Met1Leu")
    if search:
        q = search.lower()
        idx = [i for i, (h, p) in enumerate(zip(display_keys, protein_keys))
               if q in h.lower() or q in p.lower()]
        filtered_display = [display_keys[i] for i in idx]
        filtered_vk      = [vk_keys[i] for i in idx]
    else:
        filtered_display = display_keys
        filtered_vk      = vk_keys

    if not filtered_display:
        st.sidebar.warning("No variants match your search.")
        st.stop()

    sel_idx = st.sidebar.selectbox(
        f"Variant ({len(filtered_display)} shown)",
        range(len(filtered_display)),
        format_func=lambda i: filtered_display[i],
        index=0,
    )
    selected_display = filtered_display[sel_idx]
    selected_key     = filtered_vk[sel_idx]

    # ── Main area ─────────────────────────────────────────────────────────────

    row = df[df['#Uploaded_variation'] == selected_display].iloc[0]
    dt_pred, dt_proba, dt_path = predict_from_row(bundle, row)
    rf_pred, rf_proba, _       = predict_from_row(rf_bundle, row)

    # Parse variant key: REF_ALT_POS_AAold_AApos_AAnew
    _parts = selected_key.split('_')
    _aa_old, _aa_pos, _aa_new = _parts[3], _parts[4], _parts[5]
    if _aa_old != '<NA>' and _aa_pos != '<NA>' and _aa_new != '<NA>':
        if _aa_new == _aa_old:
            protein_change = f"p.{_aa_old}{_aa_pos}="
        elif _aa_new == 'Ter':
            protein_change = f"p.{_aa_old}{_aa_pos}*"
        elif _aa_new == 'fs':
            protein_change = f"p.{_aa_old}{_aa_pos}fs"
        else:
            protein_change = f"p.{_aa_old}{_aa_pos}{_aa_new}"
    else:
        protein_change = None

    _hgvs_name = row.get('#Uploaded_variation', None)

    _widths = [1.5] + ([1] if protein_change and _hgvs_name else []) + [0.5, 2]
    _label_col, *_copy_cols = st.columns([0.6] + _widths)
    with _label_col:
        st.markdown(
            "<div style='font-size:1.1rem; color:#333; font-weight:600; margin-top:10px;'>ClinVar search:</div>",
            unsafe_allow_html=True,
        )
    _ci = 0
    if _hgvs_name and not pd.isna(_hgvs_name):
        with _copy_cols[_ci]:
            st.code(str(_hgvs_name).split('(')[0].strip(), language=None)
        _ci += 1
    if protein_change:
        with _copy_cols[_ci]:
            st.code(protein_change, language=None)
        _ci += 1
    with _copy_cols[_ci]:
        pass  # spacer
    with _copy_cols[_ci + 1]:
        escaped_key = selected_key.replace("'", "\\'")
        components.html(
            f"""
            <button onclick="
                var ta = document.createElement('textarea');
                ta.value = '{escaped_key}';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                this.innerText='✓ Copied';
                setTimeout(() => this.innerText='📋 Copy internal ID', 2000);
            " style="
                margin-top:8px; font-size:0.75rem; color:#888;
                background:none; border:1px solid #ccc; border-radius:4px;
                padding:4px 10px; cursor:pointer; white-space:nowrap;
            ">📋 Copy internal ID</button>
            """,
            height=42,
        )

    st.write("")

    true_class = int(row['VariantClassification'])
    true_label = "PATHOGENIC" if true_class == 1 else "BENIGN"
    true_color = "#c0392b" if true_class == 1 else "#27ae60"
    true_bg    = "#fdecea"  if true_class == 1 else "#e8f5e9"

    missing = []
    if not row.get('has_spliceai', 0):
        missing.append('SpliceAI')
    if has_revel and not row.get('has_REVEL', 0):
        missing.append('REVEL')
    if has_phylop and not row.get('has_phyloP100way', 0):
        missing.append('phyloP100way')
    if not row.get('has_functional_score', 0):
        missing.append('Functional score')
    if not row.get('has_AF_popmax', 0):
        missing.append('Population AF')
    if pd.isna(row.get('aa_pos')):
        missing.append('AA position')

    if missing:
        st.warning(f"Missing: {', '.join(missing)}. Default values used.")

    col_dt, col_rf, col_true = st.columns(3)

    with col_dt:
        dt_correct = dt_pred == true_class
        verdict = f"<div style='font-size:0.8rem; font-weight:700; color:{'#27ae60' if dt_correct else '#c0392b'}; margin-top:6px;'>{'✓ Correct' if dt_correct else '✗ Wrong'}</div>"
        st.markdown(_pred_card_html(dt_pred, dt_proba, "Decision Tree") + verdict, unsafe_allow_html=True)

    with col_rf:
        rf_correct = rf_pred == true_class
        verdict = f"<div style='font-size:0.8rem; font-weight:700; color:{'#27ae60' if rf_correct else '#c0392b'}; margin-top:6px;'>{'✓ Correct' if rf_correct else '✗ Wrong'}</div>"
        st.markdown(_pred_card_html(rf_pred, rf_proba, "Random Forest") + verdict, unsafe_allow_html=True)

    with col_true:
        st.markdown(
            f"""
            <div style="font-size:1.15rem; font-weight:normal; color:#333;
                        margin-bottom:6px; text-align:center;">True label</div>
            <div style="border:2px solid {true_color}; border-radius:8px;
                        padding:12px; text-align:center; background:{true_bg};">
                <div style="font-size:1.6rem; font-weight:700; color:{true_color};">{true_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── What if? ─────────────────────────────────────────────────────────────

    with st.expander("Modify variant features and assess how the prediction change", expanded=False):

        # Reset counter — incrementing it changes all widget keys, forcing re-init
        if 'wif_reset_counter' not in st.session_state:
            st.session_state['wif_reset_counter'] = 0

        # Auto-reset when a different variant is selected
        if st.session_state.get('_wif_key') != selected_display:
            st.session_state['wif_reset_counter'] += 1
            st.session_state['_wif_key'] = selected_display

        _rc = st.session_state['wif_reset_counter']  # appended to every widget key

        def _wif_init(feat, row):
            """Return initial value for a WIF input in display space."""
            if feat == 'has_aa_pos':
                return int(not pd.isna(row.get('aa_pos')))
            raw = row[feat] if feat in row.index else np.nan
            if pd.isna(raw):
                if feat == 'log10_AF_popmax':
                    return 1e-6
                if feat in HARDCODED_FILLS:
                    return HARDCODED_FILLS[feat]
                v = imputer_defaults.get(feat, 0.0)
                return int(round(v)) if feat in BINARY_FEATURES else float(v)
            if feat == 'log10_AF_popmax':
                return float(10 ** float(raw))
            return int(raw) if feat in BINARY_FEATURES else float(raw)

        wif = {}
        col_form, col_result = st.columns([5, 5])

        with col_form:
            # ── Variant Type ──────────────────────────────────────────
            with st.container(border=True):
                st.markdown("**Variant Type**")
                vt_feats = [f for f in FEATURE_GROUPS['Variant Type'] if f in feat_cols_set]
                for row_start in range(0, len(vt_feats), 3):
                    for col, feat in zip(st.columns(3), vt_feats[row_start:row_start + 3]):
                        with col:
                            wif[feat] = int(st.checkbox(FEATURE_LABELS[feat],
                                value=bool(_wif_init(feat, row)), key=f'wif_{feat}_{_rc}',
                                help=FEATURE_HELP.get(feat)))

            # ── Exon & Position ───────────────────────────────────────
            with st.container(border=True):
                st.markdown("**Exon & Position**")
                ni1, ni2, ni3 = st.columns(3)
                with ni1:
                    if 'exon_number' in feat_cols_set:
                        wif['exon_number'] = float(st.number_input(
                            FEATURE_LABELS['exon_number'], min_value=1, max_value=18,
                            value=int(_wif_init('exon_number', row)), step=1, key=f'wif_exon_number_{_rc}'))
                with ni2:
                    if 'codon_change_length' in feat_cols_set:
                        wif['codon_change_length'] = float(st.number_input(
                            FEATURE_LABELS['codon_change_length'], min_value=-50, max_value=50,
                            value=int(_wif_init('codon_change_length', row)), step=1, key=f'wif_codon_change_length_{_rc}'))
                with ni3:
                    if 'has_aa_pos' in feat_cols_set:
                        wif['has_aa_pos'] = int(st.checkbox(FEATURE_LABELS['has_aa_pos'],
                            value=bool(_wif_init('has_aa_pos', row)), key=f'wif_has_aa_pos_{_rc}'))
                        aa_init = row['aa_pos'] if not pd.isna(row.get('aa_pos')) else imputer_defaults.get('aa_pos', 0.0)
                        aa_val = st.number_input(FEATURE_LABELS['aa_pos'],
                            min_value=0.0, value=float(aa_init), step=1.0,
                            key=f'wif_aa_pos_{_rc}', disabled=not wif['has_aa_pos'])
                        wif['aa_pos'] = float(aa_val) if wif['has_aa_pos'] else np.nan
                    else:
                        wif['has_aa_pos'] = 0
                        wif['aa_pos'] = np.nan

                ep_bool = [f for f in FEATURE_GROUPS['Exon & Position']
                           if f not in ('exon_number', 'codon_change_length', 'has_aa_pos', 'aa_pos')
                           and f in feat_cols_set]
                for row_start in range(0, len(ep_bool), 3):
                    for col, feat in zip(st.columns(3), ep_bool[row_start:row_start + 3]):
                        with col:
                            wif[feat] = int(st.checkbox(FEATURE_LABELS[feat],
                                value=bool(_wif_init(feat, row)), key=f'wif_{feat}_{_rc}',
                                help=FEATURE_HELP.get(feat)))

        # ── Scores + Result in the right panel ────────────────────────────
        with col_result:
            if st.button("Reset to original values", use_container_width=True):
                st.session_state['wif_reset_counter'] += 1
                st.rerun()
            result_placeholder = st.empty()   # filled AFTER wif is complete
            with st.container(border=True):
                st.markdown("**Scores**")
                n_sc = 2 + int(has_revel) + int(has_phylop)
                sc_cols = st.columns(n_sc)
                sc_i = 0

                with sc_cols[sc_i]:
                    wif['has_AF_popmax'] = int(st.checkbox(FEATURE_LABELS['has_AF_popmax'],
                        value=bool(_wif_init('has_AF_popmax', row)), key=f'wif_has_AF_popmax_{_rc}'))
                    af_init = _wif_init('log10_AF_popmax', row)
                    af_raw = st.number_input('AF popmax', value=float(af_init),
                        min_value=0.0, max_value=1.0, step=0.0001, format='%.6f',
                        key=f'wif_log10_AF_popmax_{_rc}', disabled=not wif['has_AF_popmax'])
                    wif['log10_AF_popmax'] = (-6.0 if af_raw == 0.0 else float(np.log10(af_raw))) if wif['has_AF_popmax'] else -6.0

                    wif['has_spliceai'] = int(st.checkbox(FEATURE_LABELS['has_spliceai'],
                        value=bool(_wif_init('has_spliceai', row)), key=f'wif_has_spliceai_{_rc}'))
                    sp_val = st.number_input(FEATURE_LABELS['spliceai'],
                        value=float(_wif_init('spliceai', row)),
                        min_value=0.0, max_value=1.0, step=0.01,
                        key=f'wif_spliceai_{_rc}', disabled=not wif['has_spliceai'])
                    wif['spliceai'] = float(sp_val) if wif['has_spliceai'] else 0.0
                sc_i += 1

                if has_revel:
                    with sc_cols[sc_i]:
                        wif['has_REVEL'] = int(st.checkbox(FEATURE_LABELS['has_REVEL'],
                            value=bool(_wif_init('has_REVEL', row)), key=f'wif_has_REVEL_{_rc}'))
                        rv_val = st.number_input(FEATURE_LABELS['REVEL'],
                            value=float(_wif_init('REVEL', row)),
                            min_value=0.0, max_value=1.0, step=0.01,
                            key=f'wif_REVEL_{_rc}', disabled=not wif['has_REVEL'])
                        wif['REVEL'] = float(rv_val) if wif['has_REVEL'] else np.nan
                    sc_i += 1

                if has_phylop:
                    with sc_cols[sc_i]:
                        wif['has_phyloP100way'] = int(st.checkbox(FEATURE_LABELS['has_phyloP100way'],
                            value=bool(_wif_init('has_phyloP100way', row)), key=f'wif_has_phyloP100way_{_rc}'))
                        ph_val = st.number_input(FEATURE_LABELS['phyloP100way'],
                            value=float(_wif_init('phyloP100way', row)),
                            min_value=-20.0, max_value=10.0, step=0.01,
                            key=f'wif_phyloP100way_{_rc}', disabled=not wif['has_phyloP100way'])
                        wif['phyloP100way'] = float(ph_val) if wif['has_phyloP100way'] else np.nan
                    sc_i += 1

                with sc_cols[sc_i]:
                    wif['has_functional_score'] = int(st.checkbox(FEATURE_LABELS['has_functional_score'],
                        value=bool(_wif_init('has_functional_score', row)), key=f'wif_has_functional_score_{_rc}'))
                    fs_val = st.number_input(FEATURE_LABELS['functional_score'],
                        value=float(_wif_init('functional_score', row)),
                        key=f'wif_functional_score_{_rc}', disabled=not wif['has_functional_score'])
                    se_val = st.number_input(FEATURE_LABELS['functional_se'],
                        value=float(_wif_init('functional_se', row)),
                        min_value=0.0, key=f'wif_functional_se_{_rc}', disabled=not wif['has_functional_score'])
                    sd_val = st.number_input(FEATURE_LABELS['functional_sd'],
                        value=float(_wif_init('functional_sd', row)),
                        min_value=0.0, key=f'wif_functional_sd_{_rc}', disabled=not wif['has_functional_score'])
                    if wif['has_functional_score']:
                        wif['functional_score'] = float(fs_val)
                        wif['functional_se']    = float(se_val)
                        wif['functional_sd']    = float(sd_val)
                    else:
                        wif['functional_score'] = wif['functional_se'] = wif['functional_sd'] = np.nan

        # All wif values now collected — compute and fill the placeholder
        wif_dt_pred, wif_dt_proba, _ = predict(bundle, wif)
        wif_rf_pred, wif_rf_proba, _ = predict(rf_bundle, wif)
        def _wif_flag(orig_pred, orig_proba, new_pred, new_proba, model_name):
            orig_label = "PATHOGENIC" if orig_pred == 1 else "BENIGN"
            new_label  = "PATHOGENIC" if new_pred  == 1 else "BENIGN"
            if new_pred != orig_pred:
                return (
                    f"<div style='color:#c0392b; font-size:0.88rem; font-weight:600; margin:2px 0;'>"
                    f"⚠ {model_name}: {orig_label} → {new_label}</div>"
                )
            diff = new_proba - orig_proba
            if abs(diff) >= 0.001:
                arrow = "↑" if diff > 0 else "↓"
                return (
                    f"<div style='color:#e67e22; font-size:0.88rem; margin:2px 0;'>"
                    f"{arrow} {model_name}: {abs(diff)*100:.1f}% (still {new_label})</div>"
                )
            return (
                f"<div style='color:#27ae60; font-size:0.88rem; margin:2px 0;'>"
                f"✓ {model_name}: Unchanged</div>"
            )

        wif_exon_issues = _validate_inputs(wif)
        with result_placeholder.container():
            for level, msg in wif_exon_issues:
                (st.error if level == 'error' else st.warning)(msg)
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(_pred_card_small_html(wif_dt_pred, wif_dt_proba, "Decision Tree"), unsafe_allow_html=True)
                st.markdown(_wif_flag(dt_pred, dt_proba, wif_dt_pred, wif_dt_proba, "DT"), unsafe_allow_html=True)
            with rc2:
                st.markdown(_pred_card_small_html(wif_rf_pred, wif_rf_proba, "Random Forest"), unsafe_allow_html=True)
                st.markdown(_wif_flag(rf_pred, rf_proba, wif_rf_pred, wif_rf_proba, "RF"), unsafe_allow_html=True)

    # ── Feature values helper ─────────────────────────────────────────────────
    def _feat_row(feat, row):
        label = FEATURE_LABELS.get(feat, feat)
        if feat == 'has_aa_pos':
            raw = 0.0 if pd.isna(row.get('aa_pos')) else 1.0
        else:
            raw = row[feat] if feat in row.index else np.nan
        imputed = pd.isna(raw)
        if imputed and feat == 'log10_AF_popmax' and row.get('has_AF_popmax', 0) == 1:
            dv, note = "< 1×10⁻⁶", ""
        elif imputed:
            fill = _imputed_display(feat, imputer_defaults)
            dv   = f"<span style='color:#aaa;'>{fill}</span>" if fill else "<em style='color:#aaa;'>—</em>"
            note = " <span style='font-size:0.75rem;color:#aaa;'>(imputed)</span>"
        elif feat == 'log10_AF_popmax':
            dv, note = f"{10 ** raw:.4g}", ""
        elif feat in BINARY_FEATURES:
            dv, note = ("Yes" if raw == 1 else "No"), ""
        elif feat == 'exon_number':
            dv, note = str(int(raw)), ""
        else:
            dv   = f"{raw:.4g}" if isinstance(raw, float) else str(raw)
            note = ""
        bg       = "background:#dbeafe;" if dv == "Yes" and not feat.startswith('has_') else ""
        val_color = "color:#bbb;" if feat.startswith('has_') and dv == "No" else ""
        return (
            f"<tr style='border-bottom:1px solid #f0f0f0;'>"
            f"<td style='padding:6px 10px; color:#555; font-size:0.92rem;'>{label}</td>"
            f"<td style='padding:6px 10px; font-family:monospace; font-weight:600; font-size:0.92rem; {bg}{val_color}'>{dv}{note}</td>"
            f"</tr>"
        )

    def _group_html(title, feats, row):
        header = (
            f"<tr><td colspan='2' style='padding:7px 10px 3px; font-size:0.78rem; "
            f"font-weight:700; color:#888; text-transform:uppercase; "
            f"letter-spacing:0.05em; background:#f9f9f9;'>{title}</td></tr>"
        )
        rows = "".join(_feat_row(f, row) for f in feats if f in feat_cols_set)
        return header + rows if rows else ""

    def _wrap_table(body):
        return (
            f"<table style='width:100%; border-collapse:collapse; font-size:0.88rem; "
            f"border:1px solid #e0e0e0; border-radius:6px; overflow:hidden;'>"
            f"<tbody>{body}</tbody></table>"
        )

    st.subheader("Feature Values")
    col_vt, col_ep, col_sc = st.columns([1, 1.2, 1])

    with col_vt:
        body = _group_html("Variant Type", FEATURE_GROUPS['Variant Type'], row)
        st.markdown(_wrap_table(body), unsafe_allow_html=True)

    with col_ep:
        body = _group_html("Exon & Position", FEATURE_GROUPS['Exon & Position'], row)
        st.markdown(_wrap_table(body), unsafe_allow_html=True)

    with col_sc:
        body = _group_html("Allele Frequency", FEATURE_GROUPS['Allele Frequency'], row)
        body += _group_html("SpliceAI", FEATURE_GROUPS['SpliceAI'], row)
        if has_revel:
            body += _group_html("REVEL", ['has_REVEL', 'REVEL'], row)
        if has_phylop:
            body += _group_html("phyloP100way", ['has_phyloP100way', 'phyloP100way'], row)
        body += _group_html("Functional Scores", FEATURE_GROUPS['Functional Scores'], row)
        st.markdown(_wrap_table(body), unsafe_allow_html=True)

    st.divider()
    row_inputs = {flag: row.get(flag, 0) for flag in _AVAILABILITY_FLAGS.values()}
    row_inputs['has_aa_pos'] = 0 if pd.isna(row.get('aa_pos')) else 1
    render_decision_path(dt_path, _imputed_features(row_inputs))
    with st.expander("Decision Tree Structure", expanded=True):
        render_tree(selected_model)
