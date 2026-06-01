import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.features import (
    BINARY_FEATURES, FEATURE_DEFAULTS, FEATURE_GROUPS, FEATURE_LABELS,
)
from src.model import HARDCODED_FILLS, MODEL_CATALOG, RF_CATALOG, get_imputer_defaults, load_bundle, load_rf_bundle, predict, predict_dataset, predict_from_row

DATA_PATH = Path(__file__).parent.parent / 'data' / 'concepts_withVariantKey.csv'

st.set_page_config(
    page_title="LDLR Variant Pathogenicity Classifier",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 320px; max-width: 320px; }
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
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner="Computing predictions…")
def get_error_sets(model_name: str):
    """Return sets of variant keys where DT, RF, or both are wrong."""
    df  = load_dataset()
    dt  = get_bundle(model_name)
    rf  = get_rf_bundle(model_name)
    true_labels = df['VariantClassification'].astype(int).values
    dt_preds    = predict_dataset(dt, df)
    rf_preds    = predict_dataset(rf, df)
    keys = df['variant_key'].values
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

st.sidebar.markdown(f"**Model:** {selected_model}")
if st.sidebar.button("Change model", use_container_width=True):
    st.session_state['selected_model'] = None
    for key in ('manual_result',):
        st.session_state.pop(key, None)
    st.rerun()

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
                                value=bool(FEATURE_DEFAULTS[feat]), key=feat)
                )

    st.divider()

    # ── Exon & Position ───────────────────────────────────────────────────────
    st.markdown("#### Exon & Position")

    # Numeric inputs first row
    c1, c2, c3, c4 = st.columns(4)
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
        if inputs['has_aa_pos']:
            inputs['aa_pos'] = st.number_input(
                FEATURE_LABELS['aa_pos'], min_value=0.0,
                value=imputer_defaults.get('aa_pos', 0.0), step=1.0, key='aa_pos',
            )
        else:
            inputs['aa_pos'] = np.nan

    # Boolean grid (remaining exon/position features)
    ep_bool = [f for f in FEATURE_GROUPS['Exon & Position']
               if f not in ('exon_number', 'codon_change_length', 'has_aa_pos', 'aa_pos')]
    for row_start in range(0, len(ep_bool), 4):
        cols = st.columns(4)
        for col, feat in zip(cols, ep_bool[row_start:row_start + 4]):
            with col:
                inputs[feat] = int(
                    st.checkbox(FEATURE_LABELS[feat],
                                value=bool(FEATURE_DEFAULTS[feat]), key=feat)
                )

    if inputs.get('is_exon_1_17') and inputs.get('is_exon18'):
        st.error("A variant cannot be in Exon 1–17 and Exon 18 at the same time. Please uncheck one.")

    st.divider()

    # ── Scores ────────────────────────────────────────────────────────────────
    st.markdown("#### Scores")
    n_score_cols = 3 + int(has_revel) + int(has_phylop)
    score_cols = st.columns(n_score_cols)
    ci = 0  # column index

    with score_cols[ci]:
        st.markdown("**Allele Frequency**")
        inputs['has_AF_popmax'] = int(
            st.checkbox(FEATURE_LABELS['has_AF_popmax'], value=False, key='has_AF_popmax')
        )
        if inputs['has_AF_popmax']:
            af_raw = st.number_input(
                "AF popmax",
                value=0.001,
                min_value=0.0,
                max_value=1.0,
                step=0.0001,
                format="%.6f",
                key='af_popmax_raw',
            )
            inputs['log10_AF_popmax'] = -6.0 if af_raw == 0.0 else float(np.log10(af_raw))
        else:
            inputs['log10_AF_popmax'] = -6.0
    ci += 1

    with score_cols[ci]:
        st.markdown("**SpliceAI**")
        inputs['has_spliceai'] = int(
            st.checkbox(FEATURE_LABELS['has_spliceai'], value=False, key='has_spliceai')
        )
        if inputs['has_spliceai']:
            inputs['spliceai'] = st.number_input(
                FEATURE_LABELS['spliceai'],
                value=0.0, min_value=0.0, max_value=1.0, step=0.01, key='spliceai',
            )
        else:
            inputs['spliceai'] = 0.0
    ci += 1

    if has_revel:
        with score_cols[ci]:
            st.markdown("**REVEL**")
            inputs['has_REVEL'] = int(
                st.checkbox(FEATURE_LABELS['has_REVEL'], value=False, key='has_REVEL')
            )
            if inputs['has_REVEL']:
                inputs['REVEL'] = st.number_input(
                    FEATURE_LABELS['REVEL'],
                    value=imputer_defaults.get('REVEL', 0.5),
                    min_value=0.0, max_value=1.0, step=0.01, key='REVEL',
                )
            else:
                inputs['REVEL'] = np.nan
        ci += 1

    if has_phylop:
        with score_cols[ci]:
            st.markdown("**phyloP100way**")
            inputs['has_phyloP100way'] = int(
                st.checkbox(FEATURE_LABELS['has_phyloP100way'], value=False, key='has_phyloP100way')
            )
            if inputs['has_phyloP100way']:
                inputs['phyloP100way'] = st.number_input(
                    FEATURE_LABELS['phyloP100way'],
                    value=imputer_defaults.get('phyloP100way', 0.0),
                    min_value=-20.0, max_value=10.0, step=0.01, key='phyloP100way',
                )
            else:
                inputs['phyloP100way'] = np.nan
        ci += 1

    with score_cols[ci]:
        st.markdown("**Functional Score**")
        inputs['has_functional_score'] = int(
            st.checkbox(FEATURE_LABELS['has_functional_score'], value=False,
                        key='has_functional_score')
        )
        if inputs['has_functional_score']:
            inputs['functional_score'] = st.number_input(
                FEATURE_LABELS['functional_score'],
                value=imputer_defaults.get('functional_score', 0.0), key='functional_score',
            )
            inputs['functional_se'] = st.number_input(
                FEATURE_LABELS['functional_se'], value=0.0, min_value=0.0, key='functional_se',
            )
            inputs['functional_sd'] = st.number_input(
                FEATURE_LABELS['functional_sd'], value=0.0, min_value=0.0, key='functional_sd',
            )
        else:
            inputs['functional_score'] = np.nan
            inputs['functional_se'] = np.nan
            inputs['functional_sd'] = np.nan

    st.divider()
    exon_conflict = bool(inputs.get('is_exon_1_17') and inputs.get('is_exon18'))
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
        df = load_dataset()
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    variant_keys = df['variant_key'].tolist()

    def clean_key(k):
        return str(k).replace('_<NA>', '').replace('<NA>_', '').replace('<NA>', '').strip('_')

    def protein_change(k):
        p = str(k).split('_')
        if len(p) == 6 and p[3] != '<NA>' and p[4] != '<NA>' and p[5] != '<NA>':
            return f"p.{p[3]}{p[4]}{p[5]}"
        return ""

    display_keys   = [clean_key(k) for k in variant_keys]
    protein_keys   = [protein_change(k) for k in variant_keys]
    display_to_raw = {d: r for r, d in zip(reversed(variant_keys), reversed(display_keys))}

    # Error filter
    dt_wrong, rf_wrong, both_wrong = get_error_sets(selected_model)
    error_filter = st.sidebar.radio(
        "Filter by prediction",
        ["All variants", "DT misclassified", "RF misclassified", "Both misclassified"],
    )
    if error_filter == "DT misclassified":
        variant_keys = [k for k in variant_keys if k in dt_wrong]
    elif error_filter == "RF misclassified":
        variant_keys = [k for k in variant_keys if k in rf_wrong]
    elif error_filter == "Both misclassified":
        variant_keys = [k for k in variant_keys if k in both_wrong]
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
    if selected_types:
        mask = df[selected_types].eq(1).any(axis=1)
        allowed = set(df[mask]['variant_key'])
        variant_keys = [k for k in variant_keys if k in allowed]

    display_keys = [clean_key(k) for k in variant_keys]
    protein_keys = [protein_change(k) for k in variant_keys]

    st.sidebar.divider()
    search = st.sidebar.text_input("Search", placeholder="key, position or p.Met1Leu")
    if search:
        q = search.lower()
        filtered_display = [
            d for d, p in zip(display_keys, protein_keys)
            if q in d.lower() or q in p.lower()
        ]
    else:
        filtered_display = display_keys

    if not filtered_display:
        st.sidebar.warning("No variants match your search.")
        st.stop()

    selected_display = st.sidebar.selectbox(
        f"Variant key ({len(filtered_display)} shown)",
        filtered_display,
        index=0,
    )
    selected_key = display_to_raw[selected_display]

    # ── Main area ─────────────────────────────────────────────────────────────

    row = df[df['variant_key'] == selected_key].iloc[0]
    dt_pred, dt_proba, dt_path = predict_from_row(bundle, row)
    rf_pred, rf_proba, _       = predict_from_row(rf_bundle, row)

    # Protein change extracted from variant key: REF_ALT_POS_AAold_AApos_AAnew
    _parts = selected_key.split('_')
    _aa_old, _aa_pos, _aa_new = _parts[3], _parts[4], _parts[5]
    if _aa_old != '<NA>' and _aa_pos != '<NA>' and _aa_new != '<NA>':
        protein_change = f"p.{_aa_old}{_aa_pos}{_aa_new}"
    else:
        protein_change = None

    st.markdown(
        f"**Variant:** {clean_key(selected_key)}"
        + (f" &nbsp;·&nbsp; **{protein_change}**" if protein_change else ""),
        unsafe_allow_html=True,
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
        bg = "background:#dbeafe;" if dv == "Yes" and not feat.startswith('has_') else ""
        return (
            f"<tr style='border-bottom:1px solid #f0f0f0;'>"
            f"<td style='padding:6px 10px; color:#555; font-size:0.92rem;'>{label}</td>"
            f"<td style='padding:6px 10px; font-family:monospace; font-weight:600; font-size:0.92rem; {bg}'>{dv}{note}</td>"
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
