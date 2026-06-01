FEATURE_COLS = [
    'aa_pos', 'is_missense', 'is_synonymous', 'is_frameshift', 'is_intron',
    'is_utr', 'has_AF_popmax', 'log10_AF_popmax', 'spliceai', 'has_spliceai',
    'is_multi_exon', 'exon_number', 'is_exon4', 'exon_symmetric', 'disrupts_frame',
    'is_exon_1_17', 'is_exon18', 'is_nonsense', 'nmd_aa_below_830', 'is_init_codon',
    'is_top_60_cys', 'splice_site', 'is_whole_exon', 'has_aa_pos', 'is_exonic',
    'codon_change_length', 'is_duplication', 'is_deletion', 'is_inframe_ins_or_del',
    'is_insertion', 'functional_score', 'has_functional_score', 'functional_se',
    'functional_sd',
]

TARGET_COL = 'VariantClassification'

FEATURE_LABELS = {
    'is_missense':           'Missense',
    'is_synonymous':         'Synonymous',
    'is_frameshift':         'Frameshift',
    'is_nonsense':           'Nonsense',
    'is_intron':             'Intronic',
    'is_utr':                'UTR',
    'is_exonic':             'Exonic',
    'is_duplication':        'Duplication',
    'is_deletion':           'Deletion',
    'is_insertion':          'Insertion',
    'is_inframe_ins_or_del': 'In-frame Ins/Del',
    'exon_number':           'Exon Number',
    'is_exon4':              'Exon 4',
    'is_exon_1_17':          'Exon 1–17',
    'is_exon18':             'Exon 18',
    'is_multi_exon':         'Multi-exon',
    'is_whole_exon':         'Whole Exon',
    'exon_symmetric':        'Exon Symmetric',
    'splice_site':           'Splice Site',
    'codon_change_length':   'Codon Change Length',
    'disrupts_frame':        'Disrupts Frame',
    'is_init_codon':         'Initiation Codon',
    'is_top_60_cys':         'Top-60 Cys Position',
    'nmd_aa_below_830':      'AA < 830',
    'aa_pos':                'AA Position',
    'has_aa_pos':            'AA Position known',
    'has_AF_popmax':         'Population AF known',
    'log10_AF_popmax':       'AF popmax',
    'spliceai':              'SpliceAI Score',
    'has_spliceai':          'SpliceAI available',
    'REVEL':                 'REVEL Score',
    'has_REVEL':             'REVEL available',
    'phyloP100way':          'phyloP100way Score',
    'has_phyloP100way':      'phyloP100way available',
    'functional_score':      'Functional Score',
    'has_functional_score':  'Functional score available',
    'functional_se':         'Functional Score SE',
    'functional_sd':         'Functional Score SD',
}

BINARY_FEATURES = {
    'is_missense', 'is_synonymous', 'is_frameshift', 'is_nonsense',
    'is_intron', 'is_utr', 'is_exonic', 'is_duplication', 'is_deletion',
    'is_insertion', 'is_inframe_ins_or_del', 'is_exon4', 'is_exon_1_17',
    'is_exon18', 'is_multi_exon', 'is_whole_exon', 'exon_symmetric',
    'splice_site', 'disrupts_frame', 'is_init_codon', 'is_top_60_cys',
    'nmd_aa_below_830', 'has_aa_pos', 'has_AF_popmax', 'has_spliceai',
    'has_REVEL', 'has_phyloP100way', 'has_functional_score',
}

# Ordered groups used for the UI form
FEATURE_GROUPS = {
    'Variant Type': [
        'is_missense', 'is_synonymous', 'is_frameshift', 'is_nonsense',
        'is_intron', 'is_utr', 'is_exonic',
        'is_duplication', 'is_deletion', 'is_insertion', 'is_inframe_ins_or_del',
    ],
    'Exon & Position': [
        'exon_number', 'is_exon4', 'is_exon_1_17', 'is_exon18',
        'is_multi_exon', 'is_whole_exon', 'exon_symmetric', 'splice_site',
        'codon_change_length', 'disrupts_frame',
        'is_init_codon', 'is_top_60_cys', 'nmd_aa_below_830',
        'has_aa_pos', 'aa_pos',
    ],
    'Allele Frequency': [
        'has_AF_popmax', 'log10_AF_popmax',
    ],
    'SpliceAI': [
        'has_spliceai', 'spliceai',
    ],
    'Functional Scores': [
        'has_functional_score', 'functional_score', 'functional_se', 'functional_sd',
    ],
}

FEATURE_DEFAULTS = {
    'is_missense': 0, 'is_synonymous': 0, 'is_frameshift': 0, 'is_nonsense': 0,
    'is_intron': 0, 'is_utr': 0, 'is_exonic': 1, 'is_duplication': 0,
    'is_deletion': 0, 'is_insertion': 0, 'is_inframe_ins_or_del': 0,
    'exon_number': 1, 'is_exon4': 0, 'is_exon_1_17': 1, 'is_exon18': 0,
    'is_multi_exon': 0, 'is_whole_exon': 0, 'exon_symmetric': 0,
    'splice_site': 0, 'codon_change_length': 0, 'disrupts_frame': 0,
    'is_init_codon': 0, 'is_top_60_cys': 0, 'nmd_aa_below_830': 0,
    'has_aa_pos': 0, 'aa_pos': 0.0,
    'has_AF_popmax': 0, 'log10_AF_popmax': 0.0,
    'has_spliceai': 0, 'spliceai': 0.0,
    'has_functional_score': 0, 'functional_score': 0.0,
    'functional_se': 0.0, 'functional_sd': 0.0,
}
