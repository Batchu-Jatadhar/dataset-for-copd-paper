import pandas as pd, re

prov = pd.read_csv(r'C:\Users\batch_dcahfiw\Downloads\datasetfinalforcopdemphysema\provenance_report.csv')
cx6  = prov[prov['original_dataset']=='ChestX6'].copy()

def cx6_patient_id(fname, cls):
    stem = fname.rsplit('.', 1)[0]
    if cls == 'Normal':
        m = re.search(r'\((\d+)\)', stem)
        return 'N_' + m.group(1) if m else 'N_' + stem
    else:
        parts = stem.split('_', 1)
        return 'E_' + parts[1] if len(parts) == 2 else 'E_' + stem

cx6['pid'] = cx6.apply(lambda r: cx6_patient_id(r['original_filename'], r['class_label']), axis=1)

norm_pids = set(cx6[cx6['class_label']=='Normal']['pid'])
emph_pids = set(cx6[cx6['class_label']=='Emphysema']['pid'])
print(f'Normal unique pids   : {len(norm_pids)}')
print(f'Emphysema unique pids: {len(emph_pids)}')
print(f'Cross-class pid overlap: {len(norm_pids & emph_pids)}')

norm_sample = list(cx6[cx6['class_label']=='Normal']['pid'].unique()[:8])
emph_sample = list(cx6[cx6['class_label']=='Emphysema']['pid'].unique()[:8])
print('Sample Normal pids   :', norm_sample)
print('Sample Emphysema pids:', emph_sample)
