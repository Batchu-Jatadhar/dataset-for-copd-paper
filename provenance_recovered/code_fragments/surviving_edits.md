# Surviving line-level edits

Two `replace_file_content` calls were logged in full (they were small enough to escape truncation). These are exact, verbatim edits applied to the scripts after their initial creation.

## `adversarial_audit.py` — lines 386–386

Before:
```python
saturation_flag = (near_0 + near_1) > 90
```

After:
```python
saturation_flag = bool((near_0 + near_1) > 90)
```

## `phase_ablation.py` — lines 59–60

Before:
```python
def extract_patient_id(filename, dataset):
```

After:
```python
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def extract_patient_id(filename, dataset):
```

