from dataloader.interface import load_dataset  # noqa: F401
from dataloader.mdace.constants import MDACE_INPATIENT_PATH as mdace_inpatient
from dataloader.meddec.constants import MEDDEC_PATH as meddec
from dataloader.mimiciii.constants import MIMIC_III_50_PATH as mimiciii_50
from dataloader.mimiciv.constants import MIMIC_IV_50_PATH as mimiciv_50
from dataloader.mimiciv.constants import MIMIC_IV_PATH as mimiciv
from dataloader.snomed.constants import SNOMED_PATH as snomed

_MIMICIV_CM_VERSIONS = ("3.0", "3.1", "3.2", "3.3", "3.4")


def _mimiciv_cm_config(version: str) -> dict:
    return {
        "identifier": f"mimiciv-cm-{version}",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10"],
        "options": {"subset_size": 1000, "adapter": "MimicIdentifyAdapter"},
    }


_MIMICIV_CM_CONFIGS = {
    f"mimiciv-cm-{version}": _mimiciv_cm_config(version)
    for version in _MIMICIV_CM_VERSIONS
}

DATASET_CONFIGS: dict[str, dict] = {
    "debug": {
        "identifier": "debug",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.0"],
        "split": "test",
        "options": {"subset_size": 10},
    },
    "meddec": {"identifier": "meddec", "name_or_path": meddec, "split": "validation"},
    "snomed": {"identifier": "snomed", "name_or_path": snomed, "split": "validation"},
    "mdace-icd10cm": {
        "identifier": "mdace-icd10cm",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm"],
        "options": {"adapter": "MdaceAdapter"},
    },
    "mimic-iv": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10"],
        "options": {"subset_size": 300, "adapter": "MimicIdentifyAdapter"},
    },
    **_MIMICIV_CM_CONFIGS,
    "mimic-iii-50": {
        "identifier": "mimic-iii-50",
        "name_or_path": mimiciii_50,
        "split": "test",
        "options": {"order": "alphabetical"},
    },
    "mimic-iv-50": {
        "identifier": "mimic-iv-50",
        "name_or_path": mimiciv_50,
        "split": "test",
        "subsets": ["icd10"],
        "options": {
            "order": "alphabetical",
        },
    },
}
