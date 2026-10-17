"""Code tables of the regional AMR-RETEMIC flow (Regione Siciliana, Allegati 1 e 2,
22/04/2026), used to show plain names instead of codes."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"

MATERIAL_LABELS = {
    "1": "Sangue", "2": "Liquor", "3": "Urine", "4": "Campione uretrale",
    "5": "Campione cervicale", "6": "Campione rettale", "7": "Campione faringeo", "8": "Altro",
}
ADMISSION_LABELS = {
    "1": "Ricovero ordinario", "2": "Day hospital", "3": "Non ricoverato - Pronto soccorso",
    "4": "Non ricoverato - Hospice/RSA", "5": "Non ricoverato - Altro",
}
ICU_WARD_CODES = {"49", "72", "73"}  # terapia intensiva, intensiva pediatrica, intensiva neonatale

ANTIBIOTIC_LABELS = {
    "AMK": "Amikacina", "AMC": "Amoxicillina-clavulanato", "AMP": "Ampicillina", "FEP": "Cefepime",
    "FDC": "Cefiderocol", "CTX": "Cefotaxime", "FOX": "Cefoxitina", "CPT": "Ceftarolina",
    "CAZ": "Ceftazidime", "CZA": "Ceftazidime-avibactam", "CZT": "Ceftolozano-tazobactam",
    "CRO": "Ceftriaxone", "CIP": "Ciprofloxacina", "CLI": "Clindamicina", "COL": "Colistina",
    "SXT": "Cotrimossazolo", "DAP": "Daptomicina", "DOR": "Doripenem", "ERY": "Eritromicina",
    "ETP": "Ertapenem", "FOS": "Fosfomicina", "GEN": "Gentamicina", "GEH": "Gentamicina (alto livello)",
    "IPM": "Imipenem", "IMR": "Imipenem-relebactam", "LVX": "Levofloxacina", "LNZ": "Linezolid",
    "MEM": "Meropenem", "MEV": "Meropenem-vaborbactam", "NIT": "Nitrofurantoina", "ORI": "Oritavancina",
    "OXA": "Oxacillina", "PEN": "Penicillina", "TZP": "Piperacillina-tazobactam", "RIF": "Rifampicina",
    "STH": "Streptomicina (alto livello)", "TEC": "Teicoplanina", "TCY": "Tetraciclina",
    "TGC": "Tigeciclina", "TOB": "Tobramicina", "VAN": "Vancomicina",
}
PATHOGEN_LABELS = {
    "ACISPP": "Acinetobacter spp.", "ENTFAE": "Enterococcus faecalis", "ENTFCM": "Enterococcus faecium",
    "ESCCOL": "Escherichia coli", "KLEPNE": "Klebsiella pneumoniae", "PSEAER": "Pseudomonas aeruginosa",
    "STAAUR": "Staphylococcus aureus", "STRPNE": "Streptococcus pneumoniae", "NEIGON": "Neisseria gonorrhoeae",
}


@lru_cache(maxsize=1)
def laboratory_names() -> dict[str, str]:
    path = RESOURCE_DIR / "laboratori_retemic.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["codice"]: row["ente"] for row in csv.DictReader(fh)}


def label(code: object, table: dict[str, str]) -> str:
    text = "" if code is None else str(code).strip()
    return table.get(text, "non indicato" if not text else f"codice non previsto ({text})")
