"""
Tyzdne 1-2: Stiahnutie a priprava datasetov.

Vzor: Cvicenie 2 - Praca s datami.ipynb (stahovanie datasetov, transformacie)

Domény:
  Domain A - Oxford Parkinson's Dataset (Little et al., 2008)
    https://archive.ics.uci.edu/dataset/174/parkinsons
    195 vzoriek, 31 pacientov (23 PD + 8 Healthy)

  Domain B/C/D - Istanbul Replicated Acoustic Features (Naranjo et al., 2016)
    https://archive.ics.uci.edu/dataset/489
    240 vzoriek = 3 nahrávky × 80 subjektov (40 PD + 40 Healthy)
    Každá nahrávka = samostatná doména (B=R1, C=R2, D=R3)
    → Umožňuje multi-source domain adaptation experimenty

Spoločných 12 akustických príznakov (Jitter× 4, Shimmer × 4, HNR, RPDE, DFA, PPE)
Istanbul má navyše: MFCC 0-12, Delta 0-12, GNE, HNR v 5 pásmach → 47 príznakov celkom
"""

import io
import os
import urllib.request
import zipfile

import pandas as pd

from src.config import DATA_DIR

# ---- Mapovanie stĺpcov na spoločné názvy ----

# Dataset 174: Oxford Parkinson's
OXFORD_MAP = {
    'MDVP:Jitter(%)': 'jitter_rel',
    'MDVP:Jitter(Abs)': 'jitter_abs',
    'MDVP:RAP': 'jitter_rap',
    'MDVP:PPQ': 'jitter_ppq',
    'MDVP:Shimmer': 'shimmer',
    'MDVP:Shimmer(dB)': 'shimmer_db',
    'Shimmer:APQ3': 'shimmer_apq3',
    'Shimmer:APQ5': 'shimmer_apq5',
    'HNR': 'hnr',
    'RPDE': 'rpde',
    'DFA': 'dfa',
    'PPE': 'ppe',
}

# Dataset 489: Istanbul - 12 spoločných príznakov kompatibilných s Oxford
# Pozn.: jitter_rel má rozdielne merítko (rozdiel v protokole merania = zámerný domain shift)
# HNR05 = HNR v pásme 0-500 Hz ≠ MDVP:HNR (broadband) → ďalší zdroj doménového posunu
ISTANBUL_MAP = {
    'Jitter_rel': 'jitter_rel',
    'Jitter_abs': 'jitter_abs',
    'Jitter_RAP': 'jitter_rap',
    'Jitter_PPQ': 'jitter_ppq',
    'Shim_loc': 'shimmer',
    'Shim_dB': 'shimmer_db',
    'Shim_APQ3': 'shimmer_apq3',
    'Shim_APQ5': 'shimmer_apq5',
    'HNR05': 'hnr',       # HNR v pasme 0-500 Hz (najblizsi k HNR z Oxford)
    'RPDE': 'rpde',
    'DFA': 'dfa',
    'PPE': 'ppe',
}


def download_oxford():
    """
    Stiahne Oxford Parkinson's Dataset (UCI #174).
    Priamy download CSV - najspolahlivejsia metoda.
    https://archive.ics.uci.edu/dataset/174/parkinsons
    """
    print("[Domain A] Oxford Parkinson's Dataset (Little et al., 2008)")

    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/parkinsons/parkinsons.data")
    df = pd.read_csv(url)

    # Oddelenie labelov
    labels = df['status'].values

    # Vyberieme a premenujeme spolocne priznaky
    common = list(OXFORD_MAP.values())
    df_out = df.rename(columns=OXFORD_MAP)[common].copy()
    df_out['label'] = labels
    df_out['domain'] = 'oxford'

    # Extract subject ID: phon_R01_S01_3 → 'S01' (32 unique subjects, 6-7 recordings each)
    df_out['patient_id'] = df['name'].str.extract(r'_(S\d+)_')[0]

    print(f"  Pocet vzoriek: {len(df_out)}")
    print(f"  Pocet pacientov: {df_out['patient_id'].nunique()}")
    print(f"  PD: {(df_out['label']==1).sum()}, "
          f"Healthy: {(df_out['label']==0).sum()}")
    return df_out


# Istanbul: rozšírené príznaky ktoré nie sú v Oxford
ISTANBUL_EXTRA_COLS = [
    'Shi_APQ11', 'HNR15', 'HNR25', 'HNR35', 'HNR38', 'GNE',
    'MFCC0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6',
    'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
    'Delta0', 'Delta1', 'Delta2', 'Delta3', 'Delta4', 'Delta5', 'Delta6',
    'Delta7', 'Delta8', 'Delta9', 'Delta10', 'Delta11', 'Delta12',
]


def download_istanbul():
    """
    Stiahne Istanbul Replicated Acoustic Features (UCI #489).
    Rozdelí podľa recording session → 3 samostatné domény B, C, D.

    Štruktúra: 80 pacientov × 3 nahrávky = 240 vzoriek
    Session 1 (R1) → Domain B: 80 vzoriek (40 PD + 40 Healthy)
    Session 2 (R2) → Domain C: 80 vzoriek (40 PD + 40 Healthy)
    Session 3 (R3) → Domain D: 80 vzoriek (40 PD + 40 Healthy)
    """
    print("\n[Domain B/C/D] Istanbul Replicated Features (Naranjo et al., 2016)")
    print("  3 recording sessions → 3 separate domains")

    zip_url = ("https://archive.ics.uci.edu/static/public/489/"
               "parkinson+dataset+with+replicated+acoustic+features.zip")

    resp = urllib.request.urlopen(zip_url)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))

    csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
    with z.open(csv_name) as csv_file:
        df_raw = pd.read_csv(csv_file)

    # Spoločných 12 príznakov (kompatibilné s Oxford)
    common_cols = list(ISTANBUL_MAP.values())
    df_base = df_raw.rename(columns=ISTANBUL_MAP)[common_cols].copy()
    df_base['label'] = df_raw['Status'].values
    df_base['patient_id'] = df_raw['ID'].values
    df_base['recording'] = df_raw['Recording'].values

    # Rozšírené príznaky dostupné len v Istanbul (MFCC, Delta, multi-HNR, GNE)
    extra_cols = [c for c in ISTANBUL_EXTRA_COLS if c in df_raw.columns]
    for col in extra_cols:
        df_base[f'ext_{col.lower()}'] = df_raw[col].values

    print(f"  Celkovo vzoriek: {len(df_base)}")
    print(f"  Pocet pacientov: {df_base['patient_id'].nunique()}")
    print(f"  Rozsirene priznaky: {len(extra_cols)} (MFCC, Delta, GNE, multi-HNR)")

    results = {}
    domain_names = {1: 'istanbul_r1', 2: 'istanbul_r2', 3: 'istanbul_r3'}
    for rec, name in domain_names.items():
        df_rec = df_base[df_base['recording'] == rec].copy()
        df_rec['domain'] = name
        results[name] = df_rec
        print(f"  {name}: {len(df_rec)} vzoriek "
              f"(PD={( df_rec['label']==1).sum()}, "
              f"Healthy={(df_rec['label']==0).sum()})")

    # Aj celý Istanbul (pre spätnu kompatibilitu)
    df_all = df_base.copy()
    df_all['domain'] = 'istanbul'
    results['istanbul'] = df_all

    return results


def download_datasets():
    """Stiahne všetky datasety a uloží do CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print("Stahovanie datasetov z UCI ML Repository")
    print("=" * 60)

    df_oxford = download_oxford()
    istanbul_dict = download_istanbul()

    # Oxford
    df_oxford.to_csv(os.path.join(DATA_DIR, 'oxford.csv'), index=False)

    # Istanbul: celý aj jednotlivé sessions
    for name, df in istanbul_dict.items():
        df.to_csv(os.path.join(DATA_DIR, f'{name}.csv'), index=False)

    print(f"\nData ulozene do: {DATA_DIR}/")
    print(f"  oxford.csv        ({len(df_oxford)} vzoriek)")
    for name, df in istanbul_dict.items():
        print(f"  {name}.csv  ({len(df)} vzoriek)")

    return df_oxford, istanbul_dict


if __name__ == '__main__':
    download_datasets()
