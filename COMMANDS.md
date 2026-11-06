# Spustenie projektu

Tento subor je prakticka poznamka k prikazom. Prikazy su pre Windows
PowerShell z korenoveho priecinka projektu.

## 1. Prostredie

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ak treba CUDA verziu PyTorch, pouzita bola verzia z oficialneho PyTorch indexu:

```powershell
pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 2. Rychla kontrola projektu

```powershell
.\.venv\Scripts\python.exe main.py --commands
```

Tento prikaz nespusta dlhy trening. Iba skontroluje PC-GITA cache, pocty domen
a dostupne finalne vystupy.

Ak kolega nema PC-GITA data, `main.py` ukaze, ze lokalne data chybaju, ale
report artefakty v `reports/pcgita/` budu stale dostupne.

Finalne PNG grafy sa daju znovu vytvorit z malych CSV tabuliek bez raw dat:

```powershell
.\.venv\Scripts\python.exe make_report_plots.py
```

To je vhodny sanity check po cistom klone repozitara.

## 3. Prebudovanie audio priznakov

```powershell
.\.venv\Scripts\python.exe main.py --rebuild-features
```

Pouzit iba vtedy, ked sa zmeni PC-GITA dataset alebo cache chyba.

## 4. Finalne akusticke experimenty

Pairwise cross-domain porovnanie:

```powershell
.\.venv\Scripts\python.exe run_pcgita_experiments.py --methods logreg_source,svm_source,extra_trees,coral_linear,dann,mmd,deep_coral --seeds 42,7,123,2024,99 --epochs 20 --da-epochs 30 --pairs "words->sentences2,sentences2->words,vowels->ddk,ddk->vowels,sentences->read_text,read_text->sentences" --skip-multisource
```

Multi-source beh s rychlejsimi metodami:

```powershell
.\.venv\Scripts\python.exe run_pcgita_experiments.py --methods logreg_source,svm_source,extra_trees,coral_linear --seeds 42,7,123,2024,99 --epochs 20 --da-epochs 30
```

## 5. CNN nad log-mel spektrogramami

Sledovatelny beh s logom:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pcgita_spectrogram_watch.ps1
```

Vlastne parametre:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pcgita_spectrogram_watch.ps1 -Pairs "multi->words,multi->sentences2,multi->read_text,multi->monologue" -Methods "cnn_source,cnn_dann,cnn_mmd,cnn_coral" -Seeds "42,7" -Epochs 18 -BatchSize 32
```

## 6. Sledovanie priebehu

Progress subor:

```powershell
Get-Content data\pcgita_results\pcgita_spectrogram_progress.txt -Wait
```

Ciastocne vysledky:

```powershell
Import-Csv data\pcgita_results\pcgita_spectrogram_results_partial.csv | Format-Table
```

## 7. Subory vhodne do repozitara

Do repozitara patria hlavne:

```text
README.md
COMMANDS.md
PCGITA_EXPERIMENTS.md
reports/pcgita/
make_report_plots.py
src/
run_pcgita_experiments.py
run_pcgita_pretrained_experiments.py
run_pcgita_spectrogram_experiments.py
run_pcgita_spectrogram_watch.ps1
data/pcgita_pretrained_results/
```

Do repozitara nepatria raw data, cache, logy ani `data/torch_cache/`.

## 8. Pretrained wav2vec 2.0 experimenty

Extrakcia embeddingov a spustenie vsetkych experimentov:

```powershell
$env:PYTHONUTF8="1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
.venv\Scripts\python.exe run_pcgita_pretrained_experiments.py `
    --model facebook/wav2vec2-base --segment-seconds 4 --batch-size 16
```

Ak embeddingy uz su v cache (`data/pcgita_pretrained_*_patient_domain.csv`),
experimenty prebehnu bez GPU inferecie (reklasifikacia ~2 min).

Pre HuBERT staci zamenit `--model facebook/hubert-base-ls960`.

Znovuvytvorenie grafu po experimentoch:

```powershell
.venv\Scripts\python.exe make_report_plots.py
```
