# PC-GITA report artefakty

Tato zlozka obsahuje male subory vhodne do repozitara a prezentacie. Raw
PC-GITA audio, cache a dlhe logy ostavaju lokalne v `data/` a su ignorovane cez
`.gitignore`.

## Subory

- `01_pairwise_best.csv` / `01_pairwise_best_auc.png`: najlepsi model pre kazdy source -> target par.
- `02_multisource_best.csv` / `02_multisource_best_auc.png`: najlepsi multi-source model pre kazdu cielovu domenu.
- `03_spectrogram_cnn_best.csv` / `03_spectrogram_cnn_best_auc.png`: najlepsi CNN model nad log-mel spektrogramami.
- `04_pairwise_all_methods.csv` / `04_pairwise_all_methods_heatmap.png`: priame porovnanie vsetkych DA metod na rovnakych paroch.
- `05_domain_gap.csv` / `05_domain_gap_auc.png`: rozdiel medzi in-domain CV a out-of-domain multi-source testom.
- `06_cnn_vs_acoustic.csv` / `06_cnn_vs_acoustic_auc.png`: porovnanie CNN spektrogramov s tabulkovymi audio priznakmi.
- `07_method_distribution.png`: distribucia najlepsich DA metod napriec domenami a metrikami (stlpcove grafy pre kazdu metriku).
- `08_multisource_multi_metric.png`: multi-source vysledky cez AUC, F1, senzitivitu a specificitu pre vsetky cielove domeny.
- `09_wav2vec2_vs_acoustic.csv` / `09_wav2vec2_vs_acoustic_auc.png`: porovnanie wav2vec 2.0 frozen embeddingov (facebook/wav2vec2-base) s tabulkovymi akustickymi priznakmi v multi-source nastaveni.

PNG grafy sa daju znovu vytvorit prikazom:

```powershell
.\.venv\Scripts\python.exe make_report_plots.py
```
