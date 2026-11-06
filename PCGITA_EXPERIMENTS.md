# PC-GITA experimenty: audio príznaky a domain adaptation

Tento dokument sumarizuje vetvu projektu založenú na PC-GITA audio dátach.
Experimentálne domény sú rečové úlohy v rámci PC-GITA, nie samostatné
databázy.

## Dáta

Použitá lokálna cesta:

```text
data/PC-GITA/PC-GITA_per_task_44100Hz/
```

Metadata obsahujú 100 oficiálnych subjektov:

- 50 pacientov s Parkinsonovou chorobou,
- 50 zdravých kontrol,
- klinické údaje pre PD skupinu,
- vek a pohlavie pre všetkých subjektov.

V lokálnej kópii existovali aj extra ID bez metadát. Tie sa zámerne
nepoužívajú. Po filtrácii podľa metadát ostalo 4745 wav súborov.

Domény sú definované podľa rečovej úlohy:

- `vowels`
- `ddk`
- `words`
- `sentences`
- `sentences2`
- `read_text`
- `monologue`

Každá finálna vzorka v modelovaní je jeden pacient v jednej doméne.
Po agregácii vzniká tabuľka 700 riadkov:

```text
7 domén × 100 pacientov = 700 pacient-doména vzoriek
```

Poznámka k definícii domény: zadanie uvádza viacero verejných databáz ako
prirodzený príklad domén. Keďže pre projekt bol dodatočne sprístupnený najmä
PC-GITA, hlavná verzia definuje domény podľa rečových úloh v rámci PC-GITA.

Toto je užšie než plná interdatabázová generalizácia zo zadania. Výsledky preto
interpretujeme ako kontrolovaný cross-task benchmark: držíme rovnaký korpus,
jazyk, mikrofónový protokol a extrakčný pipeline, aby sme najprv videli, či
adaptácia pomáha pri samotnom posune rečovej úlohy. Plný cross-database
experiment by mal rovnaký patient-wise a unsupervised-target protokol, ale
doménami by boli samostatné korpusy.

## Príznaky

Príznaky sa extrahujú priamo z wav audia v `src/data/pcgita.py`.
Použitých je 47 akustických príznakov:

- MFCC priemery a smerodajné odchýlky,
- RMS energia,
- zero-crossing rate,
- spektrálny centroid, bandwidth, rolloff, flatness, entropy,
- spektrálny flux,
- dominantná frekvencia v hlasovom pásme,
- trvanie, voiced ratio a silence ratio.

Súbory cache:

```text
data/pcgita_features_file.csv
data/pcgita_features_patient_domain.csv
```

## Evaluačné nastavenie

Split je striktne patient-wise:

- 60 pacientov: labeled source + unlabeled target pre adaptáciu,
- 40 pacientov: target test,
- pacient z testu sa nevyskytuje v source tréningu ani v target-unlabeled časti.

Target labely sa pri domain adaptation nepoužívajú. Používajú sa iba pri
výpočte metrík.

## Metódy

Porovnané metódy:

- `logreg_source` - source-only logistická regresia,
- `svm_source` - source-only RBF SVM,
- `extra_trees` - silný klasický source-only baseline,
- `coral_linear` - uzavretá CORAL adaptácia kovariancií,
- `dann` - Domain-Adversarial Neural Network,
- `mmd` - moment matching cez MMD,
- `deep_coral` - neural CORAL loss.

## Hlavné výsledky

### In-domain 5-fold patient-wise CV

| Doména | AUC mean |
|---|---:|
| monologue | 0.920 |
| words | 0.878 |
| sentences2 | 0.862 |
| sentences | 0.858 |
| read_text | 0.838 |
| vowels | 0.794 |
| ddk | 0.768 |

### Cross-domain pairwise, 5 seeds

| Source -> Target | Najlepšia metóda | AUC mean | F1 mean |
|---|---|---:|---:|
| sentences2 -> words | coral_linear | 0.870 | 0.762 |
| sentences -> read_text | extra_trees | 0.803 | 0.812 |
| read_text -> sentences | extra_trees | 0.779 | 0.654 |
| words -> sentences2 | extra_trees | 0.776 | 0.769 |
| vowels -> ddk | logreg_source | 0.739 | 0.708 |
| ddk -> vowels | extra_trees | 0.702 | 0.600 |

### Multi-source, 5 seeds

| Target | Najlepšia metóda | AUC mean | F1 mean |
|---|---|---:|---:|
| monologue | extra_trees | 0.900 | 0.830 |
| sentences | extra_trees | 0.850 | 0.805 |
| sentences2 | extra_trees | 0.816 | 0.814 |
| read_text | extra_trees | 0.782 | 0.781 |
| words | extra_trees | 0.775 | 0.624 |
| ddk | logreg_source | 0.734 | 0.696 |
| vowels | extra_trees | 0.690 | 0.619 |

### CNN nad log-mel spektrogramami, 2 seeds

Okrem finálnej tabuľkovej vetvy je v projekte otestovaná aj CNN vetva nad
log-mel spektrogramami. Cache spektrogramov obsahuje všetkých 4745 oficiálnych
PC-GITA wav súborov a je uložená v:

```text
data/pcgita_logmel_4s_64mels.pt
```

Multi-source CNN experiment prebehol na CUDA s metódami `cnn_source`,
`cnn_dann`, `cnn_mmd` a `cnn_coral`.

| Target | Najlepšia CNN metóda | AUC mean | F1 mean |
|---|---|---:|---:|
| sentences2 | cnn_dann | 0.817 | 0.765 |
| monologue | cnn_dann | 0.683 | 0.621 |
| words | cnn_source | 0.656 | 0.730 |
| read_text | cnn_dann | 0.650 | 0.339 |

CNN výsledky sú užitočné pre splnenie a porovnanie klasifikátora nad
spektrogramami, ale vo väčšine domén nedosahujú kvalitu agregovaných
akustických príznakov. Výnimkou je `sentences2`, kde `cnn_dann` dosahuje AUC
0.817 a je porovnateľný s tabuľkovým multi-source výsledkom.

## Interpretácia

PC-GITA poskytuje kontrolovaný testbed, v ktorom používame jednotnú extrakciu
z rovnakého typu vstupu: audio signálu. Vďaka tomu možno priamo porovnať DA
metódy bez zmeny jazyka, nahrávacieho protokolu alebo extractor pipeline.

Najdôležitejší vedecký výsledok nie je, že každá domain adaptation metóda
automaticky zlepší baseline. Naopak:

- silné klasické modely nad kvalitnými audio príznakmi sú veľmi konkurencieschopné,
- hlboké DA metódy sú pri 60 source pacientoch a 60 target-unlabeled pacientoch
  nestabilné,
- moment matching metódy sú užitočné v niektorých pároch, najmä
  `sentences2 -> words`, kde `coral_linear` dosiahol AUC 0.870,
- multi-source učenie je silné pre connected speech domény
  (`monologue`, `sentences`, `sentences2`),
- domény `vowels` a `ddk` sú ťažšie a vykazujú väčší domain gap.

Pre obhajobu je najčistejší záver:

> Najlepšiu generalizáciu v našom nastavení poskytuje jednotná extrakcia
> audio príznakov z PC-GITA, patient-wise evaluácia a multi-source tréning
> s klasickým robustným klasifikátorom. Domain adaptation metódy treba
> prezentovať porovnávaco a kriticky: pomáhajú len v niektorých doménových
> pároch, nie univerzálne.

Pre malé klinické datasety je z praktického hľadiska najvhodnejší ExtraTrees
nad ručne navrhnutými akustickými príznakmi v multi-source režime. Linear CORAL
je jednoduchý a stabilný doplnok pre blízke domény, kým DANN/CDAN a CNN by mali
byť ďalší krok skôr nad pretrained speech embeddingmi než nad malým datasetom
trénovaným od nuly.

## Výsledky: predtrénované wav2vec 2.0 embeddingy

Implementácia (model `facebook/wav2vec2-base`, 768-dim skrytý stav, 1536-dim
patient-domain embedding po mean+std pooling):

- `src/data/pcgita_pretrained.py` — extrakcia embeddingov (4 s chunky, GPU inferencia,
  CSV cache `data/pcgita_pretrained_<slug>_patient_domain.csv`),
- `run_pcgita_pretrained_experiments.py` — importuje `run_in_domain_cv`,
  `run_cross_domain` a `run_multisource` z hlavného experimentálneho skriptu.

Výsledky sú uložené v `data/pcgita_pretrained_results/facebook__wav2vec2-base/`
a vizualizácia je v `reports/pcgita/09_wav2vec2_vs_acoustic_auc.png`.

### In-domain CV (wav2vec2 LogReg vs. tabular best)

| Doména | wav2vec2 AUC | Tabular AUC | Δ |
|---|---:|---:|---:|
| read_text | 0.866 | 0.838 | +0.028 |
| sentences | 0.866 | 0.858 | +0.008 |
| ddk | 0.798 | 0.768 | +0.030 |
| sentences2 | 0.838 | 0.862 | -0.024 |
| monologue | 0.856 | 0.920 | -0.064 |
| words | 0.758 | 0.878 | -0.120 |
| vowels | 0.596 | 0.794 | -0.198 |

### Multi-source (best method per target)

| Target | wav2vec2 metóda | wav2vec2 AUC | Tabular AUC | Δ |
|---|---|---:|---:|---:|
| sentences2 | extra_trees | 0.852 | 0.816 | +0.036 |
| read_text | deep_coral | 0.813 | 0.782 | +0.031 |
| vowels | logreg_source | 0.701 | 0.690 | +0.011 |
| ddk | extra_trees | 0.738 | 0.734 | ≈0 |
| sentences | deep_coral | 0.824 | 0.850 | -0.026 |
| words | coral_linear | 0.652 | 0.775 | -0.123 |

**Záver:** wav2vec 2.0 embeddingy prekonávajú tabular príznaky pre connected
speech domény (`sentences2`, `read_text`) v multi-source nastavení. Pre
izoláté (`words`, `vowels`) sú ručne extrahované príznaky stále silnejšie.
Modelu chýba jemné dolaďovanie (fine-tuning) na španielskej patologickej
reči, čo limituje výkon pre akusticky vzdialené domény.

## Spustenie

Rýchly pairwise experiment:

```bash
python run_pcgita_experiments.py --methods logreg_source,svm_source,extra_trees,coral_linear,dann,mmd,deep_coral --seeds 42,7,123 --epochs 20 --da-epochs 30 --pairs "words->sentences2,sentences2->words,vowels->ddk,ddk->vowels,sentences->read_text,read_text->sentences" --skip-multisource
```

Výstupy:

```text
data/pcgita_results/final_pairwise_summary.csv
data/pcgita_results/final_pairwise_best.csv
data/pcgita_results/final_multisource_best.csv
data/pcgita_results/final_pairwise_best_auc.svg
data/pcgita_results/final_multisource_best_auc.svg
data/pcgita_results/pcgita_spectrogram_best.csv
data/pcgita_results/pcgita_spectrogram_best_auc.svg
```

Pre repozitár a prezentáciu sú malé kópie finálnych tabuliek a grafov uložené
v `reports/pcgita/`. Raw audio, cache a dlhé logy ostávajú iba lokálne.
