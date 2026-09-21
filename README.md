# Machine-Learning Modelling of Molecular and Environmental Data

This repository contains the analysis code for a population-genomic and
environmental-adaptation study. A single SNP dataset answers two linked
questions: (1) **Population origin assignment** — which province-level region does
an individual of unknown provenance come from? (2) **Environmental adaptation
prediction** — how do populations respond to climate, and how large is their
genetic offset under future climate?

Dataset: 180 individuals from 18 sampling populations (10 each), merged into six
province-level regions (30 each); 17,728 loci. Loci are not MAF-filtered; every
variant row in the VCF is retained.

## Layout

- `01_raw_data/` holds raw SNP data and sample coordinates
- `02_intermediate/` holds the unimputed dosage matrix, labels, and PLINK bfile
- `03_r_env/` holds the R package library, rebuilt by `install_r_pkgs.R`
- `04_population_assignment/` holds the main pipeline, diagnostics, and report builders
- `05_results/` holds grid scores, frozen panel, per-fold ranking cache, missingness and FST diagnostics
- `07_habitat_suitability/` holds occurrence filtering, WorldClim variables, and MaxEnt modelling
- `08_genetic_offset/` holds LFMM2 / RDA / GEA intersection, gradient forest and genetic offset

## Machine-Learning Modelling

Full-pipeline quality control and feature-selection engineering for molecular data
from 180 samples and 17,728 loci.

## Environmental Adaptation Prediction

Using 18 populations as samples, 6-dimensional environmental features as inputs, and
424 continuous allele frequencies as outputs, a multi-output Random Forest ensemble
regression was constructed (500 trees per output). Feature importance was aggregated
by R² weighting, response curves and response thresholds were extracted, and current
and future feature values were mapped into response space through the same set of
response curves, with Euclidean distance used to quantify the genetic offset of each
population.

## Population Origin Assignment

Using 6 province-level regions as class labels, a nested cross-validation benchmark
was constructed (outer 10-fold, inner 5-fold for hyperparameter tuning), combining
three feature-selection methods — mRMR, Relief-F, and FST — with four classifiers —
SVM, Random Forest, KNN, and Naive Bayes — and evaluated one by one across six
feature-count levels from 100 to 2000. The best combination, Relief-F + SVM, achieved
out-of-fold ACC 1.000 and AUC 1.000 at 500 features (Clopper-Pearson 95% CI
[0.980, 1.000]).

## Protocol

`04_population_assignment/pipeline.py` is a single self-contained file (~800 lines).
Run it from the package root:

```bash
python3 04_population_assignment/pipeline.py
```

The protocol is nested cross-validation. The outer
`StratifiedKFold(10, random_state=0)` holds out 18 test individuals (3 per
province); the inner 5-fold tunes classifier hyperparameters only. Feature ranking
runs once per outer fold on that fold's 162 training individuals. The grid is
3 feature selectors × 6 locus counts × 4 classifiers = 72 cells:

- selectors: mRMR, Relief-F, FST
- locus counts: 100 / 250 / 500 / 800 / 1000 / 2000
- classifiers: SVM, Random Forest, KNN, Naive Bayes

Missing genotypes are imputed with the within-fold mode, computed from that fold's
162 training individuals only; the test fold reuses the same modes. The matrix is
neither standardised nor dimension-reduced. Scores land in
`05_results/grid_results.csv`. A 150-locus panel built from the intersection of the
top-500 loci across all ten folds is frozen in `05_results/frozen_panel.csv`.

## Habitat Suitability

Three scripts under `07_habitat_suitability/03_model/`, run with the chapter's own
virtual environment:

```bash
bash 07_habitat_suitability/05_env/download_worldclim.sh
07_habitat_suitability/05_env/.venv/bin/python 07_habitat_suitability/03_model/variable_screen.py
07_habitat_suitability/05_env/.venv/bin/python 07_habitat_suitability/03_model/maxent_run.py
07_habitat_suitability/05_env/.venv/bin/python 07_habitat_suitability/03_model/variable_compare.py
```

MaxEnt is implemented with `elapid`. Variable screening iterates removal over a
Spearman rank-correlation matrix using VIF until every value is at or below the
threshold (default 10), which cuts 19 variables to 7. Rasters are sampled
nearest-neighbour, with no bilinear interpolation.

## Genetic Offset

Eleven scripts under `08_genetic_offset/03_model/`, run in numeric order. R scripts
need the in-package library on `R_LIBS_USER`:

```bash
R_LIBS_USER=03_r_env/R_libs Rscript 08_genetic_offset/03_model/02_prep_geno.R
```

The chain is: LFMM2 (19 bioclim variables, K chosen as 4) and RDA (3 SD outlier
loadings) each propose candidates; their intersection gives the GEA set.
`gradientForest` then fits the relationship between environment and allele
frequency, and current and future environmental values are pushed through the
response curves into a common space, where Euclidean distance defines each
population's genetic offset. Gradient forest is fitted at two levels: population
(18 populations × allele frequency, the main model) and individual (180 individuals
× dosage, a consistency check). Environment is screened again in chapter 08, taking
the same VIF procedure from chapter 07's 7 variables down to 6.

## Inputs You Must Supply

Point the scripts at these, using the default relative paths:

- `01_raw_data/SAMPLE.csv` — coordinates of the 18 populations, columns `NAME,X,Y`
- `02_intermediate/dose_unfilled.csv` — first column `sample`, one column per locus, values `{0,1,2,NA}`
- `02_intermediate/labels.csv` — columns `sample,pop,province,class`
- `02_intermediate/snp.{bed,bim,fam}` — PLINK bfile for the same loci, used by the FST path
- `07_habitat_suitability/02_env_vars/crop/` — WorldClim 2.1 rasters cropped to the study region

The VCF-to-dosage-matrix conversion is not included; the reproducible chain starts
at `02_intermediate/`.

## Environment

**Python (chapter 04)** — dependencies in
`04_population_assignment/requirements.txt`. PLINK 1.9 must be on `PATH`. The
Relief-F path additionally needs R and CORElearn, and must be started as
`Rscript --max-ppsize=500000`: a formula with 17,728 terms overflows R's default
pointer-protection stack.

**R (chapter 08)** — the library installs inside the package rather than touching
the system library:

```bash
R_LIBS_USER=03_r_env/R_libs Rscript 08_genetic_offset/05_env/install_r_pkgs.R
```

This installs LEA and qvalue (Bioconductor) plus vegan, psych and data.table (CRAN).

**Python (chapter 07)** — rebuild the `.venv` from the package set recorded in
`07_habitat_suitability/05_env/install.log` (elapid 1.0.4, rasterio 1.5.1,
geopandas 1.1.4, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6 and so on). The
plotting and raster scripts in chapter 08 (`08_offset_map.py`, `10_offset_figs.py`,
`11_pop_fate.py`) reuse the same interpreter.
