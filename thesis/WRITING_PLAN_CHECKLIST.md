# Thesis Writing Plan Checklist

Purpose: a concrete, chapter-by-chapter plan of what must be added to the thesis so it matches:
- EU Bratislava IS 1/2024 rules (`thesis/Example and rules/2024_is_1_2024_zaverecne_rigorozne_a_habilitacne_prace.pdf`)
- structure style from sample thesis (`thesis/Example and rules/72753b1ed3504947b2d962dfc4e56085.pdf`)
- real content and artifacts from this TSP project.

## 1. Mandatory compliance checklist (IS 1/2024)

- [ ] Language: thesis text in Slovak (state language).
- [ ] Writing style: first person plural, past tense (author plural).
- [ ] Abstract SK: one paragraph, typically 100-250 words.
- [ ] Abstract EN: translation of Slovak abstract.
- [ ] Main body contains: `Uvod`, core chapters, `Zaver`, `Zoznam pouzitej literatury`, optional `Prilohy`.
- [ ] Recommended chapter logic in body:
  - [ ] current state of topic
  - [ ] goal of thesis
  - [ ] methodology and methods
  - [ ] results
  - [ ] discussion
- [ ] Allowed for diploma thesis: merge `Ciel` + `Metodika`, and merge `Vysledky` + `Diskusia`.
- [ ] Formal setup target (already mostly in template): Times New Roman 12, line spacing 1.5, block alignment, first line indent 1.25 cm.
- [ ] Margins target: left/inner 3.5 cm, right/outer 2.0 cm, top/bottom 2.5 cm, foot 1.25 cm.
- [ ] Two-sided printing from `Uvod`; page numbers displayed from `Uvod` (counting starts from title page).
- [ ] Numbering continues up to first appendix page.
- [ ] Recommended diploma size target: about 50-70 pages (without appendices).

## 2. What the sample thesis confirms (structure pattern)

The sample thesis follows this high-level sequence:
- [ ] `Uvod` (continuous text)
- [ ] `1 Sucasny stav problematiky doma a v zahranici`
- [ ] `2 Ciel prace`
- [ ] `3 Metodika prace a metody skumania`
- [ ] `4 Vysledky prace`
- [ ] `Zaver`
- [ ] `Zoznam pouzitej literatury`
- [ ] `Prilohy`

In this project template, you already use the merged variant:
- `thesis/chapters/03_ciel_metodika.tex`
- `thesis/chapters/04_vysledky_diskusia.tex`

This is fully consistent with IS 1/2024.

## 3. Chapter-by-chapter writing plan from project artifacts

## 3.1 `thesis/chapters/01_uvod.tex` (continuous text, no 1.1/1.2)

- [ ] Explain why STSP is important in optimization/logistics and ML-for-OR context.
- [ ] Explain practical motivation of the thesis topic from AIS assignment.
- [ ] Define the thesis scope:
  - STSP only
  - edge-based neural models
  - full pipeline from data generation to evaluation on TSPLIB
- [ ] Briefly describe thesis contribution (without deep technical detail yet):
  - unified CLI workflow
  - comparable model families
  - unified evaluation protocol and post-analysis
- [ ] Briefly describe chapter roadmap.
- [ ] Keep this chapter without citations if you prefer; put literature support into chapter 2.

## 3.2 `thesis/chapters/02_sucasny_stav.tex`

- [ ] Add theoretical base:
  - formal STSP definition
  - computational complexity (NP-hard)
  - exact vs heuristic vs learning-based approaches
- [ ] Add state of AI/NN methods for TSP:
  - edge scoring / graph-based representations
  - decoding and local search post-processing
- [ ] Explain evaluation convention (optimality gap in percent).
- [ ] Add critical synthesis: what is known, what remains open, why your approach is relevant.
- [ ] This chapter must contain core citations (main literature chapter).

## 3.3 `thesis/chapters/03_ciel_metodika.tex`

- [ ] Convert assignment goal into one precise measurable main goal.
- [ ] Add 3-5 partial goals that directly map to project outputs.
- [ ] Describe full experiment pipeline and reproducibility:
  - CLI orchestration: `src/tspgnn/cli.py`
  - config validation: `src/tspgnn/config.py`
- [ ] Describe data sources:
  - synthetic data generation: `src/tspgnn/data/generate.py`
  - TSPLIB processing: `src/tspgnn/data/tsplib.py`
  - exact/near-exact label sources (Concorde, Elkai, NN+2opt)
- [ ] Describe generated data scope using actual counts:
  - `runs/data/synthetic` and `runs/data/synthetic_concorde_v1`
  - N in {20,30,40,50,60,70,80,90,100}
  - train 45k, val 9k, test 9k per dataset family
  - TSPLIB processed instances: 14 (`runs/data/tsplib/processed`)
- [ ] Describe quality checks:
  - QA flow in `src/tspgnn/eval/evaluate.py` (`run_qa`)
  - reports: `runs/qa_report.csv`, `runs/qa_report_concorde.csv`
  - mention that stored QA reports show 0 invalid labels/length mismatches
- [ ] Describe feature engineering:
  - complete graph edges + 10D edge features from `src/tspgnn/utils/geom.py`
- [ ] Describe compared model families:
  - `edge_mlp`: `src/tspgnn/models/edge_mlp.py`
  - `edge_res_mlp`: `src/tspgnn/models/edge_res_mlp.py`
  - `edge_transformer`: `src/tspgnn/models/edge_transformer.py`
- [ ] Describe training protocol:
  - BCE with class balancing, early stopping, LR plateau scheduler
  - `src/tspgnn/training/train.py`
- [ ] Describe decoding and evaluation:
  - greedy cycle + optional 2-opt + multistart/noise variants
  - `src/tspgnn/utils/tour.py`, `src/tspgnn/eval/evaluate.py`
- [ ] Describe post-analysis flow:
  - `src/tspgnn/analysis/report.py`
  - outputs in `runs/analysis/*.csv`

## 3.4 `thesis/chapters/04_vysledky_diskusia.tex`

- [ ] Start with experiment matrix overview (what was compared):
  - config set in `configs/exp_*.yaml`
  - 18 experiment IDs under `runs/experiments`
- [ ] Add table: model/config comparison matrix
  - source: `configs/exp_*.yaml`
  - include model type, hidden size, depth, edge feature mode, training dataset family
- [ ] Add table: global ranking by instance-wise performance
  - source: `runs/analysis/model_ranking.csv`
- [ ] Add table/figure: mean gap summary
  - source: `runs/analysis/eval_summary_long.csv`
  - compare MLP / ResMLP / Transformer groups
- [ ] Add table: baseline vs optimized decode profile for transformer runs
  - source: `runs/analysis/eval_profile_compare.csv`
  - explicitly discuss delta in gap
- [ ] Add table: hard/easy TSPLIB instances and spread across models
  - source: `runs/analysis/instance_gap_matrix.csv`
- [ ] Add short training dynamics analysis (epochs, best val epoch, loss delta, runtime)
  - source: `runs/analysis/experiments_summary.csv`
- [ ] Add visual examples of tours (GT vs prediction):
  - source PNGs in `runs/experiments/<exp_id>/<run>/figs/tsplib`
  - include both strong and weak cases
- [ ] In discussion part:
  - explain why transformer variants rank better in this setup
  - explain where non-transformer models fail
  - discuss decode profile impact separately from architecture impact
  - state limitations (dataset scale, compute budget, selected metrics, synthetic-to-real gap)

## 3.5 `thesis/chapters/05_zaver.tex`

- [ ] Re-state what objective was achieved.
- [ ] Summarize the concrete outcomes, not chapter text.
- [ ] Summarize practical contribution of your pipeline/reproducibility setup.
- [ ] Provide 3-5 future work directions:
  - larger TSPLIB sets and larger N
  - stronger graph architectures / decoding
  - ablation of edge features and profile settings
  - runtime-quality tradeoff analysis
  - possible extension to ATSP/VRP variants

## 4. Minimum figure/table package to include

- [ ] Table 1: datasets and splits (synthetic + concorde synthetic + TSPLIB14).
- [ ] Table 2: compared experiment configurations.
- [ ] Table 3: overall model ranking (`model_ranking.csv`).
- [ ] Table 4: baseline vs optimized decode comparison (`eval_profile_compare.csv`).
- [ ] Table 5: per-instance difficulty summary (`instance_gap_matrix.csv`).
- [ ] Figure 1: pipeline diagram (generate -> qa -> train -> eval -> analyze).
- [ ] Figure 2: example GT vs prediction on selected TSPLIB instances.
- [ ] Figure 3: mean gap by model family.

## 5. Remaining placeholders to fill before final submission

- [ ] `thesis/metadata.tex`: replace `\PageCount` after text is finalized.
- [ ] `thesis/chapters/01_uvod.tex`: replace placeholder text.
- [ ] `thesis/chapters/02_sucasny_stav.tex`: replace placeholder text.
- [ ] `thesis/chapters/03_ciel_metodika.tex`: add partial goals + full method text.
- [ ] `thesis/chapters/04_vysledky_diskusia.tex`: add results + discussion.
- [ ] `thesis/chapters/05_zaver.tex`: add final conclusion text.
- [ ] `thesis/references.bib`: add all cited sources from chapter 2 and methods/results.
- [ ] Optional appendices: fill `thesis/appendix/A_prilohy.tex` if needed.

## 6. Suggested writing order

- [ ] Write chapter 2 first (literature and definitions).
- [ ] Write chapter 3 second (methodology from project code/configs).
- [ ] Write chapter 4 third (results from `runs/analysis` and selected figures).
- [ ] Write chapter 1 (`Uvod`) after chapters 2-4.
- [ ] Write chapter 5 (`Zaver`) last.
- [ ] Final pass: language/style consistency with Slovak academic tone and IS 1/2024 rules.
