# Who Wins the 2026 World Cup? Eleven Models, One Tournament

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: Public Domain](https://img.shields.io/badge/data-public%20domain-brightgreen.svg)](data/SOURCES.md)

A small, transparent suite of forecasting models that each simulate the **2026 FIFA World Cup** and predict who lifts the trophy — built with the methods from the O'Reilly book **[*Soccer Analytics with Machine Learning*](#book)**.

Eleven models, one for (almost) every chapter of the book, are run through the same tournament simulator. They crown **four different champions** — and that disagreement is the point.

![Title probability by model](model_comparison_heatmap.png)

## The result (20,000 simulations per model)

| Team | Consensus | Range across models |
|------|----------:|--------------------:|
| Spain | **20.2%** | 13–29% |
| France | 14.0% | 10–21% |
| Argentina | 13.7% | 3–19% |
| Netherlands | 9.7% | 6–18% |
| England | 9.4% | 3–13% |

Each model's outright pick: **Spain** (Elo, Poisson, Negative Binomial, logistic, KNN, PageRank), **Argentina** (random forest, XGBoost), **France** (neural net), **Netherlands** (Colley).

> ⚠️ The team-strength ratings and market odds in the code are an **illustrative snapshot** — refresh them before relying on the numbers (see [`data/SOURCES.md`](data/SOURCES.md)).

## The models

| Model | Chapter | Idea |
|-------|---------|------|
| Elo / Colley / PageRank | 8 | Team ratings (form-based and results-graph) |
| Poisson / Negative Binomial | 4 | Goal-count models |
| Logistic regression | 5 | Win/draw/loss classifier |
| KNN | 4–5 | Prediction by nearest historical matchups |
| Random Forest / XGBoost | 6 | Tree ensembles |
| Neural network (MLP) | 7 | Small multilayer perceptron |
| Market-implied odds | 9 | Bookmaker odds → de-vigged probabilities (benchmark only) |

All eleven are reduced to a common `P(win/draw/loss)` interface and run through one vectorized simulator (12 groups → 8 best thirds → 32-team knockout).

![One interface, eleven engines](pipeline_schematic.png)

## Quick start

```bash
pip install -r requirements.txt

python wc2026_simple_model.py    # the gentle intro: Elo + Poisson, one model
python wc2026_model_suite.py     # the full 11-model suite (set WC_N for #sims)
python make_charts.py            # comparison heatmap + consensus/range
python add_article_charts.py     # market-disagreement + CV-comparison charts
python build_article2_extras.py  # model-agreement heatmap + pipeline schematic
```

The notebook [`world_cup_2026_prediction.ipynb`](world_cup_2026_prediction.ipynb) walks through the single-model version step by step.

## Repository layout

```
wc2026_simple_model.py      # single Elo + Poisson + Monte Carlo model
wc2026_model_suite.py       # the eleven-model suite
make_charts.py / add_article_charts.py / build_article2_extras.py  # figures
world_cup_2026_prediction.ipynb  # annotated notebook (single model)
data/                       # match data (public domain) + SOURCES.md
model_title_probabilities.csv, model_picks.csv, classifier_cv_metrics.csv  # results
*.png                       # generated figures
```

## Method, in one paragraph

Every team gets a strength rating. Each matchup is converted into win/draw/loss probabilities — by a Poisson goal model, a fitted classifier, or a rating curve, depending on the model. The whole 48-team tournament is then simulated 20,000 times and the champion tallied. The models are trained or computed on **358 real international matches** (World Cups 2010–2022 + Euros 2020/2024). Full write-ups are in the accompanying article series.

<a name="book"></a>
## 📘 From the book

This project applies the techniques from **[*Soccer Analytics with Machine Learning*](<BUY-LINK>)** (O'Reilly, 2026) by Haipeng Gao, Ari Joury, Weining Shen, and Guanyu Hu — Poisson and regression models (Ch 4), classification (Ch 5), tree-based methods (Ch 6), neural networks (Ch 7), team ratings (Ch 8), and market/odds analysis (Ch 9).

- 📖 **Get the book:** `<BUY-LINK>`
- 🧑‍💻 **Companion code for the whole book** (with worked StatsBomb examples): `<BOOK-REPO-URL>`

If this repo was useful, the book is where you learn to build every one of these models from scratch.

## License

Code: [MIT](LICENSE). Data: public domain (see [`data/SOURCES.md`](data/SOURCES.md)).
