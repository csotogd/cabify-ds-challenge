# Cabify · Data Science Challenge

Both parts of the challenge: the experiment to decide whether Just Eat keeps funding the photography service, and a model that tells whether two GPS traces are the same route.

## Where to start

There is a fair amount of material here. This is what I would read depending on how much time you have.

| You have | Read |
|---|---|
| 5 minutes | This README down to "What I found". |
| 20 minutes | [02_route_modelling.ipynb](02_route_modelling.ipynb), the six-line summary at the top and sections 9, 10 and 11. That is where the decision-changing material is. |
| An hour | Both notebooks. The first one is about causal inference and power; the second about modelling. |
| I want the code | `src/routes/` for the models, `scripts/` for the pipeline, `tests/` for the 13 tests. |

Both notebooks are executed with outputs. There is also HTML if you prefer a browser, though the formulas load MathJax from a CDN and render as raw LaTeX without a network connection. The notebook does not have that problem.

## What I found

Two unrelated assignments. In both, the thing that added most value was not improving the method.

**Part 1.** The experimental design works out: matched cities, a draw between immediate and deferred access, intention-to-treat analysis. The problem is that nobody had worked out how much the margin would have to grow for the programme to pay for itself. It comes to 1.25 points. The ceiling that the dilution arithmetic allows is 1 point. And detecting 1.25 points would take roughly 1,010 cities. Under those assumptions the experiment cannot tell "it pays" from "it does not pay", and that is worth knowing before spending months on it.

**Part 2.** Eight geometric measurements give 94.3% balanced accuracy. Three more sophisticated alternatives do not improve on that in any defensible way. But 83% of the labels rest on a single person, and the panel does not share a criterion: one annotator marks "they differ" 65% of the time and another 39.7%. The model agrees 98% with the second and 88% with the first. And once the problem is measured in euros instead of in correct calls, the model catches 90.2% of the cases but recovers 97.4% of the money: it fails on the cheap trips.

## What I did and what the AI did

I built this working with AI, and I think it is more useful to explain how than to pretend otherwise.

**The tool was Claude (Claude Code).**

**The AI wrote the code.** All of it: the models, the pipeline, the figures, the tests. I did not write the code by hand. What I did do was read and review it in full, which is a different thing from accepting it.

**Ideas went both ways.** Some I brought and asked it to develop. Others I asked for outright: "what would you do here?". Then I assessed what came back, set up experiments to check it, kept what held up and dropped the rest. Several proposals are not in this repository because I tried them and they did not stand.

**What is here has passed a filter.** When something only half convinced me, I refined it through prompting until I understood it or until it was clear it would not work. When a line of work ran out, I pivoted. Several of the sections I like most — the break-even threshold in Part 1, measuring the error in euros in Part 2 — came out of that back and forth, not out of a first answer.

**The decisions and the conclusions are mine.** Which model gets picked, what gets recommended, what is left out and which limitations are declared. If something in here is wrong, it is mine.

So that this review is something you can check rather than take on trust: the protocol is frozen before looking at the test set, results and seeds are stored in `results/`, and there are 13 tests. Everything in this repository can be re-run and verified.

## Contents

| File | What it is |
|---|---|
| [01_photography_experiment.ipynb](01_photography_experiment.ipynb) | Part 1. The retrospective study as supporting evidence (§1–12), a bridge (§13), and the prospective experiment (§14–27). |
| [02_route_modelling.ipynb](02_route_modelling.ipynb) | Part 2. EDA, four models, error inspection, euros, conformal prediction and a recommendation. |
| [challenge_brief.md](challenge_brief.md) | The original brief, for convenience. |
| [src/routes/](src/routes/) · [scripts/](scripts/) · [tests/](tests/) | Implementation, pipeline and tests. |
| [results/](results/) | Metrics, predictions, splits, protocol and seeds. All reproducible. |
| [figures/](figures/) | The figures the notebooks generate. |
| [requirements.txt](requirements.txt) | Python 3.12 and the exact versions I used. |

---

## Part 1 · The experiment

The numbers in this part are **synthetic**. I do not have internal Just Eat records and I am not pretending to. Public figures give scale and carry a date; they are not a sample.

I start with the historical data, because it already exists. The waiting list offers a candidate comparison: everyone requested the service, some published and others are still waiting. But everything hinges on why some went first. If the sales rep prioritised whoever was about to launch a promotion, I am measuring the promotion. So the retrospective study is good for estimating variability, choosing a metric and sizing timelines. Not for concluding.

The prospective experiment randomises **cities**, not restaurants. If I randomise restaurants within a city, the control stops being a control: the treatment is taking orders away from it. The unit has to contain the competition. Cities are matched before the draw, and within each pair one gets immediate access and the other deferred access with an announced date. The service is not withdrawn from anyone. Requesting it stays voluntary and prices stay free, because what is being funded is the policy, not a laboratory version of it.

The outcome is the weekly contribution margin net of the programme cost, weighted by each city's prior margin.

### The break-even threshold (§21.4)

This is the section that closes the assignment. The threshold is the weekly programme cost in a city divided by its prior weekly margin, which puts it in the same units as the estimated effect:

| | Growth points | Cities needed to detect it |
|---|---:|---:|
| Plausible effect given dilution | 1.00 | 1,574 |
| Break-even threshold | **1.25** | **1,010** |
| What the design was sized for | 5.00 | 68 |

Assumptions: 32 shoots per city, €40,000 weekly margin, 16 weeks, sessions at €250. The crossover sits around €200 per session. All hypothetical.

Two things follow. First, with the real cost per session and the city margin — which Finance has — that table can be recomputed in an afternoon, and it decides between launching, redesigning to concentrate the service where the exposed margin is larger, or recommending closure without spending on the trial. Second, the test should be against the threshold rather than against zero. The service is already running: the burden of proof is to show that it pays for itself, not that it does something. Under the classic null you can come out with a result that is significant, positive and still value-destroying.

---

## Part 2 · The model

This part uses the real file linked in the brief. 2,386 journeys, 3,027 annotations, 8 people.

Before modelling: a row is an annotation, not a journey. There are 123 repeated records from the same annotator on the same journey. One vote per person per journey; anyone who contradicts themselves does not vote. A 60/20/20 split by journey (1,432 / 477 / 477), stratified, fixed before any fitting. In the test set, 443 cases have a defined majority.

| Model | Balanced accuracy on test |
|---|---:|
| Basic geometry + XGBoost | 94.3% |
| Interpretable vector (42) + XGBoost | 94.8% |
| Reduced t2vec + classifier | 71.8% |
| Vector + t2vec + XGBoost | 95.5% |

The selection rule was frozen on validation using soft log loss and picks basic geometry. The improvements from the extended vector and the hybrid have paired bootstrap intervals that include zero, so they cannot be defended.

**On t2vec.** My adaptation does not reproduce the paper. Auditing it afterwards I found that free reconstruction was failing and that the sampled loss had a defect: the end-of-sequence token never took part as a negative. I retrained with normalisation over the full vocabulary and stored it separately; it improves some indicators and still does not reconstruct properly. I am keeping the failed attempt documented. It describes my code, not the published method.

### Who holds up each label

Of the 2,386 journeys, 1,816 have a single decided vote, 410 have two and 19 have three. Among those with two or more, 31 end in a tie.

The panel is very uneven. Annotator 2 signs 918 annotations across 856 journeys and is the only voter on 612 of them, 26% of the file. Annotator 5 did 21. And the criteria differ: annotator 1 marks "they differ" 65% of the time and annotator 7 does so 39.7% of the time, and they are precisely the pair that agrees least among those sharing enough journeys (82.6% raw agreement, Cohen's kappa 0.66, over 46 journeys).

That shows up in the model. On journeys with a single decided annotator, accuracy runs from 88.2% with annotator 1 (n=119) to 98.0% with annotator 7 (n=147). The stricter the person, the worse the model agrees with them. A caveat: each group contains different journeys, so part of the gap may be difficulty rather than criterion.

Performance also splits by vote count. On validation it goes from 94.3% with one vote to 100% with two or three; on test, from 92.6% to 98.6% and 100%. The 21 validation errors and 27 of the 28 test errors fall on single-vote journeys, somewhat above what the group size alone would explain (p = 0.016 and p = 0.027, hypergeometric, descriptive and after the fact). The comparison is not clean: a journey with two votes and a defined majority is, by construction, one where two people agreed. Label noise and genuine difficulty cannot be separated with this file. Figures in [errors_by_votes.json](results/routes/errors_by_votes.json).

The 7 test ties are plotted. In all seven the central body of the trip coincides and the largest separation falls in the first or last quarter: the disagreement is about how the trip starts or ends. The model does not reproduce that doubt reliably, and with seven cases the comparison concludes nothing.

### In euros (§9)

The fare is set before the trip, so the base and minimum fares cancel out when comparing the two routes and the per-kilometre and per-minute prices are enough:

```
Δ€ = €/km · (km_real − km_est) + €/min · (min_real − min_est)
```

Public Cabify fares by city, standard category, consulted on 19 October 2026. Durations from OSRM where the map match is reliable (1,398 of 2,386). This is an order-of-magnitude estimate, not an audit: the fares are from 2026 and the file from 2021, I ignore tolls, surge pricing and promotions, and Colombia and Ecuador use estimated values because the source does not publish a single figure. Results in `results/routes_euros/`.

The median deviation is €0.11, 3.1% of an average fare of €4.63. The net is −€9 over €11,054 billed, so there is no systematic leak. But that netting happens in the spreadsheet, not on the street: if the real trip runs longer, Cabify and the driver absorb it, and if it runs shorter the customer overpaid with no refund. 8.9% of customers paid more than 10% above what their actual trip was worth.

Against a group EBITDA margin of 3.3% in 2025 (€27.2M on €833.3M), 48% of journeys deviate by more than that same journey leaves in profit, and the gross amount moved is equivalent to 2.1 times the margin. That 3.3% is a group margin, not a per-trip one, and the sample is dominated by Latin American cities with low fares: the comparison gives scale, not accounting.

The impact is concentrated: 10% of journeys account for 55% of it. The Spearman correlation between the human label and the euro deviation is 0.45, yet among the top 10% by impact, 93% are labelled "they differ". And the model's errors land where there is little money: the 25 undetected differences add up to 2.5% of the total. **It catches 90.2% of the cases and recovers 97.4% of the money.** Nobody trained it for that, it is a property of this file, and it would need re-checking against new data.

### Conformal prediction (§10)

CV+ over the training set, because validation was already spent choosing model, iteration and threshold. Mondrian calibration in two groups — one vote (n=1,085) and two or more (n=263) — because they are different noise regimes; the three- and four-vote groups have 8 and 2 cases and do not even reach the theoretical minimum. The interval over the vote proportion translates into three decisions: same, different, or human review.

| α | Sent to review | Error on automated calls |
|---|---:|---:|
| today, fixed threshold | 0% | 6.3% |
| 0.20 | 4.1% | 2.8% |
| 0.15 | 6.1% | 1.9% |
| 0.10 | 12.9% | 0.5% |

At α = 0.05 abstention climbs to 84% and the method stops being usable. The 90th-percentile residual is 30% smaller when the label has two votes, which puts a number on what a second annotation is worth. Implementation in [conformal.py](src/routes/conformal.py), results in `results/routes_conformal/`.

### Cost and scale (§10.1)

Deciding a new pair costs 6.2 ms, of which XGBoost consumes 0.8 µs. The cost is in the geometry. A million journeys a day is 1.7 CPU-hours, so compute is not the problem. Human review is: that 12.9% over a million journeys is 129,000 reviews a day.

The policy that follows combines both sections: review where the model is unsure **and** there is money at stake, ranking the abstained cases by estimated deviation until the budget runs out. I do not implement it because it needs the real review budget and the internal pricing formula. Measurement in [inference_cost.json](results/routes/inference_cost.json).

### Recommendation (§11)

Deploy basic geometry with conformal prediction at α = 0.15, review by economic impact rather than by a fixed rate, and spend the next investment on a second annotation rather than on a bigger model. I would not deploy t2vec or the road-network extension: neither demonstrates an improvement and both multiply the operational cost.

### Other things inside

- **EDA.** Nulls, coordinate structure, lengths, repeated points, sampling density, trace gaps, geographic composition and votes. The file is structurally clean — zero nulls, zero invalid coordinates — but 19.2% of the steps in the real routes have zero length and 133 routes have jumps of more than 1 km. I did not drop any journey or trim outliers. Figures in [eda.json](results/routes/eda.json).
- **Error inspection.** Four correct calls and six test errors with full routes, a zoom, distance profiles and TreeSHAP contributions. Identifiers in [inspection.json](results/routes/inspection.json). It changes no models or thresholds.
- **`scripts/plot_val_errors.py`** plots the validation disagreements. Descriptive: validation took part in the selection, so it is not an independent error measurement.
- **Road-network extension.** OpenStreetMap extracts via Geofabrik, a local OSRM engine and a directed distance graph. Exploratory, done after inspecting the test set, with the earlier splits and models left untouched. Durations come from a fixed profile with no traffic; the simplified graph does not encode turn restrictions. Reports in `results/routes_graph/`.

---

## Reproducing

Python 3.12. From the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Part 1:

```sh
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 01_photography_experiment.ipynb
```

Part 2, in order:

```sh
python scripts/run_routes.py
python scripts/audit_t2vec_reconstruction.py
PYTHONPATH=src python -m routes.t2vec_corrected
python scripts/audit_t2vec_reconstruction.py --corrected
python scripts/download_geofabrik.py
python scripts/prepare_road_extract.py
python scripts/start_local_osrm.py
PYTHONPATH=src python -m routes.graph_extension
python scripts/evaluate_graph_extension.py
PYTHONPATH=src python -m routes.pricing
PYTHONPATH=src python -m routes.conformal
PYTHONPATH=src python scripts/benchmark_inference.py
python scripts/build_routes_notebook.py
python scripts/render_notebook.py 02_route_modelling.ipynb
PYTHONPATH=src python -m unittest discover -s tests -v
docker stop cabify-challenge-osrm
```

`scripts/render_notebook.py` takes either notebook and produces the styled HTML.

To predict a new pair: `python scripts/predict_route.py pair.json`, with the fields `estimated_route` and `real_route`, each a list of `[latitude, longitude]` coordinates.

**Practical notes.** The pipeline downloads the public JSON if it is missing. Raw data, caches and models stay out of Git. The road-network extension needs Docker and several GB of disk; the engine is pinned by digest in `scripts/start_local_osrm.py` and the service only listens on 127.0.0.1:5057, so traces never leave the machine. Geofabrik's "latest" URLs change over time, so keeping the caches is the only thing that guarantees reproducing that exact cartography. The original pipeline and the extension are independent: `scripts/evaluate_graph_extension.py` checks the hashes recorded in `results/routes_graph/original_artifact_hashes.json` to make sure the original models were left untouched.

Maps © OpenStreetMap contributors, under [ODbL](https://www.openstreetmap.org/copyright), distributed by [Geofabrik](https://download.geofabrik.de/).

## What is not demonstrated

- That the model transfers to a new city. The split is random and the trips share streets.
- That the favourable asymmetry in euros holds outside this file.
- That the conformal guarantee is exact. It requires exchangeability and here that only holds approximately.
- That the euro figures are the real ones. They are 2026 fares applied to 2021 trips.
- Anything in Part 1. It is a design with hypothetical numbers, pending validation of market availability, historical variability, participation and cross-city interference.
