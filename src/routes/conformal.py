"""Conformal prediction over the soft target, with explicit abstention.

The chosen model is a regressor over `target_soft`, the proportion of "they differ" votes.
Classification is a threshold applied afterwards. Here we produce an interval around
that proportion and turn the interval into a three-valued decision:

    interval entirely below 0.5   -> the same
    interval entirely above 0.5   -> they differ
    interval contains 0.5         -> no decision, send to human review

We use CV+ (Barber et al., 2021) over the training set rather than split conformal: validation
already took part in choosing model, iteration and threshold, and holding out a calibration fold
would leave ~50 cases in the two-or-more votes group. CV+ uses the 1,348 training journeys as
calibration via out-of-fold residuals, and guarantees coverage of at least 1 - 2*alpha.

Calibration is Mondrian with two groups: journeys with **one vote** and journeys with
**two or more**. These are two different noise regimes -- a journey with two votes and a defined
majority is, by construction, one where the people agreed -- and a single average width
would misstate both at once. Three and four votes are not separated: with 8 and 2 calibration
cases they do not reach the minimum n >= 1/alpha - 1.

Validity warning: the split is random and the trips share streets, so exact
exchangeability between calibration and test only holds approximately.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from .data import ROOT, SEED, load_data
from .features import build_features, BASE_FEATURES

N_FOLDS = 10
CHOSEN = dict(max_depth=2, min_child_weight=8, reg_lambda=8)
N_ROUNDS = 268          # iteration chosen on validation (best_iteration 267, inclusive)
GROUPS = {'1 vote': lambda v: v == 1, '2 or more votes': lambda v: v >= 2}

def _fit(x, y):
    model = XGBRegressor(objective='binary:logistic', eval_metric='logloss',
                         n_estimators=N_ROUNDS, learning_rate=.035, subsample=.85,
                         colsample_bytree=.9, random_state=SEED, n_jobs=1,
                         tree_method='hist', **CHOSEN)
    model.fit(x, y, verbose=False)
    return model

def out_of_fold(x_train, y_train):
    """Out-of-fold residuals and each fold's model, for the CV+ construction."""
    folds = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    residuals = np.empty(len(y_train))
    models, assignment = [], np.empty(len(y_train), dtype=int)
    for k, (fit_idx, held_idx) in enumerate(folds.split(x_train)):
        model = _fit(x_train[fit_idx], y_train[fit_idx])
        residuals[held_idx] = np.abs(y_train[held_idx]-model.predict(x_train[held_idx]))
        assignment[held_idx] = k
        models.append(model)
    return residuals, models, assignment

def cv_plus_interval(x_new, models, assignment, residuals, alpha):
    """CV+ interval: quantiles of {pred_{-k(i)}(x) -/+ R_i} over the calibration set."""
    preds = np.column_stack([model.predict(x_new) for model in models])   # (n_new, K)
    per_point = preds[:, assignment]                                      # (n_new, n_cal)
    lower = np.quantile(per_point-residuals, alpha, axis=1, method='lower')
    upper = np.quantile(per_point+residuals, 1-alpha, axis=1, method='higher')
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)

def decide(lower, upper, boundary=.5):
    """0 the same · 1 they differ · -1 no decision."""
    out = np.full(len(lower), -1)
    out[upper < boundary] = 0
    out[lower > boundary] = 1
    return out

def run(alphas=(.30, .25, .20, .15, .10, .05), evaluate_on='test'):
    df, _ = load_data()
    features = build_features().loc[df.index]
    x = np.ascontiguousarray(features[BASE_FEATURES], dtype=np.float32)
    train = ((df.split == 'train') & df.target_soft.notna()).to_numpy()
    held = ((df.split == evaluate_on) & df.target_hard.notna()).to_numpy()
    y = df.target_soft.to_numpy(dtype=np.float32)

    votes_cal, votes_held = df.n_votes.to_numpy()[train], df.n_votes.to_numpy()[held]
    residuals, models, assignment = out_of_fold(x[train], y[train])
    truth = df.target_hard.to_numpy()[held].astype(int)
    soft_held = df.target_soft.to_numpy()[held]

    records, per_case = [], {}
    for alpha in alphas:
        lower = np.empty(held.sum()); upper = np.empty(held.sum())
        widths = {}
        for name, belongs in GROUPS.items():
            cal_mask = belongs(votes_cal); new_mask = belongs(votes_held)
            if not new_mask.any(): continue
            lo, hi = cv_plus_interval(x[held][new_mask], models, assignment[cal_mask],
                                      residuals[cal_mask], alpha)
            lower[new_mask], upper[new_mask] = lo, hi
            widths[name] = float(np.mean(hi-lo))
        decision = decide(lower, upper)
        answered = decision >= 0
        covered = (soft_held >= lower) & (soft_held <= upper)
        records.append(dict(alpha=alpha, nominal_coverage=1-2*alpha,
            empirical_coverage=float(covered.mean()),
            abstention_rate=float((~answered).mean()),
            error_rate_answered=float((decision[answered] != truth[answered]).mean()) if answered.any() else float('nan'),
            n_answered=int(answered.sum()), n_abstained=int((~answered).sum()),
            mean_width=float(np.mean(upper-lower)), width_by_group=widths,
            coverage_by_group={name: float(covered[belongs(votes_held)].mean())
                               for name, belongs in GROUPS.items() if belongs(votes_held).any()},
            abstention_by_group={name: float((~answered)[belongs(votes_held)].mean())
                                 for name, belongs in GROUPS.items() if belongs(votes_held).any()}))
        per_case[alpha] = pd.DataFrame({'lower': lower, 'upper': upper, 'decision': decision,
                                        'truth': truth, 'n_votes': votes_held},
                                       index=df.index[held])
    return pd.DataFrame(records), per_case, (residuals, votes_cal)

if __name__ == '__main__':
    table, per_case, (residuals, votes_cal) = run()
    out = ROOT/'results/routes_conformal'; out.mkdir(parents=True, exist_ok=True)
    narrowing = {name: float(np.quantile(residuals[belongs(votes_cal)], .9))
                 for name, belongs in GROUPS.items()}
    payload = {'method': 'CV+ (Barber et al. 2021), Mondrian by vote count',
        'folds': N_FOLDS, 'config': CHOSEN, 'n_rounds': N_ROUNDS,
        'guarantee': 'coverage >= 1 - 2*alpha over target_soft; approximate because of street overlap',
        'calibration_sizes': {name: int(belongs(votes_cal).sum()) for name, belongs in GROUPS.items()},
        'residual_q90_by_group': narrowing,
        'curve': json.loads(table.to_json(orient='records'))}
    (out/'conformal.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for alpha, frame in per_case.items():
        frame.to_json(out/f'cases_alpha_{alpha:.2f}.json', orient='index', indent=1)
    print(table[['alpha','nominal_coverage','empirical_coverage','abstention_rate',
                 'error_rate_answered','mean_width']].to_string(index=False))
    print('\nP90 residual by group:', {k: round(v,4) for k,v in narrowing.items()})
