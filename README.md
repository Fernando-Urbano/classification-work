# Classification Reconstruction
```
import os
import sys
# Data manipulation
import pandas as pd
import numpy as np
import itertools as it
import datetime as dt
import re
import math
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 50)
# Data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
# Models
import statsmodels as sm
import scipy as sp
import xgboost as xgb
import sklearn as sk
import shap
```

```
def evaluate_metrics(
    y_real,
    y_proba,
    agg_metrics_df=pd.DataFrame({"model_name": []}),
    model_name=None,
    show=False,
    replace=True
):
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba
    class_no_default = df[df['real'] == 0]
    class_default = df[df['real'] == 1]
    ks = sp.stats.ks_2samp(class_no_default['proba'], class_default['proba'])
    roc_auc = sk.metrics.roc_auc_score(df['real'] , df['proba'])
    f1_score = 0
    best_threshold = None
    for i in list(np.arange(.06, .14, .00010)):
        proba_round = df['proba'].apply(lambda x: math.ceil(x - i))
        new_f1_score = sk.metrics.f1_score(df['real'], proba_round, average="binary")
        if new_f1_score > f1_score:
            f1_score = new_f1_score
            best_threshold = i
    new_metrics_df = pd.DataFrame(
        data={
            "model_name": [model_name],
            "ks": [ks.statistic],
            "auc": [roc_auc],
            "f1_score": [f1_score],
            "f1_best_threshold": [best_threshold]
        }
    )
    if replace:
        agg_metrics_df = agg_metrics_df.loc[lambda df: df.model_name != model_name]
    if model_name not in list(agg_metrics_df.model_name):
        agg_metrics_df = pd.concat([agg_metrics_df, new_metrics_df]).reset_index(drop=True)
    if show:
        print(agg_metrics_df)
    return agg_metrics_df
```
```
os.chdir(
    "/Users/bi003761/Desktop/credito"
    + "/project_credit_card_pj/model_reconstruction"
)
```
```
bc_pj_features = (
    pd.read_csv("data/final_df.csv", dtype={"eight_dig_cnpj": "object"})
    .fillna(0)
    .apply(lambda df: df.replace({float('inf'): 0, -float('inf'): 0}))
    .drop("Unnamed: 0", axis=1)
    .rename({"sample": "sample_df"}, axis=1)
    .loc[
        :,
        lambda df: [c for c in df.columns if bool(re.search("^first|^last|^middle", c)) == False]
    ]
)
(
    bc_pj_features
    .select_dtypes(include=np.number)
    .agg(["mean", "median", "min", "max"])
    .transpose()
    .reset_index()
    .rename({"index": "id"}, axis=1)
    .sort_values("id")
    .loc[lambda df: df.id != "cpfcnpj_eight_dig"]
    .reset_index(drop=True)
)
```
```
train_data = bc_pj_features.loc[lambda df: df.sample_df == "TR"].reset_index(drop=True)
test_data = bc_pj_features.loc[lambda df: df.sample_df == "TE"].reset_index(drop=True)
pct_train = len(train_data.sample_df) / (len(test_data.sample_df) + len(train_data.sample_df))

print(f"Test DF size: {str(len(test_data.sample_df))} - {(1 - pct_train):.1%}; Percentage of default {train_data.target.mean():.1%}")
print(f"Train DF size: {str(len(train_data.sample_df))} - {pct_train:.1%}; Percentage of default {test_data.target.mean():.1%}")
```
 # Baseline Model
 Features used:
 - Score HPJ8
```
dt_refs = pd.read_csv(
    "data/final_bcb.csv",
    dtype={"eight_dig_cnpj": "object", "dt_ref": "object"},
    usecols=["eight_dig_cnpj", "dt_ref"]
)
train_dt_ref = (
    dt_refs
    .loc[lambda df: df.eight_dig_cnpj.isin(list(train_data.eight_dig_cnpj))]
    .groupby("dt_ref", as_index=False)
    .agg({"eight_dig_cnpj": "count"})
    .assign(dt_ref=lambda df: df.dt_ref.map(lambda x: pd.to_datetime(x)))
)
test_dt_ref = (
    dt_refs
    .loc[lambda df: df.eight_dig_cnpj.isin(list(test_data.eight_dig_cnpj))]
    .groupby("dt_ref", as_index=False)
    .agg({"eight_dig_cnpj": "count"})
    .assign(dt_ref=lambda df: df.dt_ref.map(lambda x: pd.to_datetime(x)))
)
fig, ax = plt.subplots(figsize = (13, 3))
ax.plot(train_dt_ref.dt_ref, train_dt_ref.eight_dig_cnpj, color="red", marker=".")
ax.plot(test_dt_ref.dt_ref, test_dt_ref.eight_dig_cnpj, color="blue", marker=".")
plt.title("Number of Observations by Date of Reference")
plt.show()
```
```
pct_train_dt_ref = (
    test_dt_ref
    .rename({"eight_dig_cnpj": "test"}, axis=1)
    .merge(
        right=(
            train_dt_ref
            .rename({"eight_dig_cnpj": "train"}, axis=1)
        ),
        on=["dt_ref"],
        how="outer"
    )
    .assign(pct=lambda df: df.train / (df.train + df.test))
)
fig, ax = plt.subplots(figsize = (13, 3))
ax.plot(pct_train_dt_ref.dt_ref, pct_train_dt_ref.pct, color="red", marker=".")
ax.set_ylim([0, 1])
ax.set_yticks(ax.get_yticks().tolist())
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
plt.title("Percentage Allocated in Training Sample by Date of Reference")
plt.show()
```
```
features = ["score_hpj8"]
tr = train_data.loc[:, ["target"] + features]
te = test_data.loc[:, ["target"] + features]
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
print("Coefficient HPJ8: " + str(model.coef_[0][0]))
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    model_name="Baseline"
)
agg_metrics
```
```
features = ["score_hpj8"]
scaler = sk.preprocessing.StandardScaler()
scaler.fit(train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(train_data.loc[:, features]), columns=features)
tr = pd.concat([train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(test_data.loc[:, features]), columns=features)
te = pd.concat([test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Baseline + Scale"
)
agg_metrics
```
```
features = ["tempo_relacionamento", "score_hpj8", "cnae_grupos"]
scaler = sk.preprocessing.StandardScaler()
scaler.fit(train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(train_data.loc[:, features]), columns=features)
tr = pd.concat([train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(test_data.loc[:, features]), columns=features)
te = pd.concat([test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + CNAE + Relation with BCB"
)
agg_metrics
```
```
features = list(train_data.drop("target", axis=1).select_dtypes(include=np.number).columns)
print("- " + "\n- ".join(features))
```
```
features = list(train_data.drop("target", axis=1).select_dtypes(include=np.number).columns)
scaler = sk.preprocessing.StandardScaler()
scaler.fit(train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(train_data.loc[:, features]), columns=features)
tr = pd.concat([train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(test_data.loc[:, features]), columns=features)
te = pd.concat([test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + All Features"
)
agg_metrics
```
```
from statsmodels.stats.outliers_influence import variance_inflation_factor
tr_features = train_data.drop("target", axis=1).select_dtypes(include=np.number)
number_features_pre_vif = len(tr_features.columns)
vif_statistic = vif_df = pd.DataFrame({"vif": [1e3]})
while max(vif_statistic.vif) > 5:
    vif_statistic = pd.DataFrame(
        data={
            "feature": tr_features.columns,
            "vif": [
                sm.stats.outliers_influence.variance_inflation_factor(
                    exog=tr_features.values,
                    exog_idx=i
                )
                for i in range(tr_features.shape[1])
            ]
        }
    )
    vif_statistic.sort_values("vif", ascending=False, inplace=True)
    vif_statistic.reset_index(drop=True, inplace=True)
    tr_features.drop(vif_statistic.feature[0], axis=1, inplace=True)
print(f"Number of features with VIF lower than 5: {str(len(vif_statistic.feature))}/{str(number_features_pre_vif)}")
```
```
features = list(vif_statistic.feature)
print("Features used according to VIF:\n" + "- " + "\n- ".join(features))
```
```
features = [
    f for f in list(train_data.drop("target", axis=1).select_dtypes(include=np.number).columns)
    if f not in list(vif_statistic.feature)
]
print("Features not used according to VIF:\n" + "- " + "\n- ".join(features))
```
```
features = list(vif_statistic.feature)
scaler = sk.preprocessing.StandardScaler()
scaler.fit(train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(train_data.loc[:, features]), columns=features)
tr = pd.concat([train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(test_data.loc[:, features]), columns=features)
te = pd.concat([test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + VIF BCB"
)
agg_metrics
```
```
explainer = shap.Explainer(model, tr.drop("target", axis=1))
shap_values = explainer(tr.drop("target", axis=1))
shap.plots.bar(shap_values, max_display=14)
```
```
shap.plots.beeswarm(shap_values)
```
```
features = [
    c for c in train_data.select_dtypes(include=np.number).columns
    if bool(re.search("^dummy|^cnae|^target", c)) is False
]
tr_features = train_data.loc[:, features]
tr_features.hist(figsize=(25, 20), bins=50, color="cornflowerblue");
```
```
features = [
    c for c in train_data.select_dtypes(include=np.number).columns
    if bool(re.search("^dummy|^cnae|^dif|^target", c)) is False
]
tr_features = (
    train_data.loc[:, features]
    .replace({0: float("nan")})
    .applymap(lambda x: math.log(x + 2))
)
tr_features.hist(figsize=(25, 20), bins=50, color="cornflowerblue");
```
```
features = list(vif_statistic.feature)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8", c)) is False]
log_train_data = train_data.copy()
log_train_data[log_features] = log_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_test_data = test_data.copy()
log_test_data[log_features] = log_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_test_data.loc[:, features]), columns=features)
te = pd.concat([log_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + VIF Log BCB"
)
agg_metrics
```
```
from statsmodels.stats.outliers_influence import variance_inflation_factor
tr_features = train_data.drop("target", axis=1).select_dtypes(include=np.number)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8", c)) is False]
tr_features[log_features] = tr_features[log_features].applymap(lambda x: math.log(x + 1.01))
number_features_pre_vif = len(tr_features.columns)
vif_statistic = vif_df = pd.DataFrame({"vif": [1e3]})
while max(vif_statistic.vif) > 5:
    vif_statistic = pd.DataFrame(
        data={
            "feature": tr_features.columns,
            "vif": [
                sm.stats.outliers_influence.variance_inflation_factor(
                    exog=tr_features.values,
                    exog_idx=i
                )
                for i in range(tr_features.shape[1])
            ]
        }
    )
    vif_statistic.sort_values("vif", ascending=False, inplace=True)
    vif_statistic.reset_index(drop=True, inplace=True)
    tr_features.drop(vif_statistic.feature[0], axis=1, inplace=True)
print(f"Number of features with VIF lower than 5: {str(len(vif_statistic.feature))}/{str(number_features_pre_vif)}")
```
```
features = list(vif_statistic.feature)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8", c)) is False]
log_train_data = train_data.copy()
log_train_data[log_features] = log_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_test_data = test_data.copy()
log_test_data[log_features] = log_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_test_data.loc[:, features]), columns=features)
te = pd.concat([log_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + New VIF Log BCB"
)
agg_metrics
```
```
explainer = shap.Explainer(model, tr.drop("target", axis=1))
shap_values = explainer(tr.drop("target", axis=1))
shap.plots.bar(shap_values, max_display=14)
```
```
shap.plots.beeswarm(shap_values)
```
```
from statsmodels.stats.outliers_influence import variance_inflation_factor
tr_features = train_data.drop("target", axis=1).select_dtypes(include=np.number)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
tr_features[log_features] = tr_features[log_features].applymap(lambda x: math.log(x + 1.01))
number_features_pre_vif = len(tr_features.columns)
vif_statistic = vif_df = pd.DataFrame({"vif": [1e3]})
while max(vif_statistic.vif) > 5:
    vif_statistic = pd.DataFrame(
        data={
            "feature": tr_features.columns,
            "vif": [
                sm.stats.outliers_influence.variance_inflation_factor(
                    exog=tr_features.values,
                    exog_idx=i
                )
                for i in range(tr_features.shape[1])
            ]
        }
    )
    vif_statistic.sort_values("vif", ascending=False, inplace=True)
    vif_statistic.reset_index(drop=True, inplace=True)
    tr_features.drop(vif_statistic.feature[0], axis=1, inplace=True)
print(f"Number of features with VIF lower than 5: {str(len(vif_statistic.feature))}/{str(number_features_pre_vif)}")
```
```
features = list(vif_statistic.feature)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_train_data = train_data.copy()
log_train_data[log_features] = log_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_test_data = test_data.copy()
log_test_data[log_features] = log_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_test_data.loc[:, features]), columns=features)
te = pd.concat([log_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + New VIF Log BCB ex Time"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
from statsmodels.stats.outliers_influence import variance_inflation_factor
tr_features = train_data.drop("target", axis=1).select_dtypes(include=np.number)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
tr_features[log_features] = tr_features[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    tr_features[f] = (
        tr_features[f]
        .apply(lambda x: x if np.percentile(tr_features[f], 90) > x else np.percentile(tr_features[f], 90))
    )
number_features_pre_vif = len(tr_features.columns)
vif_statistic = vif_df = pd.DataFrame({"vif": [1e3]})
while max(vif_statistic.vif) > 5:
    vif_statistic = pd.DataFrame(
        data={
            "feature": tr_features.columns,
            "vif": [
                sm.stats.outliers_influence.variance_inflation_factor(
                    exog=tr_features.values,
                    exog_idx=i
                )
                for i in range(tr_features.shape[1])
            ]
        }
    )
    vif_statistic.sort_values("vif", ascending=False, inplace=True)
    vif_statistic.reset_index(drop=True, inplace=True)
    tr_features.drop(vif_statistic.feature[0], axis=1, inplace=True)
print(f"Number of features with VIF lower than 5: {str(len(vif_statistic.feature))}/{str(number_features_pre_vif)}")
```
```
features = list(vif_statistic.feature)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_threshold_train_data = train_data.copy()
log_threshold_train_data[log_features] = log_threshold_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_threshold_test_data = test_data.copy()
log_threshold_test_data[log_features] = log_threshold_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    log_threshold_train_data[f] = (
        log_threshold_train_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
    log_threshold_test_data[f] = (
        log_threshold_test_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_threshold_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_threshold_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_threshold_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_threshold_test_data.loc[:, features]), columns=features)
te = pd.concat([log_threshold_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + New VIF Log BCB + Threshold Time"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
explainer = shap.Explainer(model, tr.drop("target", axis=1))
shap_values = explainer(tr.drop("target", axis=1))
shap.plots.bar(shap_values, max_display=14)
```
```
important_variables = [
    "dif_limite", "score_hpj8", "dif_spending", "tempo_relacionamento", "dummy_spending",
    "dummy_limite", "cnae_grupos", "tempo_fundacao", "dummy_a_vencer_capital_de_giro",
    "dummy_a_vencer_conta_garantida", "avg_a_vencer_conta_garantida", "dif_a_vencer_desconto_de_duplicada"
]
```
```
features = [f for f in list(vif_statistic.feature) if f in important_variables]
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_threshold_train_data = train_data.copy()
log_threshold_train_data[log_features] = log_threshold_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_threshold_test_data = test_data.copy()
log_threshold_test_data[log_features] = log_threshold_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    log_threshold_train_data[f] = (
        log_threshold_train_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
    log_threshold_test_data[f] = (
        log_threshold_test_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_threshold_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_threshold_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_threshold_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_threshold_test_data.loc[:, features]), columns=features)
te = pd.concat([log_threshold_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + New VIF Log BCB + Threshold Time + Important Variables"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
features = [f for f in list(vif_statistic.feature) if f in important_variables]
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_threshold_train_data = train_data.copy()
log_threshold_train_data[log_features] = log_threshold_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_threshold_test_data = test_data.copy()
log_threshold_test_data[log_features] = log_threshold_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    log_threshold_train_data[f] = (
        log_threshold_train_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
    log_threshold_test_data[f] = (
        log_threshold_test_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
dif_columns = [c for c in log_threshold_train_data.columns if re.search("^dif", c)]
for d in dif_columns:
    log_threshold_train_data["negative_" + d] = log_threshold_train_data[d].apply(lambda x: 1 if x < 0 else 0)
    log_threshold_test_data["negative_" + d] = log_threshold_test_data[d].apply(lambda x: 1 if x < 0 else 0)
    new_feature = "negative_" + d
    features.append(new_feature)
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_threshold_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_threshold_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_threshold_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_threshold_test_data.loc[:, features]), columns=features)
te = pd.concat([log_threshold_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="Scale + New VIF Log BCB + Threshold Time + Important Variables + Negative Dif"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
pd.DataFrame({"coef": list(model.coef_[0, :]), "feature": list(tr.drop("target", axis=1).columns)})
```
```
explainer = shap.Explainer(model, tr.drop("target", axis=1))
shap_values = explainer(tr.drop("target", axis=1))
shap.plots.bar(shap_values, max_display=25)
```
```
features = [
    "score_hpj8", "dif_spending", "tempo_relacionamento", "cnae_grupos", "tempo_fundacao",
    "dummy_spending"
]
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_threshold_train_data = train_data.copy()
log_threshold_train_data[log_features] = log_threshold_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_threshold_test_data = test_data.copy()
log_threshold_test_data[log_features] = log_threshold_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    log_threshold_train_data[f] = (
        log_threshold_train_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
    log_threshold_test_data[f] = (
        log_threshold_test_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
dif_columns = [c for c in log_threshold_train_data.columns if re.search("^dif", c)]
for d in dif_columns:
    log_threshold_train_data["negative_" + d] = log_threshold_train_data[d].apply(lambda x: 1 if x < 0 else 0)
    log_threshold_test_data["negative_" + d] = log_threshold_test_data[d].apply(lambda x: 1 if x < 0 else 0)
    new_feature = "negative_" + d
    if new_feature == "negative_dif_spending":
        features.append(new_feature)
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_threshold_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_threshold_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_threshold_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_threshold_test_data.loc[:, features]), columns=features)
te = pd.concat([log_threshold_test_data[["target"]], te_features], axis=1)
model = sk.linear_model.LogisticRegression(penalty="none", random_state=0, max_iter=1e4)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="New Test"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
from xgboost import XGBClassifier
features = list(vif_statistic.feature)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_threshold_train_data = train_data.copy()
log_threshold_train_data[log_features] = log_threshold_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_threshold_test_data = test_data.copy()
log_threshold_test_data[log_features] = log_threshold_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    log_threshold_train_data[f] = (
        log_threshold_train_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
    log_threshold_test_data[f] = (
        log_threshold_test_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_threshold_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_threshold_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_threshold_test_data.loc[:, features]), columns=features)
te = pd.concat([log_threshold_test_data[["target"]], te_features], axis=1)
model = XGBClassifier(max_depth=6, learning_rate=.3)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="XGBoost + Scale + New VIF Log BCB + Threshold Time + Negative Dif"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
from sklearn.model_selection import cross_validate
grid = pd.DataFrame(
    data=[
        (max_depth, reg_alpha, learning_rate, subsample)
        for max_depth in np.arange(1, 1.01, 1)
        for reg_alpha in np.arange(0, 1.01, .2)
        for learning_rate in np.arange(.3, .301, .1)
        for subsample in np.arange(1, 1.01, .1)
    ],
    columns=("max_depth", "reg_alpha", "learning_rate", "subsample")
)
all_scores = {}
for i in range(len(grid.max_depth)):
    model = XGBClassifier(
        max_depth=int(grid.max_depth[i]),
        learning_rate=grid.learning_rate[i],
        reg_alpha=grid.reg_alpha[i],
        subsample=grid.subsample[i],
    )
    new_scores = sk.model_selection.cross_validate(
        model,
        X=tr.drop("target", axis=1).to_numpy(),
        y=tr.target.to_numpy(),
        scoring='balanced_accuracy',
        cv=5,
        return_train_score=True
    )
    all_scores[i] = [new_scores["train_score"].mean(), new_scores["test_score"].mean()]
    sys.stdout.write('\r' + f"Cross-validation process: {i + 1}/{len(grid.max_depth)} ({(i + 1)/len(grid.max_depth):.0%})")
```
```
train_test_cv = (
    pd.DataFrame(all_scores)
    .transpose()
    .set_axis(["train", "test"], axis=1)
    .sort_values("train", ascending=False)
)
selected_params = grid.loc[train_test_cv.sort_values("test", ascending=False).index[0], :]
selected_max_depth = selected_params.max_depth
selected_reg_alpha = selected_params.reg_alpha
selected_learning_rate = selected_params.learning_rate
selected_subsample = selected_params.subsample
plt.plot(list(train_test_cv.train), color = 'cornflowerblue', linewidth = 2)
plt.plot(list(train_test_cv.test), color = 'crimson', linewidth = 2)
plt.show()
```
```
from xgboost import XGBClassifier
features = list(vif_statistic.feature)
log_features = [c for c in features if bool(re.search("^dummy|^cnae|^dif|^target|score_hpj8|^tempo", c)) is False]
log_threshold_train_data = train_data.copy()
log_threshold_train_data[log_features] = log_threshold_train_data[log_features].applymap(lambda x: math.log(x + 1.01))
log_threshold_test_data = test_data.copy()
log_threshold_test_data[log_features] = log_threshold_test_data[log_features].applymap(lambda x: math.log(x + 1.01))
threshold_features = [c for c in features if bool(re.search("^tempo", c))]
for f in threshold_features:
    log_threshold_train_data[f] = (
        log_threshold_train_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
    log_threshold_test_data[f] = (
        log_threshold_test_data[f]
        .apply(
            lambda x: x if np.percentile(log_threshold_train_data[f], 90) > x
            else np.percentile(log_threshold_train_data[f], 90)
        )
    )
scaler = sk.preprocessing.StandardScaler()
scaler.fit(log_train_data.loc[:, features])
tr_features = pd.DataFrame(scaler.transform(log_threshold_train_data.loc[:, features]), columns=features)
tr = pd.concat([log_threshold_train_data[["target"]], tr_features], axis=1)
te_features = pd.DataFrame(scaler.transform(log_threshold_test_data.loc[:, features]), columns=features)
te = pd.concat([log_threshold_test_data[["target"]], te_features], axis=1)
model = XGBClassifier(
    max_depth=int(selected_max_depth),
    learning_rate=selected_learning_rate,
    reg_alpha=selected_reg_alpha,
    subsample=selected_subsample
)
model.fit(tr.drop("target", axis=1).to_numpy(), tr.target.to_numpy())
```
```
predicted_default = model.predict_proba(te[features].to_numpy())
agg_metrics = evaluate_metrics(
    y_real=te.target.to_numpy(),
    y_proba=predicted_default[:, 1],
    agg_metrics_df=agg_metrics,
    model_name="CV XGBoost + Scale + New VIF Log BCB + Threshold Time + Negative Dif"
)
agg_metrics.sort_values("f1_score", ascending=False)
```
```
explainer = shap.Explainer(model, tr.drop("target", axis=1))
shap_values = explainer(tr.drop("target", axis=1))
shap.plots.bar(shap_values, max_display=25)
```
```
shap_df = pd.DataFrame(shap_values.values, columns=tr.drop("target", axis=1).columns)
shap_df = (
    pd.DataFrame(shap_df.abs().sum(axis=0), columns=["shap"])
    .reset_index()
    .sort_values("shap", ascending=False)
    .rename({"index": "feature"}, axis=1)
    .reset_index(drop=True)
)
shap_df.head(5)
```
```
shap.plots.beeswarm(shap_values)
```
```
organized_imp_tr = tr.loc[:, ["target"] + list(shap_df.feature)]
from sklearn.model_selection import cross_validate
grid = pd.DataFrame(
    data=[
        (max_depth, reg_alpha, n_features)
        for max_depth in np.arange(1, 2.01, 1)
        for reg_alpha in np.arange(0, 1.01, .2)
        for n_features in (range(5, len(organized_imp_tr.columns) - 9))
    ],
    columns=("max_depth", "reg_alpha", "n_features")
)
all_scores = {}
for i in range(len(grid.max_depth)):
    model = XGBClassifier(
        max_depth=int(grid.max_depth[i]),
        learning_rate=selected_learning_rate,
        reg_alpha=grid.reg_alpha[i],
        subsample=selected_subsample,
    )
    new_scores = sk.model_selection.cross_validate(
        model,
        X=organized_imp_tr.drop("target", axis=1).iloc[:, :grid.n_features[i]].to_numpy(),
        y=organized_imp_tr.target.to_numpy(),
        scoring='balanced_accuracy',
        cv=5,
        return_train_score=True
    )
    all_scores[i] = [new_scores["train_score"].mean(), new_scores["test_score"].mean()]
    sys.stdout.write('\r' + f"Cross-validation process: {i + 1}/{len(grid.max_depth)} ({(i + 1)/len(grid.max_depth):.0%})")
```
```
train_test_cv = (
    pd.DataFrame(all_scores)
    .transpose()
    .set_axis(["train", "test"], axis=1)
    .sort_values("train", ascending=False)
)
selected_params = grid.loc[train_test_cv.sort_values("test", ascending=False).index[0], :]
selected_max_depth = selected_params.max_depth
selected_reg_alpha = selected_params.reg_alpha
selected_n_features = selected_params.n_features
plt.plot(list(train_test_cv.train), color = 'cornflowerblue', linewidth = 2)
plt.plot(list(train_test_cv.test), color = 'crimson', linewidth = 2)
plt.show()
```
