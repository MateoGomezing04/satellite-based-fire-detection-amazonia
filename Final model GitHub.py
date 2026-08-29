from tqdm import tqdm
import os
import gc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from joblib import parallel_backend           
from scipy.stats import loguniform, randint

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==========================================================
# 1. SETTINGS
# ==========================================================

DATA_FILE  = r"C:\Users\User\Desktop\Modelo incendios\focos_sin_agua.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_FILE), "name" )
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ITERATIONS     = 10
TEST_SIZE        = 0.30
RANDOM_SEED_BASE = 100
N_SAMPLE_CLASS   = 200000
NYSTROEM_COMPONENTS = 100

#FIGS]
FIG_DPI = 300

features = [
    "T7",
    "T14",
  #  "BTD",
   # "Ratio",
    #"ND",
    "Cloud_Weight",
    "LST_BT_DIFF",
    "TPW",
    "Cloud_Probabilities",
    "fire_hour"
]

# ==========================================================
# 2. PROCESSING
# ==========================================================

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
df = df.dropna(subset=features + ["label"])

X = df[features].astype(np.float32).reset_index(drop=True)
y = df["label"].astype(int).reset_index(drop=True)

sample_weights_full = df["Cloud_Weight"].values.flatten() ** 2
sample_weights_full = sample_weights_full / np.mean(sample_weights_full)
w_series = pd.Series(sample_weights_full)

del df
gc.collect()                                                    

print(f"\nDataset completo: {len(X)} filas")
print(f"Distribución:\n{y.value_counts()}\n")

# ==========================================================
# 2.1  BOXPLOTS - features for class
# ==========================================================
# A subsample is taken to avoid loading the entire dataset (~770k rows) into the plot.
# This does not affect training, only visualization.

n_box_sample = min(50000, len(X))
box_idx = X.sample(n=n_box_sample, random_state=RANDOM_SEED_BASE).index
X_box = X.loc[box_idx].copy()
X_box["label"] = y.loc[box_idx].map({0: "No Fire", 1: "Fire"})

ncols = 3
nrows = int(np.ceil(len(features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()
for idx, feat in enumerate(features):
    X_box.boxplot(column=feat, by="label", ax=axes[idx], grid=False)
    axes[idx].set_xlabel("Class")
    axes[idx].set_ylabel(feat)
    axes[idx].set_title("")         
for idx in range(len(features), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle("")                    
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplots_features_by_class.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.show()
plt.close()
print("Figure saved: boxplots_features_by_class.png "
      "— Feature distributions by class (Fire vs No Fire)")

# ==========================================================
# 2.2 CORRELATION MATRIX between features [FIGURE]
# ==========================================================

corr_matrix = X[features].corr(method="pearson")
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, label="Pearson correlation coefficient")
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(features)), features)
for i in range(len(features)):
    for j in range(len(features)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8, color="black")
plt.xlabel("Feature")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.show()
plt.close()
print("Figure saved: correlation_matrix.png "
      "— Pearson correlation matrix among input features")

del X_box, box_idx
gc.collect()

# ==========================================================
# 3. ITERATIONS
# ==========================================================

svm_metrics, rf_metrics   = [], []
svm_thresholds, rf_thresholds = [], []

for i in tqdm(range(N_ITERATIONS), desc="Entrenando modelos"):

    # ----------------------------------------------------------
    # Iterative stratified subsampling
    # ----------------------------------------------------------
    idx_0      = y[y == 0].sample(n=N_SAMPLE_CLASS, random_state=RANDOM_SEED_BASE + i).index
    idx_1      = y[y == 1].sample(n=N_SAMPLE_CLASS, random_state=RANDOM_SEED_BASE + i).index
    idx_sample = idx_0.union(idx_1)

    X_iter = X.loc[idx_sample]
    y_iter = y.loc[idx_sample]
    w_iter = w_series.loc[idx_sample].values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_iter, y_iter, w_iter,
        test_size=TEST_SIZE,
        stratify=y_iter,
        random_state=RANDOM_SEED_BASE + i
    )

    del X_iter, y_iter, w_iter
    gc.collect()                                                  

    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=RANDOM_SEED_BASE + i)

    # ==========================
    # SVM (Pipeline)
    # ==========================

    svm_pipeline = Pipeline([
        ('scaler',   StandardScaler()),
        ('nystroem', Nystroem(kernel='rbf',
                              n_components=NYSTROEM_COMPONENTS,
                              random_state=RANDOM_SEED_BASE + i)),
        ('svm',      LinearSVC(max_iter=3000))
    ])

    param_grid_svm = {
        "svm__C":                 loguniform(1e-3, 1e3),
        "nystroem__gamma":        loguniform(1e-4, 1e1),
        "nystroem__n_components": [50, 100, 150],
    }

    svm_search = RandomizedSearchCV(
        svm_pipeline,
        param_distributions=param_grid_svm,
        n_iter=30,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED_BASE + i
    )

    with parallel_backend("threading", n_jobs=-1):
        svm_search.fit(X_train, y_train, svm__sample_weight=w_train)  

    svm_model = svm_search.best_estimator_

    scores_svm_raw = svm_model.decision_function(X_test)
    y_prob_svm = (
        (scores_svm_raw - scores_svm_raw.min()) /
        (scores_svm_raw.max() - scores_svm_raw.min() + 1e-12)
    )

    prec_s, rec_s, thresh_s = precision_recall_curve(y_test, y_prob_svm)
    f1_vals_s = np.where(
        (prec_s + rec_s) == 0, 0,
        2 * prec_s * rec_s / (prec_s + rec_s)
    )
    t_svm = 0.5
    svm_thresholds.append(t_svm)

    y_pred_svm = (y_prob_svm >= t_svm).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_svm).ravel()

    svm_metrics.append({
        "accuracy":          accuracy_score(y_test, y_pred_svm),
        "precision":         precision_score(y_test, y_pred_svm, zero_division=0),
        "recall":            recall_score(y_test, y_pred_svm, zero_division=0),
        "f1":                f1_score(y_test, y_pred_svm, zero_division=0),
        "auc":               roc_auc_score(y_test, y_prob_svm),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_svm),
        "specificity":       TN / (TN + FP) if (TN + FP) else 0,
        "fpr":               FP / (FP + TN) if (FP + TN) else 0,
        "mcc":               matthews_corrcoef(y_test, y_pred_svm)
    })

    # ==========================
    # RANDOM FOREST
    # ==========================

    rf_param_dist = {                                               
        "n_estimators":      randint(50, 200),                      
        "max_depth" : [10,20,30,None],                        
        "min_samples_split": randint(2, 20),
        "min_samples_leaf":  randint(1, 10),
        "max_features":      ["sqrt", "log2"],                      
        "bootstrap":         [True],                                  
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(
                               random_state=RANDOM_SEED_BASE + i),
        param_distributions=rf_param_dist,
        n_iter=30,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED_BASE + i
    )

    with parallel_backend("threading", n_jobs=-1):                 
        rf_search.fit(X_train, y_train, sample_weight=w_train)

    rf = rf_search.best_estimator_

    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    prec_r, rec_r, thresh_r = precision_recall_curve(y_test, y_prob_rf)
    f1_vals_r = np.where(
        (prec_r + rec_r) == 0, 0,
        2 * prec_r * rec_r / (prec_r + rec_r)
    )
    t_rf = 0.5
    rf_thresholds.append(t_rf)

    y_pred_rf = (y_prob_rf >= t_rf).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_rf).ravel()

    rf_metrics.append({
        "accuracy":          accuracy_score(y_test, y_pred_rf),
        "precision":         precision_score(y_test, y_pred_rf, zero_division=0),
        "recall":            recall_score(y_test, y_pred_rf, zero_division=0),
        "f1":                f1_score(y_test, y_pred_rf, zero_division=0),
        "auc":               roc_auc_score(y_test, y_prob_rf),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_rf),
        "specificity":       TN / (TN + FP) if (TN + FP) else 0,
        "fpr":               FP / (FP + TN) if (FP + TN) else 0,
        "mcc":               matthews_corrcoef(y_test, y_pred_rf)
    })

    # Preserve objects from the last iteration
    if i == N_ITERATIONS - 1:
        svm_last        = svm_model
        rf_last         = rf
        X_test_last     = X_test
        y_test_last     = y_test
        y_prob_last_svm = y_prob_svm
        y_prob_last_rf  = y_prob_rf
        thresh_s_last   = thresh_s
        f1_vals_s_last  = f1_vals_s
        thresh_r_last   = thresh_r
        f1_vals_r_last  = f1_vals_r

# ----------------------------------------------------------
# Explicit memory release between iterations   
 # ----------------------------------------------------------
    if i < N_ITERATIONS - 1:
        del svm_search, rf_search, svm_model, rf, svm_pipeline
        del X_train, X_test, y_train, y_test, w_train, w_test
        del scores_svm_raw, y_prob_svm, y_prob_rf
        del prec_s, rec_s, thresh_s, f1_vals_s
        del prec_r, rec_r, thresh_r, f1_vals_r
        del y_pred_svm, y_pred_rf
        gc.collect()

# ==========================================================
# 4. METRICS SUMMARY
#===========================================================

gc.collect()                                                      

def summarize(metrics, thresholds, name):
    df_m = pd.DataFrame(metrics)
    df_m.to_csv(os.path.join(OUTPUT_DIR, f"{name}_iterations.csv"), index=False)

    summary = {}
    for col in df_m.columns:
        summary[f"{col}_mean"]  = df_m[col].mean()
        summary[f"{col}_p2_5"]  = np.percentile(df_m[col], 2.5)
        summary[f"{col}_p97_5"] = np.percentile(df_m[col], 97.5)

    summary["threshold_mean"] = np.mean(thresholds)
    summary["threshold_std"]  = np.std(thresholds)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_summary.csv"), index=False)
    return df_m, summary_df

svm_df, svm_summary = summarize(svm_metrics, svm_thresholds, "svm")
rf_df,  rf_summary  = summarize(rf_metrics,  rf_thresholds,  "rf")

print("\nSVM:\n", svm_summary.T)
print("\nRF:\n",  rf_summary.T)

# ==========================================================
# 5. METRIC DISTRIBUTION GRAPHS
# ==========================================================

def plot_metrics(svm_df, rf_df, output_dir):
    metrics = svm_df.columns.tolist()
    ncols   = 3
    nrows   = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    for idx, m in enumerate(metrics):
        axes[idx].hist(svm_df[m], alpha=0.6, label='SVM', bins=15, color='steelblue')
        axes[idx].hist(rf_df[m],  alpha=0.6, label='RF',  bins=15, color='darkorange')
        # 🆕 [FIGURA] Sin título; el nombre de la métrica queda en el eje X
        axes[idx].set_xlabel(m, fontsize=8)
        axes[idx].set_ylabel("Frequency", fontsize=8)
        axes[idx].legend(fontsize=8)
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"),
                dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Figure saved: metrics_comparison.png "
          "— Distribution of performance metrics across iterations (SVM vs RF)")

plot_metrics(svm_df, rf_df, OUTPUT_DIR)

# ==========================================================
# 6.ROC Y PR CURVES
# ==========================================================

def plot_curves(y_test, y_prob, name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color="steelblue")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_roc.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Figure saved: {name}_roc.png — ROC curve ({name.upper()})")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(rec, prec, color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_pr.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Figure saved: {name}_pr.png — Precision-Recall curve ({name.upper()})")

plot_curves(y_test_last, y_prob_last_svm, "svm")
plot_curves(y_test_last, y_prob_last_rf,  "rf")

# ==========================================================
# 6.2 CONFUSION MATRICES
# ==========================================================

for name, y_pred in [("svm", (y_prob_last_svm >= 0.5).astype(int)),
                     ("rf",  (y_prob_last_rf >= 0.5).astype(int))]:

    cm = confusion_matrix(y_test_last, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Fire", "Fire"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"),
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    print(f"Figure saved: {name}_confusion_matrix.png — Confusion matrix")
    

# ==========================================================
#6.3 THRESHOLD VS F1-SCORE
# ==========================================================

plt.figure(figsize=(7, 5))
plt.plot(thresh_s_last, f1_vals_s_last[:-1], label="SVM", color="steelblue")
plt.plot(thresh_r_last, f1_vals_r_last[:-1], label="Random Forest", color="darkorange")
plt.xlabel("Decision Threshold")
plt.ylabel("F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "threshold_vs_f1.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.show()
plt.close()
print("Figure saved: threshold_vs_f1.png "
      "— F1-score as a function of the decision threshold (SVM vs RF)")

# ==========================================================
# 7. IMPORTANCE OF VARIABLES
# ==========================================================

N_PERM_SAMPLES = min(500, len(X_test_last))
X_perm, y_perm = resample(
    X_test_last, y_test_last,
    n_samples=N_PERM_SAMPLES,
    stratify=y_test_last,
    random_state=RANDOM_SEED_BASE
)

perm_svm = permutation_importance(
    svm_last, X_perm, y_perm,
    n_repeats=5,
    n_jobs=-1                                                     
)

pd.DataFrame({"feature": features, "importance": perm_svm.importances_mean}) \
  .to_csv(os.path.join(OUTPUT_DIR, "svm_importance.csv"), index=False)

pd.DataFrame({"feature": features, "importance": rf_last.feature_importances_}) \
  .to_csv(os.path.join(OUTPUT_DIR, "rf_importance.csv"), index=False)

# ==========================================================
# 7.1 🆕 [FIGURE] BARPLOT - Importance of variables (RF and SVM)
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rf_importances = rf_last.feature_importances_
rf_order = np.argsort(rf_importances)[::-1]
axes[0].bar(np.array(features)[rf_order], rf_importances[rf_order], color="darkorange")
axes[0].set_xlabel("Feature")
axes[0].set_ylabel("Gini Importance (Random Forest)")
axes[0].tick_params(axis="x", rotation=45)

svm_importances = perm_svm.importances_mean
svm_order = np.argsort(svm_importances)[::-1]
axes[1].bar(np.array(features)[svm_order], svm_importances[svm_order], color="steelblue")
axes[1].set_xlabel("Feature")
axes[1].set_ylabel("Permutation Importance (SVM)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_barplot.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.show()
plt.close()
print("Figure saved: feature_importance_barplot.png "
      "— Feature importance comparison (RF Gini importance vs SVM permutation importance)")

# ==========================================================
# 7.5 🆕 [FIGURE] SHAP - Random Forest model explainability
# ==========================================================
# IMPORTANT NOTE:
# This SHAP figure is calculated only for Random Forest, using
# shap.TreeExplainer (fast and accurate for tree models).

# SHAP is NOT calculated for the SVM because:
# 1) LinearSVC does not provide probabilities and does not have an efficient explainer,
# 2) the pipeline uses a kernel approximation (Nystroem), so a
# KernelExplainer would be valid in theory but extremely costly in
# time/RAM for this dataset (exactly what this script seeks to avoid).

# If SHAP is needed for the SVM at any point, you can use
# shap.KernelExplainer(svm_last.decision_function, X_perm_small) with a
# very small subsample (<100 rows), accepting a high computational cost.
if SHAP_AVAILABLE:
    explainer = shap.TreeExplainer(rf_last)
    shap_raw = explainer.shap_values(X_perm)

    # Compatibilidad entre versiones de shap (list vs array 3D vs array 2D)
    if isinstance(shap_raw, list):
        shap_vals_fire = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_vals_fire = shap_raw[:, :, 1]
    else:
        shap_vals_fire = shap_raw

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals_fire, X_perm, feature_names=features, show=False)
    plt.xlabel("SHAP value (impact on model output for the 'Fire' class)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_rf.png"),
                dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Figure saved: shap_summary_rf.png "
          "— SHAP summary plot for Random Forest feature contributions")
else:
    print("SHAP figure skipped: the 'shap' package is not installed. "
          "Run 'pip install shap' and re-run this section to generate it.")

# ==========================================================
# 8. FINAL COMPARISON
# ==========================================================

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"],
    "SVM": [svm_summary[f"{m}_mean"][0]
            for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]],
    "RF":  [rf_summary[f"{m}_mean"][0]
            for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]]
})
comparison.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False)

# ==========================================================
# 8.1 [FIGURE] BARPLOT - Comparison of performance metrics
# ==========================================================
x = np.arange(len(comparison))
width = 0.35


print(comparison.dtypes)
print(comparison)
plt.figure(figsize=(10,6))

bars1 = plt.bar(
    x-width/2,
    comparison["SVM"],
    width,
    label="SVM",
    color="steelblue"
)

print(comparison["SVM"].values)
print(comparison["RF"].values)

bars2 = plt.bar(
    x+width/2,
    comparison["RF"],
    width,
    label="Random Forest",
    color="darkorange"
)

for bar in bars1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

for bar in bars2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xticks(x, comparison["Metric"])
plt.xlabel("Performance Metric")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison_barplot.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.show()
plt.close()
print("Figure saved: metrics_comparison_barplot.png "
      "— Bar plot comparing mean SVM vs RF performance metrics")

# ==========================================================
# 9. CONFIGURATION TABLE
# ==========================================================

config_data = []

svm_params      = svm_last.named_steps['svm'].get_params()
nystroem_params = svm_last.named_steps['nystroem'].get_params()

config_data.append({
    "Model":                 "SVM",
    "Kernel":                "RBF (aproximado via Nystroem)",
    "C":                     svm_params.get("C"),
    "Gamma":                 nystroem_params.get("gamma"),
    "N_Components_Nystroem": nystroem_params.get("n_components"),
    "Classifier":            "LinearSVC",
    "Class_Weight":          svm_params.get("class_weight"),
    "Scaler":                "StandardScaler (dentro de Pipeline)",
    "Threshold_Mean":        np.mean(svm_thresholds),
    "Threshold_Std":         np.std(svm_thresholds),
    "CV_Folds":              5,
    "Optimization_Metric":   "F1-score",
    "Iterations":            N_ITERATIONS,
    "Test_Size":             TEST_SIZE,
    "N_Sample_Class":        N_SAMPLE_CLASS,
    "CV_Backend":            "threading",                        
})

rf_params_cfg = rf_last.get_params()
config_data.append({
    "Model":             "Random Forest",
    "Kernel":            "N/A",
    "C":                 "N/A",
    "Gamma":             "N/A",
    "Class_Weight":      rf_params_cfg.get("class_weight"),
    "Scaler":            "No",
    "Threshold_Mean":    np.mean(rf_thresholds),
    "Threshold_Std":     np.std(rf_thresholds),
    "n_estimators":      rf_params_cfg.get("n_estimators"),
    "max_features":      rf_params_cfg.get("max_features"),
    "max_depth":         rf_params_cfg.get("max_depth"),
    "min_samples_split": rf_params_cfg.get("min_samples_split"),
    "min_samples_leaf":  rf_params_cfg.get("min_samples_leaf"),
    "bootstrap":         rf_params_cfg.get("bootstrap"),
    "Iterations":        N_ITERATIONS,
    "Test_Size":         TEST_SIZE,
    "N_Sample_Class":    N_SAMPLE_CLASS,
    "CV_Backend":        "threading",                             
})

config_data.append({
    "Model":                  "General",
    "Features":               ", ".join(features),
    "Label":                  "0 = No incendio, 1 = Incendio",
    "Dataset_Size":           len(X),
    "N_Sample_Class":         N_SAMPLE_CLASS,
    "Train_Test_Split":       f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}",
    "Validation":             "StratifiedKFold",
    "Threshold_Optimization": "Max F1-score via precision_recall_curve",
    "Metrics":                "Accuracy, Precision, Recall, F1, AUC, Balanced Accuracy, Specificity, FPR, MCC",
})

pd.DataFrame(config_data).to_csv(
    os.path.join(OUTPUT_DIR, "model_configuration_table.csv"), index=False
)
print("\nTabla de configuración exportada.")

# ==========================================================
# 10. SAVE MODELS
# ==========================================================

model_dir = os.path.join(OUTPUT_DIR, "saved_models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(svm_last, os.path.join(model_dir, "svm_pipeline.pkl"))
joblib.dump(rf_last,  os.path.join(model_dir, "rf_model.pkl"))
print(confusion_matrix(y_test, y_pred_rf))
print("\nModelos guardados en:", model_dir)
print("\nPROCESO COMPLETO FINALIZADO")