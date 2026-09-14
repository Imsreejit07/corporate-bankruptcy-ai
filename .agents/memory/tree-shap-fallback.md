---
name: TreeSHAP fallback
description: Runtime-specific guidance for explaining XGBoost predictions when the standalone SHAP package is unavailable.
---

Use XGBoost's native `pred_contribs` output as the explanation path when the standalone SHAP package cannot be resolved for the active Python runtime. It returns the model-native TreeSHAP feature contributions plus a final bias term; exclude only that final bias term from the per-feature list.

**Why:** The project runtime could load the imported XGBoost and calibrated scikit-learn artifacts, but the package registry did not resolve standalone SHAP for Python 3.13. Native XGBoost contributions avoid retraining or approximating the original model.

**How to apply:** Keep the model's feature order unchanged, sort returned feature contributions by absolute magnitude, and describe their direction as model behavior rather than causal financial effects.