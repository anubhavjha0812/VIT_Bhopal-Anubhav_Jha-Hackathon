# E44b: owner-run Kaggle evaluation

Status: plan fixed, not scored. No prediction or scoring has run in this implementation task.

1. From the repository, make the private upload locally. This only checks hashes and zips files:

   ```bash
   rtk proxy env UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python notebooks/e44_score/prepare_bundle.py --output /tmp/e44_kaggle_bundle.zip
   ```

2. Upload `/tmp/e44_kaggle_bundle.zip` as a **private** Kaggle Dataset. Add it to a new private notebook. Import `notebooks/e44_score/e44_score.ipynb`. Set `BUNDLE` in its setup cell to the uploaded ZIP path. Keep the notebook and outputs private: the bundle includes annotation files and hidden provenance. Source data's recorded non-commercial licence still applies.
3. Use Python 3.12 or newer with compatible Linux wheels, CPU execution, Internet **off**. The frozen inference provider is CPU; choosing a Kaggle GPU does not change it. Required package versions are recorded in the plan and the ZIP's `e44_runtime_requirements.txt`: numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, onnxruntime 1.30.0, torch 2.14.1+cpu, transformers 5.18.0. If Kaggle lacks these exact packages, attach a private offline wheelhouse containing these versions and their dependency wheels, built for that Kaggle Python/Linux version. Set `WHEELS` to its directory. No package or model is downloaded by this notebook. A missing version stops the run; do not substitute a version or loosen the check.
4. Check the paths and environment cells first. Before executing the final cell, confirm this evaluation has not been run in any previous Kaggle session. Execute that cell **once**. It calls `python -m riskengine.eval.e44_score --predict`, then `--score`. It never trains. The frozen IDs are read from the reference's id column only; no label reaches inference.
5. Save the notebook version and download the entire `/kaggle/working/e44_repo/data/outputs/` folder: `e44_predictions.csv`, `e44_results.json`, `e44.run_lock`, `e44.run_lock.score_claim`, and `e44_scoring_plan.lock`. Also retain `e44_execution_manifest.json`, which binds the uploaded bundle and notebook execution to their hashes. Preserve all files even if a stage fails. Existing locks/outputs refuse a repeat; do not clear them, create another session, or rerun to change a number.

The ZIP includes only selected frozen E44 annotation files, code, plan, guide/protocol, model assets and the historical v2 ONNX integrity pin. It excludes the full raw news, prior result tables, secrets and unrelated private notes. `bundle_paths` is the explicit upload list. The preparation script never parses any headline or annotation, and checks all frozen input hashes before packaging.

The two phases share one run record. An extra exclusive `e44.run_lock.score_claim` prevents concurrent scoring claims. Hashes expose changes; a filesystem lock cannot prevent a person from deleting artifacts or opening a fresh notebook. This is a procedural one-run constraint too.

Results compare A (reference only), C, C-noPF and E25-v3. A retains the actual E24 credit-rule-plus-model baseline. C-noPF only restores Product news on C's `model_product_to_other` route. All twelve gold classes have at least ten rows; Default has 28. AI-only agreement labels and the exclusion of 78 disagreements limit the conclusion. A qualifying challenger is reported to the owner; the running system remains C.
