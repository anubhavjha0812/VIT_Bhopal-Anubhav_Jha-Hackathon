# E40 owner steps

The code is ready. The Laya model has not been downloaded or run here. There are no E40 accuracy results yet. E23 is seen design data, and all reference labels are AI-made without a human check.

First finish the target map at `private_notes/annotation_uploads/task1/e40_inputs/8k_targets.csv`. The 8-K snapshot has no target-company column. The draft uses the filing issuer where there is exactly one; eight multi-issuer rows remain blank. Claude or the owner must confirm those targets before the job. Do not pick a target from model output or change labels. The script refuses any blank, missing or duplicate target before installing/loading Laya. Record the finished map's SHA-256 and do not change it after prediction.

1. Create a **private Kaggle dataset** named `e40-laya-inputs` with these files, preserving their names:

   - `data/outputs/e25_credit_8k_full_rows.csv`
   - `data/outputs/e24_e23_gate_v6_rows.csv`
   - the finished `8k_targets.csv`
   - `kaggle/e40_laya_baseline.py`

2. Create a Kaggle notebook, attach that dataset, enable a T4 GPU and Internet, and run this single cell once:

   ```python
   !python /kaggle/input/e40-laya-inputs/e40_laya_baseline.py --credit8k /kaggle/input/e40-laya-inputs/e25_credit_8k_full_rows.csv --e23 /kaggle/input/e40-laya-inputs/e24_e23_gate_v6_rows.csv --targets /kaggle/input/e40-laya-inputs/8k_targets.csv
   ```

   If Kaggle shows a different dataset mount, replace only `/kaggle/input/e40-laya-inputs/` with the path shown in its Input panel. The script installs `laya==0.4.11`, sets `USE_TF=0`, and loads exactly `laya.load("convaiinnovations/laya", subfolder="typed-decisions", device="cuda")`. Loading errors stop the job; do not switch checkpoints. The package pin was checked on [PyPI](https://pypi.org/project/laya/0.4.11/); the API was checked in the [official SDK](https://github.com/NandhaKishorM/laya) and [model card](https://huggingface.co/convaiinnovations/laya).

3. Download these two files from `/kaggle/working/` and put them in the local repo's `data/outputs/`:

   - `e40_laya_raw.csv` (3,180 rows: 530 headlines × three questions × two forms)
   - `e40_laya_raw_manifest.json` (actual package/checkpoint hashes, input hashes and timing)

4. Keep the same finished local target map and script. On the laptop, run the evaluator **once**, after both downloads exist:

   ```bash
   rtk proxy env UV_CACHE_DIR=/tmp/e39-e40-uv-cache UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m riskengine.eval.e40_decision_model --run
   ```

   It writes `data/outputs/e40_decision_model.json` and a persistent run lock. Keep the lock. Do not rerun a scored job or change the questions/threshold after results. On a failed or partial Kaggle job, preserve its output and report the error before deciding on recovery; do not silently start another scored run.

The script reads only `item_id,text` from 8-K, `headline_id,headline,target_company` from E23, and `item_id,target_company` from the target map. Only the headline is passed as model state; the target fills the fixed Q2 text. Labels and previous predictions never enter the model. CSV call time is repeated on Q1/Q2/Q3; count it once per headline/form. No warm-up headlines are scored; timing includes the first call and excludes package installation, checkpoint loading and hashing.

Probabilities are raw option probabilities with all temperatures set to 1, no language override and no fitted calibration. The SDK rounds its public output to four decimals; for noul, P(no) is 1−P(yes). Choice uses neutral option IDs A=yes/B=no with the same question wording. Only noul chooses the single threshold on 8-K and controls promotion; choice uses that same cutoff for reporting. Outbound Internet sockets are blocked during inference to demonstrate offline operation after download. Nothing is adopted from E23: even a pass permits only a separately registered unseen-set test.
