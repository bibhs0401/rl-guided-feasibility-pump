# RL-Guided Feasibility Pump (Option A)

This project trains a PPO policy to choose perturbation size `k` **only when the Feasibility Pump (FP) is stalled**.

- FP runs normally between decision points.
- At each stall event, PPO selects an action index mapped to a `k` value from `k_choices`.
- Variable selection remains heuristic (top-`k` by current FP ranking).

## Environment Interface (Current)

- **Action space**: `Discrete(len(k_choices))`
- **Default `k_choices`**: `1,2,5,10,20,50`
- **Observation**: compact 10D dynamic FP state (no static instance features in v1)

## Training

PowerShell example:

```powershell
python .\train_ppo.py `
  --instances ".\data\train\*.npz" `
  --runs-dir ".\runs\train_ppo" `
  --run-name "mmp_option_a_seed10" `
  --total-timesteps 200000 `
  --max-iterations 100 `
  --stall-threshold 3 `
  --k-choices "1,2,5,10,20,50" `
  --time-limit 30 `
  --cplex-threads 1 `
  --device auto `
  --progress-log-steps 1000
```

Each training run is saved in its own folder:

- `runs/train_ppo/<run_name>/model.zip`
- `runs/train_ppo/<run_name>/learning_curve.csv`
- `runs/train_ppo/<run_name>/run_args.json`
- `runs/train_ppo/<run_name>/run_summary.json`
- `runs/train_ppo/<run_name>/run_summary.csv`

If `--run-name` is omitted, a timestamped name is generated automatically.
You can still pass `--save-path` and `--curve-csv` to override default output locations.

## Evaluation

Use the same `k_choices` mapping used during training:

```powershell
python .\evaluate_ppo.py `
  --model-path ".\runs\train_ppo\mmp_option_a_seed10\model.zip" `
  --instances ".\data\test\*.npz" `
  --runs-dir ".\runs\eval_ppo" `
  --run-name "mmp_option_a_seed10_test" `
  --max-iterations 100 `
  --stall-threshold 3 `
  --k-choices "1,2,5,10,20,50" `
  --time-limit 30 `
  --cplex-threads 1 `
  --device auto
```

Each evaluation run is saved in its own folder:

- `runs/eval_ppo/<run_name>/eval_instances.csv`
- `runs/eval_ppo/<run_name>/eval_summary.csv`

If `--run-name` is omitted, a timestamped name is generated automatically.
You can still pass `--per-instance-csv` and `--summary-csv` to override default output locations.

## Important Notes

- `--k-choices` in evaluation must match the action mapping used by the trained model.
- If model action count and `k_choices` length differ, evaluation will fail fast with a validation error.
- This code requires IBM CPLEX/DOcplex available in your Python environment for FP solves.
