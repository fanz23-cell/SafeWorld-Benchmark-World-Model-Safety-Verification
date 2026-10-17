# SafeWorld-Benchmark

**A robotics benchmark and experimental pipeline for the SafeWorld world-model safety verification research.**

This repository builds the benchmark and experiments around the SafeWorld method proposed by our advisor. It turns temporal safety requirements into executable tasks, collects robot trajectories in simulation, trains a DreamerV3-style latent world model, and connects the model outputs to temporal-property verifiers.

Our contribution here is the benchmark and its experimental realization: task design and grounding, trajectory and dataset generation, world-model training, and verifier/baseline integration. The mathematical verification method is advisor-led work. This repository should be read as research infrastructure and a pilot study, not as a claim that every specification has a complete formal proof or final paper-scale evaluation.

## What the benchmark asks

Instead of measuring only reward or task completion, the benchmark asks whether an embodied agent's behavior follows rules over time. Example requirements include:

- Stay out of hazard regions.
- Reach a goal while remaining safe.
- Visit regions in a required order.
- Slow down after approaching an obstacle.
- Revisit one or more patrol regions.
- Combine navigation, safety, and response requirements into a longer mission.

The task layer defines atomic propositions (APs) such as `hazard`, `goal`, `fast`, and `near_obs`, extracts them from simulator state, and evaluates the resulting traces. A run records the verdict and, when applicable, the first violation step.

## What is in this repository

| Component | What it does | Current scope |
|---|---|---|
| Task benchmark | Task registry, environment grounding, AP extraction, finite-trace evaluators, rollout and result logging | 23 task instances across four Safety Gymnasium environments and eight complexity levels; 20 are runnable by default and three remain placeholders. Runnable tasks span seven of the eight levels. |
| Goal2 data pipeline | Scripted, task-aware controllers; episode collection; bucketed manifests; dataset exporters | Collects success, perturbed, and failure/recovery trajectories for `SafetyPointGoal2-v0`. The `near_success` bucket is a perturbation category and is not a guarantee that every episode nearly succeeded. |
| Latent world model | PyTorch DreamerV3-style RSSM with observation and reward prediction plus safety-related auxiliary heads | Predicts cost, speed, goal distance, nearest hazard distance, nearest vase distance, and human distance from latent features. The human-distance target is derived from a fixed virtual position. |
| SafeWorld verifier and baselines | Robustness monitoring, transfer calibration, legacy LPPM path, and action-level MPC baseline runners | Produces pilot results for the APs supported by the current model. MPC and shielding results are empirical compliance measurements, not formal warrants. |
| VeriWorld method draft | Newer certificate-based approach for temporal properties | The manuscript draft describes co-Büchi/LPPM and Büchi/LBSM certificates with a retention function. The current implementation does not yet contain the full LBSM and retention-certificate pipeline. |

The 23 task instances in the benchmark are environment/task configurations, not 23 distinct formulas. The task layer contains 15 distinct formula strings. Its evaluators are task-specific finite-trace checks; they are not a general-purpose STL/LTL parser or an infinite-horizon proof engine.

## Experimental snapshot

The repository includes saved Goal2 pilot outputs under [`external/SafeWorld`](external/SafeWorld/):

| Saved run | Recorded result | How to interpret it |
|---|---|---|
| SafeWorld verifier | 23 specification rows: 2 `WARRANT`, 5 `STL_MARGIN`, 5 `VIOLATION`, 11 `N/A` | 10 rows were marked supported, 2 partial, and 11 unsupported by the model's AP heads. `N/A` means the current model cannot evaluate a required signal; it does not mean the robot failed the task. |
| Dreamer-style action baselines | SafeDreamer-style MPC and shielding MPC on the supported subset | These are local model-predictive approximations built from the available RSSM checkpoint setup, not official SafeDreamer training or formal shield synthesis. Their rates are empirical compliance, not verification guarantees. |

The comparison note records 30 rollouts of up to 150 steps with one seed. Treat these files as diagnostic pilot evidence: many higher-level specifications are unavailable because the model lacks zone and carrying prediction heads, and the human-related rows use a synthetic distance signal. The stored baseline table does not establish that SafeWorld outperforms the baselines.

The verifier's Goal2 oracle protocol encodes offline observations through the RSSM posterior. Results therefore use recorded environment observations; they are not produced solely by free-running latent imagination without environment data.

The checked-in `datasets/goal2_master` snapshot contains 376 JSON files and 58,500 serialized environment steps. Content-based checks find about 140 distinct trajectories, so the file count must not be interpreted as 376 independent episodes. Training NPZ exports and model checkpoints are not included in this checkout.

## Quick start

Run one runnable benchmark task:

```bash
python scripts/run_single_task.py --task_id E1_L1_HazardAvoid --seed 0
```

Run a level suite:

```bash
python scripts/run_level_suite.py --level 1 2 --seed 0
```

These task-layer commands use random actions by default and require the Safety Gymnasium environment dependencies. They test the task, labeling, and logging pipeline; they do not run the world-model verifier.

Generate a Goal2 pilot dataset (requires the environment dependencies):

```bash
python scripts/generate_goal2_master_dataset.py --plan pilot --output-root datasets/goal2_pilot --allow-partial
```

The world-model trainer is [`train_world_model.py`](train_world_model.py). It expects Dreamer-style replay data in the configured data directory; this checkout contains episode JSONs but not the exported training NPZs or trained checkpoint. The verifier scripts in [`external/SafeWorld`](external/SafeWorld/) also use machine-specific model and data paths that must be configured before running elsewhere.

## Project map

- [`benchmark/`](benchmark/): task definitions, AP extraction, evaluators, rollout runners, and result serialization.
- [`data_generation/`](data_generation/): Goal2 scripted controllers, episode collection, subset export, and Dreamer replay adaptation.
- [`training/dreamer_world_model/`](training/dreamer_world_model/): RSSM world model, auxiliary safety heads, and offline trainer.
- [`external/SafeWorld/`](external/SafeWorld/): verifier integration, experiment runners, saved pilot outputs, and comparison analysis.
- [`_ICLR_27_VeriWorld/main.md`](_ICLR_27_VeriWorld/main.md): current method draft available in this checkout. It contains the certificate-based method section; a newer benchmark/experiment manuscript section is not present here.

## Demo

This checkout does not yet include an up-to-date SafeWorld demo video.
