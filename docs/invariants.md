# Invariants

These are assumptions the current pipeline relies on.

- `num_train_epoch` is expected to stay fixed after configs are created.
- Saved checkpoints are expected to follow the `epoch-*` naming pattern.
- Eval configs are meant to describe the full set of checkpoints for one training path.
- Retrain and eval directories are expected to map cleanly to one workflow run.
- `no_pretraining` is treated as a retrain variant, not as true pretraining.
- LoRA is a variant inside the main workflows, not a separate top-level pipeline.

If one of these stops being true, parts of `config_manager.py` may behave incorrectly even if the code still runs.
