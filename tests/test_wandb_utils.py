import os
import sys
import types


class DummyHFObject:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def _install_trainer_stubs():
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")
        fake_torch.load = lambda *args, **kwargs: None
        fake_torch.save = lambda *args, **kwargs: None
        sys.modules["torch"] = fake_torch

    fake_nn = types.ModuleType("torch.nn")
    fake_functional = types.ModuleType("torch.nn.functional")
    fake_functional.mse_loss = lambda *args, **kwargs: None
    sys.modules.setdefault("torch.nn", fake_nn)
    sys.modules.setdefault("torch.nn.functional", fake_functional)

    if "transformers" not in sys.modules:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = DummyHFObject
        fake_transformers.AutoModel = DummyHFObject
        fake_transformers.AutoModelForTokenClassification = DummyHFObject
        fake_transformers.DataCollatorForTokenClassification = DummyHFObject
        fake_transformers.Trainer = type("Trainer", (), {})
        fake_transformers.TrainerCallback = type("TrainerCallback", (), {})
        fake_transformers.TrainingArguments = TrainingArgumentsStub
        sys.modules["transformers"] = fake_transformers

    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    fake_modeling_utils.unwrap_model = lambda model: model
    sys.modules.setdefault("transformers.modeling_utils", fake_modeling_utils)

    fake_trainer_module = types.ModuleType("transformers.trainer")
    fake_trainer_module.TRAINING_ARGS_NAME = "training_args.bin"
    sys.modules.setdefault("transformers.trainer", fake_trainer_module)

    if "evaluate" not in sys.modules:
        fake_evaluate = types.ModuleType("evaluate")

        class MetricStub:
            def compute(self, predictions, references, average=None):
                return {"pearsonr": 1.0, "mse": 0.0, "mae": 0.0}

        fake_evaluate.load = lambda *args, **kwargs: MetricStub()
        sys.modules["evaluate"] = fake_evaluate


class TrainingArgumentsStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


_install_trainer_stubs()

from src.utils import tissue_trainer_utils, trainer_utils
from src.utils.trainer_utils import WandbConfigCallback
from src.utils.wandb_utils import (
    build_wandb_metadata,
    get_wandb_report_to,
    prepare_wandb_environment,
)




class TrainerCapture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.args = kwargs["args"]

    def add_callback(self, callback):
        return None


def _trainer(module, monkeypatch, wandb_config=None, wandb_metadata=None):
    monkeypatch.setattr(module, "TrainingArguments", TrainingArgumentsStub)
    monkeypatch.setattr(module, "get_trainer_type", lambda *args, **kwargs: TrainerCapture)
    monkeypatch.setattr(module, "get_compute_func", lambda *args, **kwargs: None)
    return module.get_trainer(
        dataset=[],
        model=object(),
        tokenizer=object(),
        model_type="classification_analysis",
        eval_mode=False,
        save_stratagy="no",
        add_epoch_end_prediction=False,
        wandb_config=wandb_config,
        wandb_metadata=wandb_metadata,
    )

def _training_args(
    module,
    monkeypatch,
    report_to="none",
    run_name=None,
    save_stratagy="no",
    number_of_steps=-1,
    eval_steps=None,
    logging_steps=None,
    has_eval_dataset=False,
):
    monkeypatch.setattr(module, "TrainingArguments", TrainingArgumentsStub)
    return module.get_training_args(
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=4,
        num_train_epochs=1,
        load_best_at_end=False,
        output_dir="/tmp/model",
        metric_for_best_model=None,
        eval_mode=False,
        save_stratagy=save_stratagy,
        number_of_steps=number_of_steps,
        save_total_limit=-1,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        has_eval_dataset=has_eval_dataset,
        report_to=report_to,
        run_name=run_name,
    )


def test_wandb_disabled_reports_to_none():
    assert get_wandb_report_to({}) == "none"
    assert get_wandb_report_to({"enabled": False}) == "none"


def test_wandb_disabled_does_not_set_dir(monkeypatch):
    monkeypatch.delenv("WANDB_DIR", raising=False)

    prepare_wandb_environment({"enabled": False, "dir": "/tmp/unused"}, {"dir": "/tmp/unused"})

    assert "WANDB_DIR" not in os.environ


def test_wandb_enabled_metadata_defaults_and_env(monkeypatch):
    for name in ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_NAME", "WANDB_RUN_GROUP", "WANDB_MODE", "WANDB_TAGS", "WANDB_DIR"]:
        monkeypatch.delenv(name, raising=False)

    wandb_config = {
        "enabled": True,
        "project": "personalized-methylation",
        "mode": "online",
        "dir": "/tmp/wandb-runs",
    }
    task_config = {"task_type": "cpg_training", "analysis_name": "run_a"}
    model_config = {"model_name": "nt", "model_type": "classification_analysis", "use_lora": True}

    metadata = build_wandb_metadata(wandb_config, task_config, model_config)
    prepare_wandb_environment(wandb_config, metadata)

    assert get_wandb_report_to(wandb_config) == "wandb"
    assert metadata["name"] == "run_a"
    assert metadata["group"] == "cpg_training_nt"
    assert metadata["tags"] == ["cpg_training", "classification_analysis", "lora"]
    assert os.environ["WANDB_PROJECT"] == "personalized-methylation"
    assert os.environ["WANDB_NAME"] == "run_a"
    assert os.environ["WANDB_RUN_GROUP"] == "cpg_training_nt"
    assert os.environ["WANDB_MODE"] == "online"
    assert os.environ["WANDB_TAGS"] == "cpg_training,classification_analysis,lora"
    assert os.environ["WANDB_DIR"] == "/tmp/wandb-runs"


def test_trainer_utils_keep_wandb_disabled_by_default(monkeypatch):
    args = _training_args(trainer_utils, monkeypatch)

    assert args.report_to == "none"
    assert args.run_name is None


def test_trainer_utils_accept_wandb_reporting(monkeypatch):
    args = _training_args(trainer_utils, monkeypatch, report_to="wandb", run_name="run_a")

    assert args.report_to == "wandb"
    assert args.run_name == "run_a"


def test_tissue_trainer_utils_keep_wandb_disabled_by_default(monkeypatch):
    args = _training_args(tissue_trainer_utils, monkeypatch)

    assert args.report_to == "none"
    assert args.run_name is None


def test_tissue_trainer_utils_accept_wandb_reporting(monkeypatch):
    args = _training_args(tissue_trainer_utils, monkeypatch, report_to="wandb", run_name="run_a")

    assert args.report_to == "wandb"
    assert args.run_name == "run_a"


def test_training_args_split_save_eval_and_logging_steps(monkeypatch):
    args = _training_args(
        trainer_utils,
        monkeypatch,
        save_stratagy="steps",
        number_of_steps=10000,
        eval_steps=1000,
        logging_steps=500,
        has_eval_dataset=True,
    )

    assert args.save_steps == 10000
    assert args.eval_steps == 1000
    assert args.logging_steps == 500
    assert args.logging_strategy == "steps"


def test_training_args_eval_steps_falls_back_to_number_of_steps(monkeypatch):
    args = _training_args(
        trainer_utils,
        monkeypatch,
        save_stratagy="steps",
        number_of_steps=10000,
        has_eval_dataset=True,
    )

    assert args.save_steps == 10000
    assert args.eval_steps == 10000


def test_get_trainer_ignores_run_name_when_wandb_disabled(monkeypatch):
    trainer = _trainer(
        trainer_utils,
        monkeypatch,
        wandb_config={"enabled": False},
        wandb_metadata={"name": "should_not_be_used"},
    )

    assert trainer.args.report_to == "none"
    assert trainer.args.run_name is None


def test_get_trainer_uses_run_name_when_wandb_enabled(monkeypatch):
    trainer = _trainer(
        trainer_utils,
        monkeypatch,
        wandb_config={"enabled": True},
        wandb_metadata={"name": "run_a"},
    )

    assert trainer.args.report_to == "wandb"
    assert trainer.args.run_name == "run_a"


def test_wandb_config_callback_updates_active_run(monkeypatch):
    updates = []

    class ConfigCapture:
        def update(self, config, allow_val_change=False):
            updates.append((config, allow_val_change))

    fake_wandb = types.SimpleNamespace(run=object(), config=ConfigCapture())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    callback = WandbConfigCallback({"train": {"number_of_steps": 10000}})
    callback.on_train_begin(None, None, None)

    assert updates == [({"train": {"number_of_steps": 10000}}, True)]
