import math
import os
import sys
import types
from pathlib import Path

import pytest
import yaml

try:
    from datasets import Dataset
except ModuleNotFoundError:
    import json

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows

        @classmethod
        def from_dict(cls, data):
            keys = list(data.keys())
            rows = []
            for i in range(len(data[keys[0]])):
                rows.append({key: data[key][i] for key in keys})
            return cls(rows)

        def save_to_disk(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / 'data.json').write_text(json.dumps(self._rows), encoding='utf-8')

        def select(self, indices):
            return _FakeDataset([self._rows[i] for i in indices])

        def filter(self, fn):
            return _FakeDataset([row for row in self._rows if fn(row)])

        def map(self, fn, batched=False):
            if not batched:
                return _FakeDataset([fn(row) for row in self._rows])
            batch = {key: [row[key] for row in self._rows] for key in self.column_names}
            updates = fn(batch)
            new_rows = []
            for idx, row in enumerate(self._rows):
                updated = dict(row)
                for key, values in updates.items():
                    updated[key] = values[idx]
                new_rows.append(updated)
            return _FakeDataset(new_rows)

        @property
        def column_names(self):
            return list(self._rows[0].keys()) if self._rows else []

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

        def __iter__(self):
            return iter(self._rows)

    class _FakeSequence:
        def __init__(self, feature):
            self.feature = feature

    class _FakeValue:
        def __init__(self, dtype):
            self.dtype = dtype

    def _fake_load_from_disk(path, keep_in_memory=False):
        rows = json.loads((Path(path) / 'data.json').read_text(encoding='utf-8'))
        return _FakeDataset(rows)

    def _fake_concatenate_datasets(datasets):
        rows = []
        for dataset in datasets:
            rows.extend(list(dataset))
        return _FakeDataset(rows)

    fake_datasets = types.ModuleType('datasets')
    fake_datasets.Dataset = _FakeDataset
    fake_datasets.DatasetDict = dict
    fake_datasets.Sequence = _FakeSequence
    fake_datasets.Value = _FakeValue
    fake_datasets.concatenate_datasets = _fake_concatenate_datasets
    fake_datasets.load_dataset = lambda *args, **kwargs: None
    fake_datasets.load_from_disk = _fake_load_from_disk
    sys.modules['datasets'] = fake_datasets
    Dataset = _FakeDataset

sys.path.insert(
    0,
    os.path.abspath('/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code'),
)
sys.path.insert(
    0,
    os.path.abspath('/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src'),
)

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    fake_transformers = types.ModuleType('transformers')

    class _DummyHFObject:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _DummyTrainer:
        def evaluate(self):
            return {'mse': 0.0}

        def predict(self, *args, **kwargs):
            return types.SimpleNamespace(predictions=[])

        def train(self, *args, **kwargs):
            return None

    fake_transformers.AutoTokenizer = _DummyHFObject
    fake_transformers.AutoModel = _DummyHFObject
    fake_transformers.AutoModelForTokenClassification = _DummyHFObject
    fake_transformers.TrainingArguments = _DummyHFObject
    fake_transformers.Trainer = _DummyTrainer
    fake_transformers.DataCollatorForTokenClassification = _DummyHFObject
    sys.modules['transformers'] = fake_transformers

try:
    import peft  # noqa: F401
except ModuleNotFoundError:
    fake_peft = types.ModuleType('peft')
    fake_peft.LoraConfig = type('LoraConfig', (), {})
    fake_peft.TaskType = type('TaskType', (), {})
    fake_peft.get_peft_model = lambda *args, **kwargs: None
    sys.modules['peft'] = fake_peft

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules['torch'] = types.ModuleType('torch')

class _TrainerStub:
    def evaluate(self):
        return {'mse': 0.0}

    def predict(self, *args, **kwargs):
        return types.SimpleNamespace(predictions=[])

    def train(self, *args, **kwargs):
        return None

def _model_stub(*args, **kwargs):
    return types.SimpleNamespace(eval=lambda: None)

fake_trainer_utils = types.ModuleType('src.utils.trainer_utils')
fake_trainer_utils.get_compute_func = lambda *args, **kwargs: None
fake_trainer_utils.get_trainer = lambda *args, **kwargs: _TrainerStub()
fake_trainer_utils.get_trainer_type = lambda *args, **kwargs: None
sys.modules.setdefault('src.utils.trainer_utils', fake_trainer_utils)
sys.modules.setdefault('utils.trainer_utils', fake_trainer_utils)

fake_model_utils = types.ModuleType('src.utils.model_utils')
fake_model_utils.get_base_model = _model_stub
fake_model_utils.get_fine_tuned_model = _model_stub
sys.modules.setdefault('src.utils.model_utils', fake_model_utils)
sys.modules.setdefault('utils.model_utils', fake_model_utils)

fake_tissue_model_utils = types.ModuleType('src.utils.tissue_model_utils')
fake_tissue_model_utils.get_base_model = _model_stub
fake_tissue_model_utils.get_fine_tuned_model = _model_stub
sys.modules.setdefault('src.utils.tissue_model_utils', fake_tissue_model_utils)
sys.modules.setdefault('utils.tissue_model_utils', fake_tissue_model_utils)

fake_tissue_trainer_utils = types.ModuleType('src.utils.tissue_trainer_utils')
fake_tissue_trainer_utils.get_trainer = lambda *args, **kwargs: _TrainerStub()
sys.modules.setdefault('src.utils.tissue_trainer_utils', fake_tissue_trainer_utils)
sys.modules.setdefault('utils.tissue_trainer_utils', fake_tissue_trainer_utils)

if 'evaluate' not in sys.modules:
    fake_evaluate = types.ModuleType('evaluate')

    class _Metric:
        def __init__(self, name):
            self.name = name

        def compute(self, predictions, references, average=None):
            predictions = list(predictions)
            references = list(references)
            if self.name == 'mse':
                if not predictions:
                    return {'mse': 0.0}
                return {'mse': sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions)}
            if self.name == 'mae':
                if not predictions:
                    return {'mae': 0.0}
                return {'mae': sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)}
            if self.name == 'pearsonr':
                return {'pearsonr': 1.0}
            return {'value': 0.0}

    fake_evaluate.load = lambda name, *args, **kwargs: _Metric(name)
    sys.modules['evaluate'] = fake_evaluate

from src.atlas_evaluation_creator_script import create_atlas_eval_configs
from src.config_manager import create_project_config
from src.evaluator import evaluate_checkpoint, perform_evaluation
from src.training import get_datasets
from src.utils.label_transform_utils import transform_label_values


def test_transform_label_values_preserves_masks_and_applies_log1p():
    values = [-100.0, 0.0, math.e - 1.0, 1.0]
    transformed = transform_label_values(values, 'log1p')

    assert transformed[0] == -100.0
    assert transformed[1] == 0.0
    assert transformed[2] == pytest.approx(1.0)
    assert transformed[3] == pytest.approx(math.log1p(1.0))


def test_training_dataset_load_applies_label_transform(tmp_path):
    train_dataset = Dataset.from_dict({
        'input_ids': [[1, 2]],
        'labels': [[0.0, 1.0]],
    })
    eval_dataset = Dataset.from_dict({
        'input_ids': [[3, 4]],
        'labels': [[-100.0, math.e - 1.0]],
    })
    train_path = tmp_path / 'train_dataset'
    eval_path = tmp_path / 'eval_dataset'
    train_dataset.save_to_disk(str(train_path))
    eval_dataset.save_to_disk(str(eval_path))

    transformed_eval, transformed_train = get_datasets(
        str(train_path),
        str(eval_path),
        top_rows=-1,
        load_dataset_to_memory=False,
        label_transform='log1p',
    )
    untouched_eval, untouched_train = get_datasets(
        str(train_path),
        str(eval_path),
        top_rows=-1,
        load_dataset_to_memory=False,
        label_transform='none',
    )

    assert transformed_train[0]['labels'][1] == pytest.approx(math.log1p(1.0))
    assert transformed_eval[0]['labels'][0] == -100.0
    assert transformed_eval[0]['labels'][1] == pytest.approx(1.0)
    assert untouched_train[0]['labels'] == [0.0, 1.0]
    assert untouched_eval[0]['labels'] == [-100.0, math.e - 1.0]


def test_perform_evaluation_applies_label_transform_on_loaded_dataset(tmp_path, monkeypatch):
    dataset = Dataset.from_dict({
        'input_ids': [[1, 2]],
        'labels': [[0.0, math.e - 1.0]],
        'window_id': ['chr1:0-12'],
        'start': [0],
    })
    dataset_path = tmp_path / 'eval_dataset'
    dataset.save_to_disk(str(dataset_path))

    checkpoint_path = tmp_path / 'checkpoint-1'
    checkpoint_path.mkdir()

    captured = {}

    def fake_evaluate_checkpoint(*args, **kwargs):
        captured['dataset'] = args[4]
        captured['label_transform'] = kwargs.get('label_transform')
        return {'mse': 0.0}

    monkeypatch.setattr('src.evaluator.evaluate_checkpoint', fake_evaluate_checkpoint)

    cfg = {
        'verbose': False,
        'paths': {
            'dataset_path': str(dataset_path),
            'model_path': str(checkpoint_path),
        },
        'task': {
            'sub_task': 'evaluate_single_checkpoint',
            'top_rows': -1,
            'use_variant_file': False,
            'label_transform': 'log1p',
        },
        'model': {
            'model_repo': 'repo',
            'model_name': 'model',
            'model_type': 'regression_analysis',
            'num_labels': 1,
        },
        'testing_params': {
            'test_mode': False,
        },
    }

    perform_evaluation(cfg)

    assert captured['dataset'][0]['labels'][1] == pytest.approx(1.0)
    assert captured['label_transform'] == 'log1p'


def test_variant_evaluation_prediction_decoding_depends_on_label_transform(monkeypatch):
    class Prediction:
        def __init__(self, value):
            self.predictions = [
                [[value], [value]],
                [[value], [value]],
            ]

    dataset = Dataset.from_dict({
        'window_id': ['window_1', 'window_2'],
        'labels': [[-100.0, 0.0], [-100.0, 0.0]],
        'input_ids': [[1, 2], [3, 4]],
        'start': [0, 0],
    })
    relevant_bins = {'window_1': {'bin': [0]}, 'window_2': {'bin': [0]}}
    dataset_labels_none = {'window_1': {'bin': [0.0]}, 'window_2': {'bin': [0.0]}}
    dataset_labels_log1p = {'window_1': {'bin': [0.5]}, 'window_2': {'bin': [0.5]}}

    monkeypatch.setattr('src.evaluator.predict_checkpoint', lambda *args, **kwargs: Prediction(0.0))

    score_none = evaluate_checkpoint(
        'repo',
        'model',
        False,
        1,
        dataset,
        'regression_analysis',
        'checkpoint',
        'regression_analysis',
        relevant_bins,
        True,
        'bins',
        ['bin'],
        dataset_labels_none,
        False,
        label_transform='none',
    )
    score_log1p = evaluate_checkpoint(
        'repo',
        'model',
        False,
        1,
        dataset,
        'regression_analysis',
        'checkpoint',
        'regression_analysis',
        relevant_bins,
        True,
        'bins',
        ['bin'],
        dataset_labels_log1p,
        False,
        label_transform='log1p',
    )

    assert score_none[0][2]['mse'] == pytest.approx(0.0)
    assert score_log1p[0][2]['mse'] == pytest.approx(0.0)


def test_project_config_propagates_label_transform_and_atlas_configs_ignore_it(tmp_path):
    created_configs_path = tmp_path / 'generated_configs'
    dataset_base_dir = tmp_path / 'datasets'
    base_model_location = tmp_path / 'models'
    dataset_base_dir.mkdir()
    base_model_location.mkdir()
    pretrain_dir = base_model_location / 'A_pretrain_unit_label_transform'
    retrain_dir = base_model_location / 'A_epoch-1-step-1_retrain_unit_label_transform'
    pretrain_dir.mkdir(parents=True)
    retrain_dir.mkdir(parents=True)
    (pretrain_dir / 'epoch-1-step-1').mkdir()
    (retrain_dir / 'epoch-1-step-1').mkdir()

    create_project_config(
        project_suffix='_unit_label_transform',
        bigwig_files=['/tmp/A.bigwig', '/tmp/B.bigwig'],
        names=['A', 'B'],
        created_configs_path=str(created_configs_path),
        tokenizer_name='InstaDeepAI/nucleotide-transformer-500m-1000g',
        dataset_base_dir=str(dataset_base_dir),
        base_model_location=str(base_model_location),
        model_type='regression_analysis',
        use_lora=False,
        freeze_model=False,
        num_labels=1,
        load_best_model_at_end=False,
        num_train_epoch=1,
        num_pretrain_epoch=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-6,
        metric_for_best_model='mse',
        save_stratagy='epoch',
        number_of_steps=10,
        save_total_limit=1,
        add_epoch_end_save_callback=False,
        save_at_end=False,
        continue_from_last=False,
        use_variant_filtering=False,
        variant_filtering_upper_bound=-1,
        variant_filtering_lower_bound=-1,
        chromosomes=['chr1'],
        seq_size=600,
        number_of_bins=5,
        test_size=0.2,
        load_dataset_to_memory=False,
        override_dataset=True,
        label_transform='log1p',
    )

    pretrain_config_path = next((created_configs_path / 'pretrain_training').glob('*.yaml'))
    eval_config_path = next((created_configs_path / 'eval_configs').glob('*.yaml'))
    pretrain_config = yaml.safe_load(pretrain_config_path.read_text())
    eval_config = yaml.safe_load(eval_config_path.read_text())

    assert pretrain_config['task']['label_transform'] == 'log1p'
    assert eval_config['task']['label_transform'] == 'log1p'

    project_config_path = tmp_path / 'project.yaml'
    project_config = {
        'params': {
            'names': ['A', 'B'],
            'bigwig_files': ['/data/A.bw', '/data/B.bw'],
            'created_configs_path': str(created_configs_path),
            'dataset_base_dir': str(dataset_base_dir),
            'project_suffix': 'atlas_project',
            'base_suffix': 'atlas_project',
            'tokenizer_name': 'repo/model',
            'model_type': 'regression_analysis',
            'seq_size': 600,
            'number_of_bins': 4,
            'chromosomes': ['chr1'],
            'label_transform': 'log1p',
        }
    }
    project_config_path.write_text(yaml.safe_dump(project_config), encoding='utf-8')

    written_paths = create_atlas_eval_configs(str(project_config_path))
    atlas_config = yaml.safe_load(Path(written_paths[0]).read_text(encoding='utf-8'))
    assert 'label_transform' not in atlas_config.get('task', {})
