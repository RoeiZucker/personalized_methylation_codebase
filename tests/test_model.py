import json
from pathlib import Path

import numpy as np
import pytest


def test_tiny_model_training_and_prediction(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, Regex
    from methylation import model

    tokenizer_path = tmp_path / "tokenizer"
    base = Tokenizer(models.WordLevel({"[PAD]":0, "[UNK]":1, "[CLS]":2, "AAAAAA":3}, unk_token="[UNK]"))
    base.pre_tokenizer = pre_tokenizers.Split(Regex(".{6}"), behavior="isolated")
    base.post_processor = processors.TemplateProcessing(single="[CLS] $A", special_tokens=[("[CLS]", 2)])
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=base, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]")
    tokenizer.save_pretrained(tokenizer_path)
    config = transformers.BertConfig(vocab_size=4, hidden_size=12, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=1024, num_labels=3)
    transformers.BertForTokenClassification(config).save_pretrained(tokenizer_path)
    monkeypatch.setattr(model, "MODEL", str(tokenizer_path))
    folder = tmp_path / "reference"
    labels = [-100] + [i%3 for i in range(900)]
    dataset = datasets.Dataset.from_dict({"input_ids":[[2]+[3]*900], "attention_mask":[[1]*901], "labels":[labels],
        "window_id":["chr1:0-5400"], "start":[0], "end":[5400]})
    dataset.save_to_disk(folder / "TRAIN")
    (folder / "study.json").write_text(json.dumps({"donor":"reference", "mode":"reference"}))
    model.train([folder/"TRAIN"], tmp_path/"model", "donor-excluded", "target", model=str(tokenizer_path), batch_size=1, epochs=1)
    model.predict(folder/"TRAIN", tmp_path/"model/epoch-1", "target", "T", tmp_path/"prediction.csv", batch_size=1)
    import pandas as pd
    result = pd.read_csv(tmp_path/"prediction.csv")
    assert len(result) == 900
    assert result.genomic_position.iloc[[0,-1]].tolist() == [0, 5394]
    np.testing.assert_allclose(result.filter(like="probability_class_").sum(axis=1), 1, atol=1e-6)
