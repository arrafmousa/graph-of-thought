from pathlib import Path

from libs.config import ConfigLoader
from libs.training import SentimentFinetuner

_REPO = Path(__file__).resolve().parents[3]


def _finetuner() -> SentimentFinetuner:
    return SentimentFinetuner(
        model_id="distilbert-base-uncased",
        model_revision="main",
        dataset_id="nyu-mll/glue",
        dataset_revision="main",
        dataset_config="sst2",
        text_field="sentence",
        label_field="label",
        num_labels=2,
        train_samples=100,
        eval_samples=100,
        num_epochs=1,
        batch_size=16,
        learning_rate=5e-5,
        max_length=128,
        precision="fp16",
        seed=42,
    )


def test_constructs_without_heavy_deps():
    # Module must import and construct on machines without torch/transformers.
    assert _finetuner() is not None


def test_train_config_matches_schema():
    loader = ConfigLoader(_REPO / "configs" / "schema" / "train_config.schema.json")
    config = loader.load(_REPO / "configs" / "train_sst2.json")
    t = config["training"]
    assert t["model_id"] == "distilbert-base-uncased"
    assert t["dataset_id"] == "nyu-mll/glue" and t["dataset_config"] == "sst2"
    assert t["train_samples"] == 100
    assert config["run"]["dashboard_template"] == "training"
