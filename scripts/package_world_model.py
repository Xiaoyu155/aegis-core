"""Package the shipped inference checkpoint and a bounded replay anchor set."""
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil

from aegis_core.world_model import CausalWorldModel, STATE_LABELS, encoder_digest
from scripts.train_and_eval_world_model import ROOT, build_corpus, split_by_group


def main():
    source = ROOT / "artifacts" / "causal_world_model"
    target = ROOT / "aegis_core" / "data"
    target.mkdir(exist_ok=True)
    for name in ("model_weights.json", "feature_encoder.json"):
        shutil.copyfile(source / name, target / name)
    model = CausalWorldModel.load(source)
    examples, provenance = build_corpus(ROOT)
    train, validation, _ = split_by_group(examples)
    rng = random.Random(20261118)
    training_anchors = []
    for label in STATE_LABELS:
        rows = [i for i in train if examples[i]["label"] == label]
        rng.shuffle(rows)
        training_anchors.extend(rows[:24])

    def anchors(indices):
        result = []
        for index in indices:
            row = examples[index]
            encoded = model.encoder.encode(row["state"], row["action"], row.get("context"),
                                           row.get("prior_attempts", []))
            result.append({"features": encoded, "label": row["label"], "reward": row["reward"],
                           "high_risk": row["high_risk"]})
        return result

    pack = {"schema_version": "aegis.replay_anchors.v1", "encoder_sha256": encoder_digest(model.encoder),
            "source_weights_sha256": hashlib.sha256((source / "model_weights.json").read_bytes()).hexdigest(),
            "source_files": provenance["source_files"], "training": anchors(training_anchors),
            "validation": anchors(validation[:96])}
    data = json.dumps(pack, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with (target / "replay_anchors.json.gz").open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as archive:
            archive.write(data)


if __name__ == "__main__":
    main()
