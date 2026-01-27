import json
import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from typing import Optional

from src.data_generation import build_publish


class TestBuildPublish(TestCase):
    def test_combines_without_duplicates_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            data_generation_dir = "data/generated"
            combined_data_dir = "data/combined"
            lang = "en"
            split = "train"

            sources = {
                "source_a": [
                    {"id": "1", "note": "a1"},
                    {"id": "2", "note": "a2"},
                    {"id": "4", "note": "a4"},
                ],
                "source_b": [
                    {"id": "2", "note": "b2"},
                    {"id": "3", "note": "b3"},
                ],
            }

            for source, entries in sources.items():
                data_dir = os.path.join(
                    tmp_root,
                    data_generation_dir,
                    source,
                    "keep",
                    lang,
                )
                os.makedirs(data_dir, exist_ok=True)
                data_path = os.path.join(data_dir, f"{split}.jsonl")
                with open(data_path, "w", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")

            build_publish.logistics = SimpleNamespace(
                project_root_dir=tmp_root,
                data_generation_dir=data_generation_dir,
                combined_data_dir=combined_data_dir,
                hf_datatset_ids={"source_a": "hf_a", "source_b": "hf_b"},
            )
            build_publish.local_dirs = SimpleNamespace(
                source_a=os.path.join(tmp_root, "media", "source_a"),
                source_b=os.path.join(tmp_root, "media", "source_b"),
            )

            def fake_load_hf_dataset(hf_id: str, split: str, config_name: Optional[str] = None):
                return [
                    {"id": "1", "label": 0},
                    {"id": "2", "label": 1},
                    {"id": "3", "label": 0},
                ]

            with patch("src.data_generation.build_publish.load_hf_dataset", side_effect=fake_load_hf_dataset):
                build_publish.build_create_combined(langs=[lang], splits=[split])

            out_path = os.path.join(
                tmp_root,
                combined_data_dir,
                lang,
                f"{split}.jsonl",
            )
            with open(out_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            by_id = {record["id"]: record for record in records}
            self.assertEqual(sorted(by_id.keys()), ["1", "2", "3"])
            self.assertEqual(by_id["2"]["note"], "b2")
            self.assertEqual(by_id["2"]["source"], "source_b")
            self.assertEqual(by_id["2"]["label"], 1)
