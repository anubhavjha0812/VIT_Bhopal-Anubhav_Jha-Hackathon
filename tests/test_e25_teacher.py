"""E25 teacher labelling script: prompt, letter map, probabilities, and the input guard.
Only the pure functions are tested; no model is loaded."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskengine.eval.e24_gate import CLASSES
from riskengine.eval.leaks import norm
from riskengine.paths import DATA, ROOT

SCRIPT = ROOT / "notebooks" / "e25_teacher" / "label_events.py"
spec = importlib.util.spec_from_file_location("label_events", SCRIPT)
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)


def test_classes_and_letters_match_the_e24_taxonomy():
    assert le.CLASSES == CLASSES
    assert len(le.LETTERS) == len(le.CLASSES) == 12


def test_prompt_lists_every_letter_and_holds_the_headline():
    prompt = le.build_prompt('Acme "unveils" a van')
    for letter in le.LETTERS:
        assert re.search(rf"^{letter}\. ", prompt, re.MULTILINE)
    assert "Acme 'unveils' a van" in prompt  # double quotes would break the quoted headline
    assert prompt.rstrip().endswith("Answer with one letter, A to L.")


def test_prompt_examples_are_not_from_any_test_set():
    examples = re.findall(r'^"(.+)" -> [A-L]$', le.INSTRUCTIONS, re.MULTILINE)
    assert len(examples) >= 12
    tests = pd.read_csv(DATA / "gold/gold_to_label.csv", dtype=str)["headline"].tolist()
    tests += pd.read_csv(DATA / "gold/credit/credit_to_label.csv", dtype=str)["headline"].tolist()
    seen = {norm(t) for t in tests}
    assert not [e for e in examples if norm(e) in seen]


def test_letter_probs_rows_sum_to_one_and_keep_the_top_letter():
    scores = np.array([[0.0] * 11 + [5.0], [9.0] + [0.0] * 11])
    p = le.letter_probs(scores)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0].argmax() == 11 and p[1].argmax() == 0


@pytest.mark.parametrize("name", ["D2_to_label.csv", "final_news_to_label.csv", "x_d2.csv.gz"])
def test_refuses_d2_and_final_files(name):
    with pytest.raises(SystemExit):
        le.check_input_name(Path(name))


@pytest.mark.parametrize("name", ["D1_to_label.csv", "pool.csv.gz"])
def test_accepts_d1_and_pool_files(name):
    le.check_input_name(Path(name))


def test_batch_order_is_fixed_and_sorted_by_length():
    texts = ["bbb", "a", "cc", "dd", "e"]
    assert le.order(texts) == [1, 4, 2, 3, 0]
    assert le.order(texts) == le.order(list(texts))


def test_prompt_hash_is_stable_and_versioned():
    assert le.prompt_sha256() == le.prompt_sha256()
    assert le.PROMPT_VERSION == "v2"


def test_default_teachers_are_two_and_llama_is_only_a_backup():
    assert le.DEFAULT_TEACHERS == ["qwen3_8b", "mistral_nemo"]
    assert "llama31_8b" in le.TEACHERS and "llama31_8b" not in le.DEFAULT_TEACHERS
    assert le.TEACHERS["mistral_nemo"]["four_bit"] is True


def test_a_failing_teacher_is_recorded_and_does_not_raise(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError("out of memory")

    monkeypatch.setattr(le, "_run_teacher", boom)
    le.run_teacher("mistral_nemo", le.TEACHERS["mistral_nemo"], ["a"], ["text"], tmp_path, "dry")
    failed = tmp_path / "dry_mistral_nemo_FAILED.json"
    assert failed.exists() and "out of memory" in failed.read_text()


def test_stages_point_at_d1_pool_and_extra_only():
    assert set(le.STAGES) == {"dry", "d1", "pool", "extra"}
    assert le.STAGE in le.STAGES
    for stage in le.STAGES.values():
        le.check_input_name(Path(stage["file"]))  # none is a D2 or final file
    assert le.STAGES["dry"]["limit"] == 500 and le.STAGES["d1"]["limit"] is None


def test_find_input_finds_a_nested_file_and_fails_loudly_otherwise(tmp_path):
    (tmp_path / "ds" / "sub").mkdir(parents=True)
    (tmp_path / "ds" / "sub" / "pool.csv.gz").write_text("x")
    assert le.find_input("pool.csv.gz", (str(tmp_path),)).name == "pool.csv.gz"
    with pytest.raises(SystemExit):
        le.find_input("D1_to_label.csv", (str(tmp_path),))


def test_find_input_accepts_an_unpacked_gz_and_any_case(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "Pool.CSV").write_text("x")  # Kaggle unpacked pool.csv.gz
    assert le.find_input("pool.csv.gz", (str(tmp_path),)).name == "Pool.CSV"


def test_find_input_error_lists_the_files_it_saw(tmp_path):
    (tmp_path / "other.txt").write_text("x")
    with pytest.raises(SystemExit) as err:
        le.find_input("pool.csv.gz", (str(tmp_path),))
    assert "other.txt" in str(err.value) and "Add Input" in str(err.value)


def test_shared_prefix_stops_at_a_blank_line_before_the_headline():
    class Tok:
        name_or_path = "fake"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            return "<s>[USER]" + msgs[0]["content"] + "[/USER][ASSISTANT]"

    prefix = le.shared_prefix(Tok())
    assert prefix.endswith("\n\n") and "Headline:" not in prefix
    assert prefix.startswith("<s>[USER]") and "Examples (invented headlines)" in prefix
    full = Tok().apply_chat_template([{"content": le.build_prompt("x")}], False, True)
    assert full.startswith(prefix)


def test_cached_prefix_scores_match_the_full_prompt_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )  # fmt: skip
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    prefix = list(range(5, 30))
    suffixes = [[40, 41, 42], [50, 51, 52, 53, 54], [60], [70, 71, 72, 73]]
    letters = [[100 + i] for i in range(12)]
    got = le.score_cached_ids(model, prefix, suffixes, letters, pad_id=0, batch=3)
    with torch.no_grad():
        for row, suffix in zip(got, suffixes, strict=True):
            logits = model(input_ids=torch.tensor([prefix + suffix])).logits[0, -1]
            want = torch.stack([torch.logsumexp(logits[t], dim=-1) for t in letters])
            assert np.abs(row - want.numpy()).max() < 1e-4
