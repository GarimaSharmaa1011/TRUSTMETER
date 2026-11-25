from train.evaluate import _risk_coverage, extract_choice, split_question_ids


def test_question_splits_are_disjoint_complete_and_reproducible():
    question_ids = [f"arc-{index}" for index in range(300)]

    first = split_question_ids(question_ids, seed=42)
    second = split_question_ids(question_ids, seed=42)

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "train": 150,
        "validation": 75,
        "test": 75,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set.union(*(set(ids) for ids in first.values())) == set(question_ids)


def test_arc_choice_grading_requires_an_exact_known_label():
    choices = [{"label": label, "text": label.lower()} for label in ["A", "B", "C", "D"]]
    assert extract_choice("A", choices) == "A"
    assert extract_choice("Answer: C.", choices) == "C"
    assert extract_choice("option Z", choices) is None


def test_aurc_is_not_zero_for_a_constant_confidence_curve():
    curve, aurc = _risk_coverage([True, False], [0.8, 0.8])
    assert aurc == 0.5
    assert len(curve) == 101
