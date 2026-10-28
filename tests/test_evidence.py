import pytest

from analyze import fake_score, largest_face, summarize


def test_no_face_is_insufficient_evidence():
    report = summarize([], 10)
    assert report["outcome"] == "insufficient_evidence"
    assert report["mean_fake_score"] is None


def test_zero_score_is_valid_evidence():
    assert fake_score([{"label": "Fake", "score": 0.0}]) == 0
    assert summarize([0.0, 0.0], 2)["frames_with_face"] == 2


def test_sparse_faces_do_not_create_video_conclusion():
    assert summarize([0.99], 100)["outcome"] == "insufficient_evidence"


def test_missing_fake_label_fails_closed():
    with pytest.raises(ValueError, match="configured fake label"):
        fake_score([{"label": "LABEL_0", "score": 0.9}])


@pytest.mark.parametrize("score", [float("nan"), -1, 1.1])
def test_invalid_model_scores(score):
    with pytest.raises(ValueError):
        fake_score([{"label": "fake", "score": score}])


def test_threshold_boundary_and_largest_face():
    assert summarize([0.75], 1)["outcome"] == "elevated_manipulation_score"
    assert largest_face([(0, 0, 10, 10), (1, 1, 20, 20)]) == (1, 1, 20, 20)
