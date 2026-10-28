# Deepfake video analysis

Sample faces from a local video, apply an image manipulation classifier, and
export a report with score, face coverage, and an explicit insufficient-evidence outcome.

The main engineering concern is evidence handling. A video with no detected
faces cannot support a conclusion about authenticity. Zero-valued fake scores
are valid observations and remain in the aggregation.

## Pipeline

```mermaid
flowchart LR
  V[Local video] --> S[Sample every N frames]
  S --> F[Largest frontal face]
  F --> C[Image classifier]
  C --> A[Mean score and face coverage]
  A --> R[Local JSON report]
```

## Run

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt
python analyze.py --input data/example.mp4 --output outputs/analysis.json
```

Python 3.10+. The first run downloads the configured Hugging Face model to
the local model cache. Input video is processed locally. Review the model's
license and model card before use. For repeatable analysis, pass a reviewed
commit with `--revision`; the report records the resolved model commit when available.

Options include `--every 5`, `--threshold 0.75`, `--min-coverage 0.2`,
`--model` and `--fake-label`. The default thresholds are demonstration
settings, with no calibration claim.

## Reading the result

| Outcome | Meaning |
| --- | --- |
| `insufficient_evidence` | No usable faces, or face coverage below the configured minimum |
| `elevated_manipulation_score` | Mean model score meets the demo threshold |
| `below_demo_threshold` | Mean score falls below the demo threshold |

A score below the threshold does not prove a video is authentic. The image
classifier does not analyze audio or temporal consistency. The largest face
alone is sampled; other people can be missed. Frontal Haar detection can fail
on pose, occlusion and lighting, and coverage refers to sampled frames only.
Compression and domain shifts can change results substantially.

## Design notes and validation

The report tool uses OpenCV's bundled Haar cascade for frontal-face crops.
This keeps face detection available through a single explicit dependency. There is no video-level
accuracy benchmark in this repository. Tests cover missing evidence, sparse
coverage, label mapping, valid zero scores and invalid classifier output.

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```
