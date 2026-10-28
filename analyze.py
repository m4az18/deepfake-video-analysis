"""Sample visible faces in a local video and export model scores as JSON."""

import argparse
import json
import math
from pathlib import Path


def fake_score(predictions, fake_label="fake"):
    """Require an explicit label; missing predictions are never treated as real."""
    for prediction in predictions:
        if prediction["label"].casefold() == fake_label.casefold():
            value = float(prediction["score"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Classifier returned an invalid score")
            return value
    raise ValueError(f"Classifier did not return the configured fake label: {fake_label}")


def summarize(scores, sampled_frames, threshold=0.75, min_coverage=0.2):
    coverage = len(scores) / sampled_frames if sampled_frames else 0.0
    mean = sum(scores) / len(scores) if scores else None
    if mean is None or coverage < min_coverage:
        outcome = "insufficient_evidence"
    elif mean >= threshold:
        outcome = "elevated_manipulation_score"
    else:
        outcome = "below_demo_threshold"
    return {
        "outcome": outcome,
        "mean_fake_score": mean,
        "face_coverage": coverage,
        "sampled_frames": sampled_frames,
        "frames_with_face": len(scores),
        "threshold": threshold,
        "minimum_face_coverage": min_coverage,
    }


def largest_face(boxes):
    return max(boxes, key=lambda box: int(box[2]) * int(box[3]), default=None)


def analyze(args):
    import cv2
    from PIL import Image
    from transformers import pipeline

    if not Path(args.input).is_file():
        raise ValueError("Input video does not exist")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; choose a new path")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV face cascade could not be loaded")
    options = {"revision": args.revision} if args.revision else {}
    classifier = pipeline("image-classification", model=args.model, trust_remote_code=False, **options)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Video cannot be opened")
    scores, sampled, decoded = [], 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index = decoded
            decoded += 1
            if index % args.every:
                continue
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            face = largest_face(boxes)
            if face is None:
                continue
            x, y, width, height = map(int, face)
            # Include a small border without allowing the crop outside the image.
            margin_x, margin_y = int(width * 0.075), int(height * 0.075)
            crop = frame[
                max(0, y - margin_y) : min(frame.shape[0], y + height + margin_y),
                max(0, x - margin_x) : min(frame.shape[1], x + width + margin_x),
            ]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            predictions = classifier(Image.fromarray(rgb), top_k=None)
            scores.append(fake_score(predictions, args.fake_label))
    finally:
        capture.release()
    if not decoded:
        raise ValueError("Video has no decodable frames")
    report = summarize(scores, sampled, args.threshold, args.min_coverage)
    report.update(
        {
            "decoded_frames": decoded,
            "sampling_interval": args.every,
            "model": args.model,
            "requested_revision": args.revision,
            "model_commit": getattr(classifier.model.config, "_commit_hash", None),
            "face_detector": "opencv-haar-frontalface",
            "selection": "largest_face",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/analysis.json")
    parser.add_argument("--model", default="dima806/deepfake_vs_real_image_detection")
    parser.add_argument("--revision", help="Reviewed Hugging Face model commit or revision")
    parser.add_argument("--fake-label", default="fake")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--min-coverage", type=float, default=0.2)
    args = parser.parse_args()
    if args.every < 1 or not 0 < args.threshold < 1 or not 0 < args.min_coverage <= 1:
        parser.error("every must be positive; threshold and coverage must be between 0 and 1")
    try:
        analyze(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
