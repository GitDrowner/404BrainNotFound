#!/usr/bin/env python3
"""Deterministically sample diverse train-only data with a fail-closed DALL-E gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from PIL import Image


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_image(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def dhash_bytes(payload: bytes) -> int:
    with Image.open(io.BytesIO(payload)) as image:
        return dhash_image(image)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extension(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format, ".img")


def forbidden(text: str, patterns: list[re.Pattern]) -> bool:
    normalised = text.casefold().replace("·", "").replace("–", "-")
    return any(pattern.search(normalised) for pattern in patterns)


def claim_unique_digest(
    digest: str,
    seen_hashes: set[str],
    duplicate_skips: Counter[str],
    source: str,
) -> bool:
    """Claim an exact image digest, or audit and skip an already accepted image."""
    if digest in seen_hashes:
        duplicate_skips[source] += 1
        return False
    seen_hashes.add(digest)
    return True


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    attempts: int = 7,
) -> requests.Response:
    """Retry transient Hub/CDN errors while failing fast on permanent 4xx."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=120)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}: {response.url}", response=response
                )
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as error:
            last_error = error
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if attempt + 1 == attempts:
                break
            delay = min(30.0, 2.0 ** attempt)
            print(json.dumps({"event": "http_retry", "attempt": attempt + 1, "delay_seconds": delay, "url": url, "error": str(error)}))
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def holdout_fingerprints(path: Path) -> tuple[set[str], set[int]]:
    records = read_jsonl(path)
    if not records:
        raise RuntimeError(f"Holdout manifest is missing/empty; refusing to build training data: {path}")
    hashes, perceptual = set(), set()
    for record in records:
        value = record.get("sha256")
        if not value:
            raise RuntimeError(f"Holdout record lacks sha256: {record.get('path')}")
        hashes.add(str(value))
        dhash = record.get("dhash64")
        if dhash is None:
            record_path = Path(record["path"])
            if not record_path.is_file():
                raise RuntimeError(f"Holdout record lacks dHash and image is unavailable: {record_path}")
            with Image.open(record_path) as image:
                perceptual.add(dhash_image(image))
        else:
            perceptual.add(int(str(dhash), 16))
    return hashes, perceptual


def save_payload(payload: bytes, directory: Path, stem: str, blocked_hashes: set[str], blocked_dhashes: set[int]) -> tuple[Path, str]:
    digest = digest_bytes(payload)
    if digest in blocked_hashes:
        raise RuntimeError(f"Exact DALL-E Advanced holdout collision: sha256={digest}")
    perceptual = dhash_bytes(payload)
    if any((perceptual ^ blocked).bit_count() <= 4 for blocked in blocked_dhashes):
        raise RuntimeError(f"Near-duplicate DALL-E Advanced holdout collision: dHash={perceptual:016x}")
    path = directory / f"{stem}{extension(payload)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest_file(path) != digest:
        raise RuntimeError(f"Refusing to overwrite non-matching image: {path}")
    if not path.exists():
        path.write_bytes(payload)
    return path, digest


def sample_openfake(
    config: dict,
    profile: dict,
    destination: Path,
    blocked_hashes: set[str],
    blocked_dhashes: set[int],
    patterns: list[re.Pattern],
    rng: random.Random,
    seen_hashes: set[str],
    duplicate_skips: Counter[str],
) -> list[dict]:
    settings = config["openfake"]
    wanted = int(profile["openfake_total"])
    per_model = int(profile["openfake_per_model"])
    page_limit = int(profile["openfake_pages"])
    row_count = int(settings["row_count"])
    counts: Counter[str] = Counter()
    records: list[dict] = []
    offsets = list(range(0, row_count, 100))
    rng.shuffle(offsets)
    session = requests.Session()
    for page_index, offset in enumerate(offsets[:page_limit]):
        if len(records) >= wanted:
            break
        url = "https://datasets-server.huggingface.co/rows"
        params = {"dataset": settings["repo"], "config": settings["config"], "split": settings["split"], "offset": offset, "length": 100}
        response = get_with_retry(session, url, params=params)
        for item in response.json().get("rows", []):
            row = item["row"]
            model = str(row.get("model", "unknown"))
            if forbidden(model, patterns) or counts[model] >= per_model:
                continue
            image_url = row.get("image", {}).get("src")
            if not image_url:
                continue
            image_directory = destination / "images" / "openfake" / re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
            image_stem = f"row_{int(item['row_idx']):08d}"
            existing = sorted(image_directory.glob(f"{image_stem}.*"))
            payload = existing[0].read_bytes() if existing else get_with_retry(session, image_url).content
            payload_digest = digest_bytes(payload)
            if not claim_unique_digest(payload_digest, seen_hashes, duplicate_skips, "openfake"):
                continue
            path, digest = save_payload(
                payload,
                image_directory,
                image_stem,
                blocked_hashes,
                blocked_dhashes,
            )
            label = 1 if str(row.get("label", "")).casefold() == "fake" else 0
            records.append({
                "path": str(path), "label": label, "source": "openfake_core_train", "generator": model,
                "dataset_repo": settings["repo"], "dataset_revision": settings["revision"],
                "dataset_config": settings["config"], "dataset_split": settings["split"],
                "dataset_row": int(item["row_idx"]), "release_date": row.get("release_date"),
                "generator_type": row.get("type"), "sha256": digest, "role": "train"
            })
            counts[model] += 1
            if len(records) >= wanted:
                break
        if page_index and page_index % 25 == 0:
            print(json.dumps({"event": "openfake_progress", "pages": page_index + 1, "downloaded": len(records), "models": len(counts)}))
    if len(records) < wanted:
        raise RuntimeError(f"OpenFake sampler found {len(records)}/{wanted} allowed rows before page limit")
    return records


def sample_tigas(
    config: dict,
    profile: dict,
    destination: Path,
    blocked_hashes: set[str],
    blocked_dhashes: set[int],
    patterns: list[re.Pattern],
    rng: random.Random,
    seen_hashes: set[str],
    duplicate_skips: Counter[str],
) -> list[dict]:
    settings = config["tigas"]
    annotation_path = Path(hf_hub_download(settings["repo"], settings["annotation"], repo_type="dataset", revision=settings["revision"]))
    grouped: dict[str, list[str]] = defaultdict(list)
    with annotation_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            relative = row["image_path"].replace("\\", "/")
            parts = relative.split("/")
            generator = parts[1]
            if row["label"] == "1" and generator in settings["fake_generators"] and not forbidden(generator, patterns):
                grouped[generator].append(relative)
    records = []
    per_generator = int(profile["tigas_fake_per_model"])
    for generator in settings["fake_generators"]:
        candidates = grouped.get(generator, [])
        rng.shuffle(candidates)
        if len(candidates) < per_generator:
            raise RuntimeError(f"TIGAS source {generator} has only {len(candidates)} fake rows")
        accepted = 0
        for relative in candidates:
            repo_path = f"train/{relative}"
            cached = Path(hf_hub_download(settings["repo"], repo_path, repo_type="dataset", revision=settings["revision"]))
            payload = cached.read_bytes()
            payload_digest = digest_bytes(payload)
            if not claim_unique_digest(payload_digest, seen_hashes, duplicate_skips, "tigas"):
                continue
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(relative).stem)
            path, digest = save_payload(payload, destination / "images" / "tigas" / generator, safe_name, blocked_hashes, blocked_dhashes)
            records.append({
                "path": str(path), "label": 1, "source": "tigas_train", "generator": generator,
                "dataset_repo": settings["repo"], "dataset_revision": settings["revision"],
                "dataset_split": settings["split"], "dataset_path": repo_path,
                "sha256": digest, "role": "train"
            })
            accepted += 1
            if accepted >= per_generator:
                break
        if accepted < per_generator:
            raise RuntimeError(
                f"TIGAS source {generator} produced only {accepted}/{per_generator} unique fake rows"
            )
    return records


def add_local_generated(
    config: dict,
    destination: Path,
    blocked_hashes: set[str],
    blocked_dhashes: set[int],
    seen_hashes: set[str],
    duplicate_skips: Counter[str],
) -> list[dict]:
    records = read_jsonl(Path(config["local_generated_manifest"]))
    output = []
    for index, record in enumerate(records):
        source = Path(record["path"])
        digest = digest_file(source)
        if digest != record.get("sha256"):
            raise RuntimeError(f"Local generated checksum mismatch: {source}")
        if digest in blocked_hashes:
            raise RuntimeError(f"Local generated image collides with DALL-E Advanced: {source}")
        if not claim_unique_digest(digest, seen_hashes, duplicate_skips, "local_generated"):
            continue
        with Image.open(source) as image:
            perceptual = dhash_image(image)
        if any((perceptual ^ blocked).bit_count() <= 4 for blocked in blocked_dhashes):
            raise RuntimeError(f"Local generated image is a near-duplicate of DALL-E Advanced: {source}")
        target = destination / "images" / "local_generated" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        copied = dict(record)
        copied.update({"path": str(target), "sha256": digest, "role": "train", "source": "codex_builtin_imagegen_train_only"})
        output.append(copied)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("configs/fusion_v2_sources.json"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("data/fusion_v2"))
    parser.add_argument("--skip-openfake", action="store_true")
    parser.add_argument("--skip-tigas", action="store_true")
    parser.add_argument(
        "--exclude-openai-family",
        action="store_true",
        help="Optional ablation only: exclude every OpenAI-family generator, not just DALL-E 3.",
    )
    args = parser.parse_args()
    config = json.loads(args.sources.read_text())
    profile = config["profiles"][args.profile]
    rng = random.Random(int(config["seed"]))
    pattern_text = list(config["forbidden_generator_patterns"])
    if args.exclude_openai_family:
        pattern_text.extend(config["optional_family_disjoint_patterns"])
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_text]
    blocked, blocked_dhashes = holdout_fingerprints(Path(config["holdout_manifest"]))
    destination = args.output / args.profile
    started = time.monotonic()
    records = []
    seen_hashes: set[str] = set()
    duplicate_skips: Counter[str] = Counter()
    if not args.skip_openfake:
        records.extend(
            sample_openfake(
                config, profile, destination, blocked, blocked_dhashes, patterns, rng,
                seen_hashes, duplicate_skips,
            )
        )
    if not args.skip_tigas:
        records.extend(
            sample_tigas(
                config, profile, destination, blocked, blocked_dhashes, patterns, rng,
                seen_hashes, duplicate_skips,
            )
        )
    if profile.get("include_local_generated", False):
        records.extend(
            add_local_generated(
                config, destination, blocked, blocked_dhashes, seen_hashes, duplicate_skips
            )
        )
    hashes = [record["sha256"] for record in records]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Duplicate SHA-256 values in sampled training data")
    for record in records:
        identity = " ".join(str(record.get(key, "")) for key in ("source", "generator", "dataset_repo"))
        if forbidden(identity, patterns):
            raise RuntimeError(f"Forbidden generator reached manifest: {identity}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "train_additions.jsonl"
    with manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter(record["generator"] for record in records)
    receipt = {
        "status": "PASS", "profile": args.profile, "manifest": str(manifest),
        "manifest_sha256": digest_file(manifest), "records": len(records),
        "generators": dict(sorted(counts.items())), "holdout_manifest": config["holdout_manifest"],
        "holdout_hash_count": len(blocked), "holdout_dhash_count": len(blocked_dhashes),
        "exact_holdout_overlap": 0, "near_duplicate_holdout_overlap_radius_4": 0,
        "forbidden_generator_patterns": pattern_text,
        "duplicate_sha256_skips": dict(sorted(duplicate_skips.items())),
        "elapsed_seconds": time.monotonic() - started,
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
