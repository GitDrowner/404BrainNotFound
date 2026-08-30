# Provenance and limitations

## Revision-pinned upstreams

| Dataset | Revision | Declared license at preparation time | Upstream |
|---|---|---|---|
| DiTFake | `66105473704c6ebcc03cb971322bb95f04c7bdc4` | Apache-2.0 | <https://huggingface.co/datasets/lioooox/DiTFake> |
| AI detector benchmark test data | `cc47a51ab993b4b6937acd29774d931cb756d43a` | Apache-2.0 | <https://huggingface.co/datasets/ash12321/ai-detector-benchmark-test-data> |
| Qwen-Image-Bench | `d2493deb153b020cf169c7e3f57d15e4dd697038` | Apache-2.0 | <https://huggingface.co/datasets/Qwen/Qwen-Image-Bench> |
| Defactify Image Dataset | `787334f7857fa54f29027a7f09c30e895ad486ef` | Check upstream card | <https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset> |
| MS-COCO-unique | `21aa022d20f360544a30e0c5a303b4c2bdbec8b5` | Check upstream card | <https://huggingface.co/datasets/bitmind/MS-COCO-unique> |
| GenImage MidJourney | `583d750ace927061a1159de64ca65b82f784f661` | Check upstream card | <https://huggingface.co/datasets/bitmind/GenImage_MidJourney> |

The configuration pins revisions rather than relying on a moving default branch. Operators remain
responsible for checking the upstream dataset card and terms before redistribution or commercial
use. This repository does not redistribute upstream files.

## Scientific interpretation

- DiTFake and Qwen-Image-Bench are maintainer/paper benchmark releases and are the stronger sources
  in this suite.
- `frontier_small_2026` is a small community release containing processed images and limited source
  provenance. Treat it as a supplementary stress test, not primary scientific evidence.
- `qwen_image_bench_frontier_fake_only` contains no negative class. AUROC, balanced accuracy and
  real recall are undefined for this subset.
- `seen_in_training` is a conservative exact normalized-name annotation. It is not a family-level
  novelty guarantee and does not alter sample selection or metrics.
- Exact SHA-256 blocking rules out identical files against supplied manifests. It does not prove the
  absence of semantic near-duplicates unless the operator supplies additional perceptual checks.
- The DALL·E 3 set computes 64-bit dHash against all resolvable training images at radius 4 and
  quarantines the real/fake caption pair together. The historical audited run started from 1,500
  pairs and retained 1,496 pairs.
- The COCO/MidJourney benchmark reproduces the original fixed-revision, fixed-seed reservoir
  sampling. Its upstream split is named `train`; locally every record is marked
  `external_evaluation_only` and is never a detector-training input.

## Historical contamination boundary

The audited 2026-08-30 preparation blocked exact hashes from:

- job 773086 training: 44,377 records;
- DALL-E Advanced external holdout: 2,992 records;
- commercial frontier blind set: 13 records.

No blocked manifests or images are committed here. Reproduction requires the operator to provide
their own local copies through command-line arguments.
