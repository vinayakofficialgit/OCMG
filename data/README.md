# Data

## LoCoMo (primary benchmark)

- File: `locomo10.json`
- Source: https://github.com/snap-research/locomo (file `data/locomo10.json`)
- Paper: Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F. and Fang, Y. (2024)
  Evaluating very long-term conversational memory of LLM agents. ACL 2024.
- Licence: see the LoCoMo repository. The file is included here unchanged for reproducibility.

Contents, as checked by `python scripts/verify_data.py`:

| Item | Count |
|---|---|
| Conversations | 10 |
| Sessions | 272 |
| Dialogue turns | 5,882 (1,226 share an image, described by a caption) |
| Questions | 1,986 |

Question categories (integer codes in the file, meaning confirmed from the evidence counts):

| Code | Category | Questions |
|---|---|---|
| 1 | Multi-hop | 282 |
| 2 | Temporal | 321 |
| 3 | Open-domain | 96 |
| 4 | Single-hop | 841 |
| 5 | Adversarial (unanswerable) | 446 |

Known issues in the release, handled by the loader:

- Category 5 questions have no `answer`, only `adversarial_answer`, which is the misleading answer.
  The loader uses an abstention ("Not mentioned in the conversation") as the reference.
- 9 evidence ids are malformed (e.g. `D8:6; D9:17`, `D:11:26`, `D30:05`). The loader normalises
  them; 2 point to turns missing from the release and are dropped and reported.

## LongMemEval-S (secondary benchmark, not included)

Download from https://github.com/xiaowu0162/LongMemEval and place the JSON in this folder.
