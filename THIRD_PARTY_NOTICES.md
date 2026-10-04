# Dependencies and references

Dependencies are installed separately; their code, licenses and SC2 game assets are not bundled.

- [PySC2](https://github.com/google-deepmind/pysc2): SC2 environment and engine interface.
- [s2client-proto](https://github.com/Blizzard/s2client-proto): SC2 protocol definitions.
- PyTorch and NumPy: models, optimization and numerical arrays.
- s2protocol and mpyq: replay decoding and archive access.
- absl-py: PySC2 runtime configuration.
- Zstandard: optional compressed data storage.
- pytest and pytest-subtests: tests.

Research references:

- [DI-star](https://github.com/opendilab/DI-star): structured-action and replay-training approach. The coarse action conventions reference [features.py](https://github.com/opendilab/DI-star/blob/12b1c69350ad41e17895c602a66a52d98dd58452/distar/agent/default/lib/features.py).
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z): multi-agent SC2 learning and population methods.

Upstream licenses are not the license of PoorStar. Source-specific copyright and attribution notices must accompany any reused source.
