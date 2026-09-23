# Dependencies and upstream materials

The [MIT license](LICENSE) applies to this repository's integration code.
Installed dependencies, simulator assets and remote model weights retain their
upstream terms. This repository contains no model weights, engine binaries,
simulator asset bundle or vendored SDK implementation.

| Material | Pinned source | License information |
| --- | --- | --- |
| HUD SDK | [hud-python](https://github.com/hud-evals/hud-python/tree/0b63b4d3b9acb6d095e0886e18b2c905219e1e5a) | [MIT](https://github.com/hud-evals/hud-python/blob/0b63b4d3b9acb6d095e0886e18b2c905219e1e5a/LICENSE) |
| Dreamscale SDK | [PyPI `dropbear` 0.1.0a15](https://pypi.org/project/dropbear/0.1.0a15/) | Apache-2.0, as included in the published wheel |
| OpenPI client | [PyPI 0.1.2](https://pypi.org/project/openpi-client/0.1.2/) | Apache-2.0, as included in the published wheel |
| NumPy | [2.2.6](https://numpy.org/doc/2.2/license.html) in the agent; 1.26.4 in the simulator | BSD-3-Clause and bundled third-party notices |
| LIBERO package | [hf-libero 0.1.3](https://pypi.org/project/hf-libero/0.1.3/), from [Hugging Face's LIBERO fork](https://github.com/huggingface/LIBERO) | MIT for the package; see its upstream notices |
| MuJoCo | [3.3.7](https://github.com/google-deepmind/mujoco/tree/3.3.7) | [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/3.3.7/LICENSE) |
| robosuite | [1.4.1](https://github.com/ARISE-Initiative/robosuite/tree/v1.4.1) | [MIT](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/LICENSE) |
| Simulator assets | [lerobot/libero-assets](https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6) | Separate upstream materials; the pinned snapshot has no top-level license declaration |
| Remote model | [MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2) | Separately hosted; the pinned checkpoint card does not declare a license |

The simulator build fetches assets from their pinned upstream repository and
installs third-party packages. Publishing this source recipe does not relicense
those materials. A separately distributed container must retain the licenses and
notices of its included packages and assets.

The agent and simulator lockfiles record their full dependency graphs. Package
metadata and installed license files provide the notices for transitive
dependencies; the table above is an orientation guide, not a replacement for them.
