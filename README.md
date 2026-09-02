# RayDer
[![Conference](https://img.shields.io/badge/ECCV-2026-4b2e83)](https://compvis.github.io/rayder/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://compvis.github.io/rayder/)
[![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://arxiv.org/abs/2605.31535)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/CompVis/rayder)

<h2 align="center">Scalable Self-Supervised Novel View Synthesis from Real-World Video</h2>

<p align="center">
  <b>ECCV 2026</b>
</p>

<div align="center">
  <a href="https://scholar.google.com/citations?user=-kkLqx0AAAAJ&hl=en" target="_blank">Ulrich Prestel</a><sup>*</sup> &middot;
  <a href="https://stefan-baumann.eu/" target="_blank">Stefan Andreas Baumann</a><sup>*</sup> &middot;
  <a href="https://nickstracke.dev/" target="_blank">Nick Stracke</a> &middot;
  <a href="https://ommer-lab.com/people/ommer/" target="_blank">Björn Ommer</a>
</div>

<p align="center">
  <b>CompVis @ LMU Munich, MCML</b>
</p>

Self-supervised novel view synthesis methods are fundamentally *data-limited*: they require static-scene training data, which is scarce. RayDer removes this bottleneck by enabling stable training on **general, dynamic real-world video**. By consolidating three separate networks into one unified transformer, introducing dynamic state prediction with dropout, and improving pose learning through autoregressive training, RayDer's performance scales predictably with data, model size, and compute — following power-law scaling relationships (R² > 0.99) analogous to those observed in LLMs.

<p align="center">
  <img src="docs/static/images/paper-svg/teaser.png" alt="RayDer enables training NVS from abundant general video, removing the static-scene data bottleneck" width="100%">
</p>

This is a minimal, self-contained PyTorch re-implementation of RayDer (covering inference, training code coming soon).



## Results
Zero-shot qualitative comparison of RayDer-L against E-RayZer in various NVS settings and an extreme setting with near-zero context-view overlap:

<p align="center">
  <img src="docs/static/images/paper-svg/qualitative.png" alt="Zero-shot qualitative samples of RayDer-L compared with E-RayZer in typical NVS and extreme settings" width="100%">
</p>

See the [project page](https://compvis.github.io/rayder/) for additional samples (incl. videos) and analysis.



## Usage

<!--
### Via `torch.hub`
The simplest way to use RayDer is via `torch.hub`:
```python
model = torch.hub.load("CompVis/rayder", "rayder_l_576")  # downloads checkpoint automatically, no code install required
```
-->

### Setup
The model only depends on a recent `torch`, `torchvision`, `einops`, and `jaxtyping`; the demos additionally need `gradio`, `Pillow` and `imageio`/`imageio-ffmpeg`.
Install them via:
```shell
pip install -r requirements.txt
```

### Standalone
If you want to integrate RayDer into your own codebase, copy `rayder/model.py` and you should be good to go. Then instantiate the model as:
```python
from rayder.model import RayDer_L

model = RayDer_L()
model.load_state_dict(torch.load("rayder_l_576.pt", weights_only=True))
model.requires_grad_(False)
model.eval()
```

The `RayDer` class exposes three high-level inference methods:
- `predict_cameras(x)`: estimate camera parameters from a set of input views (trained for 8 views, but the models extrapolate quite well)
- `predict_cameras_and_states(x, temporal_ranks=None)`: estimate cameras and per-view dynamic nuisance states, optionally using an explicit temporal positional-rank permutation for controlled experiments
- `predict_views(x_in, cam_in, cam_target)`: synthesize novel views at target camera poses (trained for 1-7 input views, arbitrarily many output views)

Cameras are represented as custom dataclasses that can be directly sliced/indexed as a whole.

### About the Codebase
Code is separated into clearly labeled blocks with comments explaining relevant design choices and conventions.
For all public-facing APIs involving tensors, type hints with [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) are provided (e.g. `img: Float[torch.Tensor, "b t h w c"]`), annotating dtype, tensor type, and shape.

**Conventions.** Images are channels-last `(b, t, h, w, 3)`, **not** the PyTorch-default `(b, t, 3, h, w)`, with pixel values in [-1, 1].
Camera extrinsics use the **camera-to-world (c2w)** convention: `R` rotates camera-space directions into world space and `t` is the camera position in world coordinates.
The focal length `f` is normalized by the shorter image side: `f = f_pixels / min(h-1, w-1)`.

### Generating Videos
Use `generate_video.py` to produce smooth view-interpolation videos from a set of input images:
```shell
python generate_video.py --image_dir /path/to/input/images --output output.mp4 --steps_per_pair 10 --fps 15
```
A checkpoint will be downloaded automatically if not explicitly specified.

### Batched OpenIllumination Compositing

The OpenIllumination helper can stream many mixtures directly into an inference
loop. A full 142-column weight matrix is accepted, but only lights used by at
least one pattern are loaded. The selected OLAT basis is cropped/resized once in
linear RGB and reused for every batch; no rendered images need to be written to
disk.

```python
import numpy as np

from rayder.open_illumination import OpenIlluminationOLAT

dataset = OpenIlluminationOLAT("/path/to/openillumination", object_id=5)
light_positions = dataset.load_light_positions()                 # (142, 3)
light_directions = dataset.load_light_positions(normalize=True)  # unit vectors

light_indices = [0, 8, 16, 32, 64, 112]
weights = np.random.default_rng(0).dirichlet(np.ones(6), size=500)

for images in dataset.iter_render_weight_batches(
    weights,
    cameras=["A1", "A2", "A3", "C1", "C4"],
    light_indices=light_indices,
    batch_size=4,
    size=256,
    center_crop=True,
):
    # images: float32 sRGB in [0, 1], shape (batch, camera, 256, 256, 3)
    # x = torch.from_numpy(images).to(device).mul_(2).sub_(1)
    ...
```

For small outputs, `render_weight_batch(...)` accepts the same arguments and
returns the entire array. `download()` includes the dataset-wide
`light_pos.npy` metadata by default and downloads up to eight files concurrently.
Pass `max_workers=1` for serial downloads, or tune the worker count for your
connection.

### Interactive Demo
Launch the Gradio app for an interactive browser-based demo:
```shell
python app.py
```
Upload a set of views, adjust the number of interpolation steps, and generate a novel-view video.
The RayDer-L-576² model is loaded automatically via `torch.hub`.



## Models
We currently release the following model variants:

| Variant | Width | Depth | Params | Resolution | `torch.hub` name |
| :------ | ----: | ----: | -----: | ---------: | :--------------- |
| RayDer-L | 1024 | 24 | ~743M | 256² | `rayder_l` |
| RayDer-L-576² | 1024 | 24 | ~743M | 576² | `rayder_l_576` |

Weights are released via [HuggingFace](https://huggingface.co/CompVis/rayder).
Additional model variants and licensing available upon request.



## Acknowledgments
RayDer conceptually builds upon [RayZer](https://hwjiang1510.github.io/RayZer/) (Jiang et al., ICCV 2025), which introduced self-supervised NVS from unposed images via ray map-conditioned rendering.
We extend their method to more general pretraining on dynamic video, consolidate the architecture, and enable variable input-view-count inference.

Parts of this repo are taken from the [Flow Poke Transformer](https://github.com/CompVis/flow-poke-transformer) (Baumann et al., ICCV 2025) public implementation (MIT).
We also acknowledge code adapted from [HDiT](https://github.com/crowsonkb/k-diffusion) (Crowson et al., ICML 2024; MIT).
The gradio app loosely adapts some code from [E-RayZer](https://github.com/QitaoZhao/E-RayZer) (Zhao et al., CVPR 2026; MIT).

## License
This software is released under a license for personal and scientific non-commercial research purposes -- see [LICENSE.md](LICENSE.md) for the full terms.
For any commercial use or exploitation, please contact <license.compvis@ifi.lmu.de>.

## Citation
If you find our model or code useful, please cite our paper:
```bibtex
@misc{prestel2026rayderscalableselfsupervisednovel,
      title={RayDer: Scalable Self-Supervised Novel View Synthesis from Real-World Video}, 
      author={Ulrich Prestel and Stefan Andreas Baumann and Nick Stracke and Björn Ommer},
      year={2026},
}
```
