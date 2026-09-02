## Summary

This document investigates three questions:

1. How much information about illumination does RayDer retain in its representations?
2. Is that information encoded consistently enough to transfer between objects and camera views in a straightforward manner?
3. Can a small post-training modification make illumination explicitly controllable?

The experiments give a qualified answer. Illumination can be decoded almost perfectly when the object and view are fixed, but the tested representations do not expose a shared, view-independent illumination geometry. Directly transplanting changes in RayDer's nuisance state consequently does not produce useful cross-view relighting: it transfers donor-view appearance or residual donor structure instead. A lightweight AdaRMSNorm adaptation does learn coarse lighting control in-distribution, but its generalization and rendering fidelity remain limited.

## Experimental setting and limitations

The experiments use the [OpenIllumination](https://oppo-us-research.github.io/OpenIllumination/) dataset, which provides one-light-at-a-time (OLAT) images for up to 142 point lights and a matrix of camera views:

$$
\begin{bmatrix}
A1 & A2 & A3 & A4 & A5 & A6 \\
& B2 & B3 & B4 & B5 & B6 \\
C1 & C2 & C3 & C4 & C5 & C6 \\
D1 & & D3 & D4 & D5 & D6
\end{bmatrix}.
$$

Lighting mixtures are synthesized by converting JPEG images from sRGB to linear color space, combining the OLAT images, and converting the result back to sRGB. The mixtures are therefore approximate rather than physically exact.

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <p align="center">The same camera track, A1 through A6, under distinct lighting mixtures.</p>
  <img src="input_example.png" alt="One object shown from all views under different lighting" width="50%">
</div>

OpenIllumination images have no tangible background and differ substantially from RayDer's pretraining distribution. The author's explicitly highlight this in their evaluations on robotics datasets that are similar in nature to this one. The model's pose estimation and rendering quality are therefore expected to suffer. This is an important limitation throughout: the experiments characterize RayDer under this deliberately controlled dataset, but do not establish what would happen after training on broader, more representative relighting data.

# 1. Does RayDer encode illumination?

This is somewhat of a trivial question given that without information about illumination in the representations, a novel view would be unlit. However we briefly quantify the answer in a specific controlled settings.

## 1.1 Within a fixed object and view: linearly recoverable.

We first ask whether illumination can be recovered at all from the model's representations.
Six lights are selected so that they are roughly distributed around the object.

<p align="center">
  <img src="light_rig.png" alt="Six-point-light setup in 3D" width="50%">
</p>

Light intensities are sampled from four families of Dirichlet distributions, ranging from concentrated to diffuse illumination. A normalized combination of $N$ OLAT images lies in a simplex of at most dimension $N-1$ in linear RGB space. **For the six-light setup, linear regression on raw pixels can consequently recover the mixture essentially perfectly**.

We fit ridge regressions from several representations to the first five mixture weights; the sixth is implied by $w_6=1-\sum_{k=1}^5w_k$. One fixed 80/20 split is shared by all representations, and ridge strength is selected using only the training set. The evaluated representations are RayDer's nuisance state $r_\text{dynamic}\in\mathbb{R}^{256}$, the post-norm vector from its camera-pose head $r_\text{backbone}\in\mathbb{R}^{1024}$, DINOv3 features $r_\text{DINOv3}\in\mathbb{R}^{1024}$, and raw pixels.

| Representation | Ridge $R^2$ | Ridge MAE |
| --- | ---: | ---: |
| RayDer backbone token | 0.998 | 0.0059 |
| RayDer nuisance state | 0.997 | 0.0071 |
| DINOv3 | 0.972 | 0.0185 |
| Linear Pixels | 1.000 | $1\times10^{-17}$ |

**Result.** All evaluated representations retain enough information to decode a fixed, low-dimensional lighting mixture with high precision. This establishes availability of illumination information, but not a separable or transferable illumination representation.

## 1.2 Across objects and views: no shared linear geometry is evident

We next test whether one linear probe transfers beyond the objects and camera views on which it was fitted. Four regimes distinguish mixture interpolation from object and view transfer:

| Objects | Views | What is tested |
| --- | --- | --- |
| Source | Source | Mixture-only control |
| Held out | Source | Object transfer |
| Source | Held out | View transfer |
| Held out | Held out | Joint object-and-view transfer |

For every object-view pair $(o,v)$, the representation under uniform illumination $\omega_0=\langle1/6,\dots,1/6\rangle$ supplies a reference $z_{ov}(\omega_0)$. The absolute representation under illumination $\omega$ is $z_{ov}(\omega)$; its anchored form is

$$
z_{ov}^{\mathrm{anchor}}(\omega)=z_{ov}(\omega)-z_{ov}(\omega_0).
$$

The prediction target is the centered mixture expressed in an orthonormal basis $H$,

$$
q(\omega)=(\omega-\omega_0)^\top H.
$$

A single affine ridge probe $f(z)=W^\top z+b$ is trained on absolute source representations. In the reference-calibrated evaluation, it remains frozen and predicts

$$
f(z_{ov}(\omega))-f(z_{ov}(\omega_0))=W^\top\bigl(z_{ov}(\omega)-z_{ov}(\omega_0)\bigr).
$$

For RayDer, we evaluate both:
- Representations inferred independently for each image.
- Contextual representations inferred while the model jointly observes multiple views.

The following table reports the hardest regime, in which both object and view are held out:

| Representation | Absolute $R^2$ | Anchored $R^2$ | Absolute MAE | Anchored MAE |
| --- | ---: | ---: | ---: | ---: |
| RayDer token — independent | -17.293 | -2.393 | 0.4995 | 0.2088 |
| RayDer token — context | -4.559 | **-0.271** | 0.2950 | **0.1330** |
| RayDer nuisance — independent | -53.376 | -9.889 | 0.7901 | 0.3302 |
| RayDer nuisance — context | -21.916 | -4.182 | 0.5481 | 0.2522 |
| DINOv3 — per image | -2.982 | -0.357 | 0.2646 | 0.1404 |
| Linear OLAT pixels | -13.045 | -4.782 | 0.3322 | 0.2043 |

**Result.** Reference anchoring reduces error substantially, but all $R^2$ values remain negative even with target-domain calibration. Contextual RayDer backbone tokens transfer least poorly; RayDer's nuisance state transfers worst.

The pixel probe is not expected to succeed across views. Its role is comparative rather than as a useful baseline. The nuisance state's similarly poor behavior supports the interpretation that illumination affects it primarily through view-specific residual appearance.

Refitting the probe on anchored rather than absolute source representations does not materially change the conclusion. Those results and the full measurement matrix can be found in [Appendix A](#appendix-a-additional-probe-results).

# 2. Can nuisance-state changes relight another view?

The findings in Appendix C of the RayDer paper and the failed transfer probes, make direct cross-view state transplantation unlikely to work without further alignment. We test this directly by taking a lighting-induced change observed at a donor view and injecting it into a different target view.

For equal-weight illumination $\omega_0$, donor view $v_d$, target view $v_t$, and target mixture $\omega$, define

$$
\Delta s_{d,\omega}=s_{v_d,\omega}-s_{v_d,\omega_0},
\qquad
s^{\mathrm{rel}}_{t,\omega}(\lambda)
=s_{v_t,\omega_0}+\lambda\Delta s_{d,\omega}.
$$

The primary relative intervention uses $\lambda=1$. An absolute intervention, $s^{\mathrm{abs}}_{t,\omega}=s_{v_d,\omega}$, directly supplies the donor state under the requested illumination.

RayDer first infers camera intrinsics and extrinsics jointly from all six views under uniform illumination; these are then held fixed. The interventions are evaluated on OpenIllumination Object 4 using 16 lighting mixtures sampled from the same four Dirichlet families used by the probes.

In the "donor observed" intervention the model sees $v_d$ under the intended light condition $\omega$ instead of $\omega_0$, along with the other non-held-out views under uniform illumination.
In "Uniform context" interventions all views are observed under $\omega_0$.

<div align="center">
  <img src="transplant_all_approaches_1.png" alt="Absolute and relative nuisance-state transplantation results" width="70%">
  <p>Object 4 from target views A2 through A6, with A1 as donor. The displayed weights specify the target light mixture.</p>
</div>

The figure compares the following interventions:

| Grid row | Decoder context | State supplied for the target |
| --- | --- | --- |
| Uniform context + anchor | Five non-target views under $\omega_0$ | $s_{t,\omega_0}$ |
| Uniform context + relative state | Same five uniform views | $s_{t,\omega_0}+s_{d,\omega}-s_{d,\omega_0}$ |
| Uniform context + absolute state | Same five uniform views | $s_{d,\omega}$ |
| Donor observed + anchor | A1 observed as $I_{d,\omega}$ | $s_{t,\omega_0}$ |
| Donor observed + relative state | Same donor-observed context | $s_{t,\omega_0}+s_{d,\omega}-s_{d,\omega_0}$ |
| Donor observed + absolute state | Same donor-observed context | $s_{d,\omega}$ |
| Target-state oracle | Same donor-observed context | $s_{t,\omega}$ |

We quantify whether a transplant moves the decoding toward the requested target image with the masked-MSE gain

$$
G_{\mathrm{GT}}
=1-\frac{\operatorname{MSE}_{\mathrm{mask}}(\hat I_{\mathrm{trans}},I_{t,\omega})}
{\operatorname{MSE}_{\mathrm{mask}}(\hat I_0,I_{t,\omega})},
$$

where $\hat I_0$ is the decoding from the target view's uniform-light anchor state and $\hat I_{\mathrm{trans}}$ is the decoding after transplantation, both under the same uniform decoder context. $I_{t,\omega}$ is off course the target view under target illumination. This is a fractional reduction in reconstruction error on the target object: $G_{\mathrm{GT}}=0$ means no improvement over the anchor, $1$ means zero error, and a negative value means that transplantation increased the error. The mean of the per-case gains over all 30 ordered cross-view donor-target pairs and 16 mixtures (480 cases) is $G_{\mathrm{GT}}=0.100$ for relative transplantation at $\lambda=1$, corresponding to a 10.0% reduction from the anchor error. Additionally observing the donor under target lighting raises this mean to $G_{\mathrm{GT}}=0.105$. Absolute transplantation gives $G_{\mathrm{GT}}=-1.106$, meaning 2.106 times the anchor error. For scale, decoding the true target-view state under uniform context gives $G_{\mathrm{GT}}=0.510$. The relative transplant's modest pixel-error reduction does not by itself establish clean relighting: as the qualitative outputs show, it can coincide with the transfer of donor-view structure.

**Result.** Absolute transplantation tends to reproduce the donor view at varying levels of sharpness rather than relight the requested target view. Relative transplantation leaves a blurred, subtraction-like imprint of donor-view structure.

The animation below cross-fades the observed donor image with relative-transplant decodings using an amplified contribution, $\lambda\in[1.5,2.0]$, to make the residual donor structure easier to see.

<p align="center">
  <img src="relative_state_transplant_cycle_alpha_025.gif" alt="Blend between the donor image and relative-transplant decoding" width="100%">
</p>

This visual resemblance is an interpretation of the qualitative result, not direct evidence that a relative nuisance representation literally implements image subtraction. The experiment more conservatively shows that its lighting-dependent changes cannot be transplanted between views as clean lighting edits. Additional interpolation results appear in [Appendix B](#appendix-b-additional-transplantation-results).

# 3. Can minimal post-training adaptation introduce lighting control?

## 3.1 Coarse in-distribution control is possible

We add a deliberately small lighting-conditioning path inspired by [LumiTokens](https://arxiv.org/abs/2608.18215). Illumination is restricted to a single Gaussian lobe described by mean world-space direction $\mu=[\mu_x,\mu_y,\mu_z]$, angular standard deviation $\sigma$, and energy $E$:

$$
x_\ell=(\mu,\log\sigma,\log E).
$$

_This is not a good general prior for illumination but it let's us forego the complications that come with richer representations like spherical harmonics or similar._

Rather than adding lighting tokens as in LumiTokens, this representation modulates the adaptive RMSNorm scales of RayDer's transformer layers. The experiment is a targeted feasibility check, not a general relighting system: training uses only nine OpenIllumination objects, approximate single-lobe illumination, no natural backgrounds, and a limited set of 17 views.

The adapted model learns a coarse target-light condition, expressed mainly through diffuse changes in color, intensity, and approximate spatial position.

<p align="center">
  <img src="rms_norm_only_fine_tune_ex1.1.png" alt="Adapted model results for object 4 under multiple target lights" width="100%">
</p>

Without a perceptual loss, in-distribution performance is:

| Metric | Zero-init | Adapted | Change |
| --- | ---: | ---: | ---: |
| Target MSE | 0.0515 | 0.0193 | -62.6% |
| Difference MSE | 0.0542 | 0.0264 | -51.4% |
| Target PSNR | 13.22 dB | 17.54 dB | +4.33 dB |
| Correct-light top-1 | 12.5% | 42.9% | 6/14 cases |

“Zero-init” is the model with the new conditioning path silenced; “adapted” allows that path to change normalization scales. Correct-light top-1 ranks the correct condition against several distractor light conditions using $\operatorname{MSE}(\operatorname{pred}_{\omega_i},I_\omega)$. It is successful when no tested distractor produces an image closer to the requested target. The mean in-distribution rank is approximately 1.92.

On the two objects held out entirely from training, top-1 is 50% over 16 cases and mean rank worsens to 4.5. Object 12 appears to be particularly difficult, reaching only 25% top-1 with mean rank 7.

<p align="center">
  <img src="rms_norm_only_fine_tune_ex2.2.png" alt="Adaptation results for held-out object 12" width="80%">
</p>

<p align="center">
  <img src="rms_norm_only_fine_tune_ex2.1.png" alt="Adaptation results for held-out object 20" width="80%">
</p>

## 3.2 Perceptual loss improves in-distribution fidelity, not held-out ranking

Naive normalization modulation substantially reduces novel-view sharpness. Adding an LPIPS perceptual loss with weight $\lambda=0.05$ empirically preserves structure better.

<p align="center">
  <img src="nvs_retention_fine_tune_percp_ex1.png" alt="Novel-view synthesis retention with perceptual loss" width="100%">
</p>

| Metric | Zero-init | Adapted | Change | Change over MSE-only |
| --- | ---: | ---: | ---: | ---: |
| Target MSE | 0.0515 | 0.0178 | -65.5% | -2.9% |
| Difference MSE | 0.0542 | 0.0229 | -57.7% | -6.3% |
| Target PSNR | 13.22 dB | 17.91 dB | +4.69 dB | +0.36 dB |
| Correct-light top-1 | 12.5% | 71.4% | 10/14 cases | +28.5 pp |
| Perceptual loss | 0.765 | 0.646 | -15.5% | — |
| Perceptual source | 0.446 | 0.363 | -18.7% | — |
| Perceptual target | 0.542 | 0.465 | -14.2% | — |

The improvement is confined to the in-distribution evaluation. Held-out-object top-1 remains 50%, while mean rank changes slightly from 4.5 to 4.75 because object 12 degrades from rank 7 to 7.5.

For comparison, novel-view outputs without perceptual loss are shown below. They illustrate both the dataset's initial distribution shift and the loss of sharpness caused by naive scale modulation.

<p align="center">
  <img src="nvs_retention_fine_tune_ex1.1.png" alt="Novel-view synthesis retention without perceptual loss" width="100%">
</p>

## 3.3 A privileged affine oracle remains stronger on average

To determine whether the learned model does more than globally adjust color and intensity, we compare it with a deliberately strong affine oracle. The oracle sees the model's decoding of the target view under source illumination and fits three global RGB gains and three biases directly against the desired image. It is therefore privileged: unlike the adapted model, it has access to the target during fitting. Its value is diagnostic rather than as a fair deployable baseline. It answers the question whether the best possible global color and intensity adjustment on the existing prediction of a view beats the learned light conditioning that can in principle adjust color and intensity spatially.

| Object | Camera | Learned MSE | Affine-oracle MSE | Learned wins |
| --- | --- | ---: | ---: | :---: |
| 12 | A4 | 0.005734 | 0.003278 | No |
| 12 | A4 | 0.008136 | 0.005349 | No |
| 12 | B4 | 0.008813 | 0.002946 | No |
| 12 | B4 | 0.008856 | 0.002117 | No |
| 12 | C4 | 0.010593 | 0.001469 | No |
| 12 | C4 | 0.009625 | 0.001024 | No |
| 20 | A4 | 0.005011 | 0.006478 | Yes |
| 20 | A4 | 0.015695 | 0.014986 | No |
| 20 | B4 | 0.008599 | 0.009522 | Yes |
| 20 | B4 | 0.011268 | 0.007908 | No |
| 20 | C4 | 0.007863 | 0.008105 | Yes |
| 20 | C4 | 0.006073 | 0.005128 | No |

| Mean learned MSE | Mean affine-oracle MSE |
| ---: | ---: |
| 0.008855 | **0.005692** |

**Result.** The adapted model beats the oracle in 3 of 12 individual cases, all on object 20, but is worse on average. The adaptation therefore exhibits some spatial behavior beyond a global affine adjustment, yet the aggregate result does not establish an advantage over that simpler privileged transformation.

## 3.4 What the adaptation establishes

The post-adaptation experiment demonstrates that a small conditioning path can steer RayDer toward requested illumination in its training domain. It does not yet demonstrate robust relighting:

- the training set contains only seven training objects and two completely held-out objects;
- the requested illumination is restricted to one approximate Gaussian lobe plus background energy;
- held-out-object ranking does not improve with perceptual loss;
- conditioning impairs novel-view fidelity, particularly sharpness; and
- average error remains above the privileged affine oracle.

The most promising next step would likely be to introduce light tokens explicitly, in-line with the LumiTokens approach and test again on the small dataset. Since that dataset at least provides rapid iterability. Once a cheap enough yet capacity sufficient approach is identified here, one should likely consider:
1. Impose the appropriate geometric equivariance on the lighting representation. In particular, rotating RayDer's inferred scene coordinate frame should transform the represented illumination accordingly. Rotating camera and lighting jointly should leave the rendered image invariant.
2. Expand the post-training data until the model can produce spatially non-affine, geometrically stable, and multi-view-consistent lighting changes on scenes resembling RayDer's pretraining distribution. Real-domain generalization should be evaluated on controlled paired-illumination captures where direct accuracy metrics are possible. An example would be the (MIT Multi-Illumination dataset)[https://projects.csail.mit.edu/illumination/].

# 4. Adaptation method

This section records the architectural and training details supporting the preceding result.

## 4.1 AdaRMSNorm conditioning

RayDer already uses RMSNorm conditioning. A two-layer MLP projects the requested light representation $x_\ell=(\mu,\log\sigma,\log E)$ once into a 128-dimensional feature:

$$
z_\ell=\operatorname{MLP}(x_\ell).
$$

Zero-initialized linear projections add a light-specific scale residual to each adaptive RMSNorm:

$$
c_\ell^{(i)}=A_\ell^{(i)}z_\ell,
$$

$$
\operatorname{RMSNorm}\!\left(x,1+L^{(i)}x_\text{cond}+c_\ell^{(i)}\right),
$$

where $L^{(i)}x_\text{cond}$ is RayDer's existing adaptive conditioning. The light-specific modulation is applied only to target/rendering tokens, not to camera-estimation or source-view tokens.

## 4.2 Data split and training examples

The adaptation dataset contains nine OpenIllumination objects, each rendered under 256 randomly sampled Gaussian lobes. A constant 10% background contribution from all available lights avoids completely black shadow regions. Each lobe is approximated by projecting the 142 available point lights onto it.

Seventeen views are available across camera tracks A, B, and C. One randomly selected view is held out for every training object. Objects 12 and 20 are held out from training entirely.

Each training example contains three source views under the same source illumination $\ell_s$ and a corresponding target view requested under $\ell_t$. Camera conditioning is performed analogous to RayDer's pretraining setup. The model infers camera poses from all views under the source illumination. OpenIlluminations exact camera poses are not used.

## 4.3 Aligning illumination with RayDer's inferred coordinates

OpenIllumination supplies light positions in its calibrated world coordinates, while RayDer reconstructs the target camera in its own inferred coordinate system. Therefore our samples mean light direction $\mu$ is in OpenIllumination calibrated world space. A best-fit rotation $Q$ maps the calibrated directions into RayDer's inferred frame.

Each camera pose suggests

$$
Q_i=R_i^\text{inf}(R_i^\text{cal})^\top,
$$

and the aggregate matrix is

$$
M=\sum_i R_i^\text{inf}(R_i^\text{cal})^\top.
$$

Given the singular value decomposition $M=U\Sigma V^\top$, the closest proper rotation is

$$
Q=U\operatorname{diag}\!\left(1,1,\det(UV^\top)\right)V^\top.
$$

This rotation, corrected to exclude reflection, is applied to $\mu$.

## 4.4 Training objective

The model minimizes

$$
L=L_\text{target}+\gamma L_\text{source}+\beta L_\Delta
+\lambda L_\text{perceptual},
$$

with a difference-image term

$$
L_\Delta=\operatorname{SMSE}\!\left(
\hat I_{\ell_t}-\hat I_{\ell_s},
I_{\ell_t}-I_{\ell_s}
\right).
$$

Here $\hat I_{\ell_t}$ and $\hat I_{\ell_s}$ are the predicted target- and source-light images. The masked error is

$$
\operatorname{SMSE}(\hat I,I;m)
=\operatorname{MSE}_m(\hat I,I)
+\eta\operatorname{MSE}_{\neg m}(\hat I,I),
\qquad \eta=0.05.
$$

OpenIllumination object masks are used so that background pixels contribute only weakly. $L_\text{target}$ and $L_\text{source}$ are the corresponding masked losses. $L_\text{perceptual}$ is optional LPIPS; the MSE-only configuration uses $\lambda=0$, and the perceptual experiment uses $\lambda=0.05$.

# Appendix A: Additional probe results

<div align="center">
  <img src="r_sq_general_transfer.png" alt="R-squared for illumination inference under different held-out conditions" width="100%">
  <p>Figure A1: R² for independent and contextual illumination inference under different held-out conditions.</p>
</div>

## Refitting on anchored representations

The main evaluation trains on absolute representations and applies anchoring only at evaluation time. The table below instead refits the probe on
$z_{ov}^{\mathrm{anchor}}(\omega)=z_{ov}(\omega)-z_{ov}(\omega_0)$.

RayDer nuisance features benefit from refitting: $R^2$ improves by 2.05 and 0.74 for independent and contextual inference, respectively, and MAE falls. RayDer token features are effectively unchanged. Most of the benefit therefore comes from subtracting the target-domain reference during evaluation; relearning the probe on anchored source features adds little for token and DINO representations, though it helps the nuisance representations somewhat.

| Representation | Frozen $R^2$ | Refit $R^2$ | $\Delta R^2$ | Frozen MAE | Refit MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| RayDer token — independent | -2.391 | -2.301 | 0.090 | 0.2087 | 0.2094 |
| RayDer token — context | -0.271 | -0.249 | 0.022 | 0.1330 | 0.1327 |
| RayDer nuisance — independent | -9.888 | -7.836 | 2.051 | 0.3302 | 0.3126 |
| RayDer nuisance — context | -4.182 | -3.438 | 0.744 | 0.2522 | 0.2369 |
| DINOv3 — per image | -0.357 | -0.267 | 0.090 | 0.1404 | 0.1345 |
| Linear OLAT pixels | -4.782 | -5.257 | -0.474 | 0.2043 | 0.2074 |

# Appendix B: Additional transplantation results

<div align="center">
  <img src="relative_state_interpolation.png" alt="Interpolation between donor state and anchored target state" width="100%">
  <p>Figure B1: Interpolation between the target state under uniform illumination and the donor's representation under the requested illumination. Ground-truth sources appear at the far left and intended targets at the far right.</p>
</div>
