# mlx-moebius-swift

[Moebius](https://github.com/hustvl/Moebius) (hustvl, 0.22B) **image inpainting / object removal**
on Swift/MLX for Apple silicon — a LambdaNetworks-based latent-diffusion inpainter, ported
component-by-component against the PyTorch reference and parity-locked at every level.

**Weights:** [`mlx-community/Moebius-Places2-fp16`](https://huggingface.co/mlx-community/Moebius-Places2-fp16)
(fp16, chosen by measured parity; bf16 rejected at ~9× worse per forward).

Two products:
- **`MoebiusMLX`** — engine-agnostic core: the UNet (LλMI linear attention, depthwise-separable
  convs, GLU-MBConv FFN), the SDXL KL-f8 `AutoencoderKL`, DDIM, and the full pipeline.
- **`MLXMoebius`** — the [MLXEngine](https://github.com/xocialize/mlx-engine-swift) `imageInpaint`
  `ModelPackage` (surface `moebius-inpaint`): resize to the model's hard 512², fill, paste back
  through a blurred mask at the original resolution. metaData: `seed` / `cfgScale` / `paste`.

## Parity (vs the PyTorch reference, CPU stream fp32)

| component | rel |
|---|---|
| LλMI self / cross | 2.2e-07 / 1.9e-07 |
| DepthwiseSeparableConv · GLUMBConv | 2.8e-07 · 3.0e-07 |
| resnet · transformer block · down/up block | 6.8e-07 · 7.5e-07 · 1.4e-06 / 3.3e-06 |
| **UNet end-to-end (226M)** | **1.1e-06** |
| DDIM add_noise / step | 3.0e-08 / 6.0e-08 |
| full 19-step pipeline (decoded image) | 3.6e-04 — the VAE's measured fp32 floor |

~4 s per 512² inpaint (19 DDIM steps × CFG-2 = 38 UNet forwards) on an M-series GPU, release.

## Gates

Parity gates are an **executable**, not XCTests (the SPM test host cannot resolve the mlx-swift
metallib). Build with `--build-system swiftbuild`:

```bash
swift run -c release --build-system swiftbuild moebius-gate --pipeline-gate --no-cpu \
  --model-dtype fp16 --checkpoint <dir>/unet.safetensors --vae-checkpoint <dir>/vae.safetensors
```

Offline conformance (CAN / MAT / manifest): `swift test --build-system swiftbuild`.

## Things that will silently produce wrong output

Documented in-source where they live; headlines: 512×512 is **structurally** hard (spatially-baked
`rel_pos_emb`); mask is white=remove; the denoiser input is `noisy(4) | mask(1) | masked(4)` with
the mask in the middle; 20 requested DDIM steps run **19** (strength 0.99) starting from the
*clean* latents noised at t=900; two different activations coexist per block (SiLU at resnet level,
ReLU inside the depthwise-separable convs); GroupNorm eps differs one level apart (1e-5 resnets,
1e-6 transformer norm); BatchNorm is written as explicit inference math on purpose.

## License

MIT (port). Upstream: weights MIT, reference code Apache-2.0 (hustvl).
