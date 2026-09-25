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

## GPU numerics: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

Moebius hits the window in two places:

- **The SDXL KL-f8 VAE**: 20 encoder convs per encode (two encodes per inpaint) and 31 decoder
  convs at 512². The fp16 weights meet fp32 activations, so these convs compute in fp32.
- **One UNet conv per denoise step**: the `up_blocks.1` upsampler, 640→640 at 64², batch 2.

MoebiusGate pins the CPU stream by default, so none of this showed there.

Every stride-1 3×3 VAE conv is now a `WinogradFreeConv2d` with a route (`MoebiusConvRoute`). The
UNet `Upsample2D` routes through `WinogradFreeConv2d.conv`.

**Defaults: encoder `.conv3d`, UNet upsampler `.conv3d`, decoder `.winograd`.** The decoder follows
the fleet audit policy for FLUX-class KL decoders: its fp32 loss is below 8-bit visibility, so
parity lanes opt in.

Measurements: 512² DIV2K photo, against the CPU lane, production dtypes.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Encode mean (the UNet's image / masked-image latents) | 1.7e-2 · **max 5.1** | 8.7e-5 · max 2e-2 |
| Decode | 1.7e-3 · 66.8 dB · max 3.2e-2 | 7.9e-5 · 93.6 dB |
| UNet upsampler conv (probe, fp16 weights × fp32 activations) | 6.6e-3 | 1.7e-6 |
| Time: encode / decode at 512² | 507 ms / 926 ms | +32 ms / +142 ms |

The UNet upsampler route adds about 7 ms per step.

Controls and tests:

- Environment override: `MOEBIUS_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- `swift test --filter MoebiusMLXTests.WinogradProbeTests` is weight-free.
- `MOEBIUS_PARITY=1 MOEBIUS_DIR=<dir with vae.safetensors> swift test -c release -Xswiftc
  -enable-testing --filter MoebiusMLXTests.VAEGPULaneTests` compares the GPU and CPU lanes.

## License

MIT (port). Upstream: weights MIT, reference code Apache-2.0 (hustvl).
