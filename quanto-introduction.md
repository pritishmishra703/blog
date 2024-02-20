---
title: "Quanto: a pytorch quantization toolkit"
thumbnail: /blog/assets/quanto-intro/thumbnail.png
authors:
- user: dacorvo
---

# Quanto: a pytorch quantization toolkit

Quantization is a technique to reduce the computational and memory costs of evaluating Deep Learning Models by representing their weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).

Reducing the number of bits means the resulting model requires less memory storage, which is crucial for deploying Large Language Models on consumer devices.
It also allows to take advantage of specific optimizations for lower bitwidth datatypes, such as `int8` of `float8` matrix multiplications on CUDA devices.

Many open-source libraries are available to quantize pytorch Deep Learning Models, including pytorch own [quantization tools](https://pytorch.org/docs/stable/quantization.html).

These tools are however often restricted to specific model configurations and devices.

Today, we are excited to introduce [🤗 quanto](https://github.com/huggingface/quanto), a versatile pytorch quantization toolkit, that provides several unique features:

- available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA and MPS),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from a float model to a dynamic to a static quantized model,
- supports quantized model serialization as a `state_dict`,
- supports not only `int8` weights, but also `int2` and `int4`,
- supports not only `int8` activations, but also `float8`.

### Quantized tensors

At the heart of quanto are Tensor subclasses that corresponds to:
- the projection using a `scale` of a source Tensor into the optimal range for a given quantization type,
- the mapping of projected values to the destination type.

For floating-point destination types, the mapping is done by the native pytorch cast (i.e. `Tensor.to()`).

For integer destination types, the mapping is a simple rounding operation (i.e. `torch.round()`).

The goal of the projection is to increase the accuracy of the conversion by minimizing the number of:
- saturated values (i.e. mapped to the destination type min/max),
- zeroed values (because they are below the smallest number that can be represented by the destination type)

For efficiency, the projection is symmetric for `8-bit` quantization types, i.e. it is centered around zero.
Symmetric quantized Tensors are usually compatible with many standard operations.

For lower bitwidth quantization types, such as `int2` or `int4`, the projection is affine, i.e. it uses a `zeropoint` to shift the
projected values, which allows a better coverage of the quantization range. Affine quantized Tensors are typically harder to work with
and require custom operations.

## Quantized modules

Quanto provides a generic mechanism to replace torch modules (`torch.nn.Module`) by `quanto` modules that are able to process `quanto` tensors.

Quanto modules dynamically convert their `weight` parameter until a model is frozen, which slows down inference a bit but is
required if the model needs to be tuned (a.k.a Quantization Aware Training).

Module `bias` are not quantized because to be added to the result of an `matmul` or `conv2d` operation, they would need to be converted
with a scale equal to the product of the input and weight scales.

This would lead in most cases to a ridiculously small scale, and conversely require a very high quantization bitwidth to avoid clipping.
Typically, with `int8` inputs and weights, biases need to be quantized with at least `12` bits of precision, i.e. in `int16`.
Since most biases are today `float16`, this is a waste of time.

Activations are dynamically quantized using static scales (defaults to the range `[-1, 1]`). The model needs to be calibrated to evaluate
the best activation scales (using a momentum).

The following modules can be quantized:

- [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (QLinear).
Weights are always quantized, and biases are not quantized. Inputs and outputs can be quantized.
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) (QConv2D).
Weights are always quantized, and biases are not quantized. Inputs and outputs can be quantized.
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
Weights and biases are __not__ quantized. Outputs can be quantized.

## Quantization workflow

Quanto is available as a pip package.

```sh
pip install quanto
```

Quanto does not make a clear distinction between dynamic and static quantization: models are always dynamically quantized,
but their weights can later be "frozen" to static values.

A typical quantization workflow consists of the following steps:

**1. Quantize**

The first step converts a standard float model into a dynamically quantized model.

```python
quantize(model, weights=quanto.qint8, activations=quanto.qint8)
```

At this stage, only the inference of the model is modified to dynamically quantize the weights.

**2. Calibrate (optional if activations are not quantized)**

Quanto supports a calibration mode that allows to record the activation ranges while passing representative samples through the quantized model.

```python
with calibration(momentum=0.9):
    model(samples)
```

This automatically activates the quantization of the activations in the quantized modules.

**3. Tune, aka Quantization-Aware-Training (optional)**

If the performance of the model degrades too much, one can tune it for a few epochs to recover the float model performance.

```python
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data).dequantize()
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()
```

**4. Freeze integer weights**

When freezing a model, its float weights are replaced by quantized integer weights.

```python
freeze(model)
```

Please refer to the [examples](https://github.com/huggingface/quanto/tree/main/examples) for instantiations of that workflow.

## Performances

TO BE COMPLETED

## Integration in 🤗 transformers

TO BE COMPLETED
