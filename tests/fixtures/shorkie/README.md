# Pretrained Shorkie regression fixture

`params.json` is the published pretrained configuration. The NumPy fixture is
its fold-0 T0-track mean over each of the 896 output bins, generated with the
unmodified loader at commit `0e37d14` and PyTorch 2.11.0 on CPU (one thread).

Seed PyTorch with zero, build and load the model, then generate the one-hot
probe using `torch.randint(4, (1, 16384))`. Model construction consumes random
numbers, so preserve this ordering. Run in evaluation mode without gradients.
The test checks exact equality and skips when the published checkpoint is absent.
The full before/after outputs were also checked with `np.array_equal`.
