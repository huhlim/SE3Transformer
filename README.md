# SE3Transformer


It is a fork of [NVIDIA's SE(3)-Transformer implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer).
I made some minor modifications, including
- removal of torch.cuda.nvtx.nvtx_range
- addition of the `nonlinearity` argument to `NormSE3`, `SE3Transformer`, and so on.
- addition of some basic network implementations using SE(3)-Transformer.
