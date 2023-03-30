# SE3Transformer


It is a fork of [NVIDIA's SE(3)-Transformer implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer).
I made some minor modifications, including
- removal of torch.cuda.nvtx.nvtx_range
- addition of the `nonlinearity` argument to `NormSE3`, `SE3Transformer`, and so on.
- addition of some basic network implementations using SE(3)-Transformer.

---

## Code Snippets
- `se3_transformer.LinearModule`: `LinearSE3` and `NormSE3`
https://github.com/huhlim/SE3Transformer/blob/b74f7079cf6694fdf9b0185ce15ba4ed6e1ec747/se3_transformer/snippets.py#L14-L64
- `se3_transformer.InteractionModule`: A wrapper of SE3Transformer
https://github.com/huhlim/SE3Transformer/blob/b74f7079cf6694fdf9b0185ce15ba4ed6e1ec747/se3_transformer/snippets.py#L67-L118


---

## Usage
- LinearModule + InteractionModule
https://github.com/huhlim/SE3Transformer/blob/b74f7079cf6694fdf9b0185ce15ba4ed6e1ec747/example/example.py#L1-L84
In this example, 
  - A fully connected graph is created with random coordinates
  - Input features: 8 scalars and 4 vectors
  - Output features: 2 scalars and 1 vector
  - LinearModule: two `LinearSE3` with `NormSE3`, returns 16 scalars and 8 vectors. 
  - InteractionModule: two layers of attention blocks with two heads, takes the output of the LinearModule as `node_feats` and no `edge_feats`.
