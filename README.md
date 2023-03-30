# SE3Transformer


It is a fork of [NVIDIA's SE(3)-Transformer implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer).
I made some minor modifications, including
- removal of torch.cuda.nvtx.nvtx_range
- addition of the `nonlinearity` argument to `NormSE3`, `SE3Transformer`, and so on.
- addition of some basic network implementations using SE(3)-Transformer.

---
## Installation
#### for CPU only
```bash
pip install git+http://github.com/huhlim/SE3Transformer
```
#### for CUDA (GPU) usage
1. Install [DGL](https://www.dgl.ai/pages/start.html) library with CUDA support
```bash
# This is an example with cudatoolkit=11.3.
# Set a proper cudatoolkit version that is compatible with your CUDA drivier and DGL library.
conda install dgl -c dglteam/label/cu113
# or
pip install dgl -f https://data.dgl.ai/wheels/cu113/repo.html
```
2. Install this package
```bash
pip install git+http://github.com/huhlim/SE3Transformer
```

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
