# Generative Recommenders

Repository hosting code for ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations`` ([ICML'24 paper](https://proceedings.mlr.press/v235/zhai24a.html)) and related code, where we demonstrate that the ubiquitously used classical deep learning recommendation paradigm (DLRMs) can be reformulated as a generative modeling problem (Generative Recommenders or GRs) to overcome known compute scaling bottlenecks, propose efficient algorithms such as HSTU and M-FALCON to accelerate training and inference for large-scale sequential models by 10x-1000x, and demonstrate scaling law for the first-time in deployed, billion-user scale recommendation systems.

## Getting started

We recommend using `requirements.txt`. This has been tested with Ubuntu 22.04, CUDA 12.4, and Python 3.10.

```bash
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyTorch based on official instructions. Then,

```bash
pip3 install gin-config pandas fbgemm_gpu torchrec tensorboard
```

## Experiments

### Public Experiments

To reproduce the public experiments in our paper (traditional sequential recommender setting, Section 4.1.1) on MovieLens and Amazon Reviews in the paper, please follow these steps:

#### Download and preprocess data.

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

A GPU with 24GB or more HBM should work for most datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```

Other configurations are included in configs/ml-1m, configs/ml-20m, and configs/amzn-books to make reproducing these experiments easier.

#### Verify results.

By default we write experimental logs to exps/. We can launch tensorboard with something like the following:

```bash
tensorboard --logdir ~/generative-recommenders/exps/ml-1m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/ml-20m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/amzn-books-l50/ --port 24001 --bind_all
```

With the provided configuration (.gin) files, you should be able to reproduce the following results (verified as of 04/15/2024):

**MovieLens-1M (ML-1M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------| --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2853           | 0.1603          | 0.5474          | 0.2185          | 0.7528          | 0.2498          |
| BERT4Rec      | 0.2843 (-0.4%)   | 0.1537 (-4.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2811 (-1.5%)   | 0.1648 (+2.8%)  |                 |                 |                 |                 |
| HSTU          | 0.3097 (+8.6%)   | 0.1720 (+7.3%)  | 0.5754 (+5.1%)  | 0.2307 (+5.6%)  | 0.7716 (+2.5%)  | 0.2606 (+4.3%)  |
| HSTU-large    | **0.3294 (+15.5%)**  | **0.1893 (+18.1%)** | **0.5935 (+8.4%)**  | **0.2481 (+13.5%)** | **0.7839 (+4.1%)**  | **0.2771 (+10.9%)** |

**MovieLens-20M (ML-20M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2889           | 0.1621          | 0.5503          | 0.2199          | 0.7661          | 0.2527          |
| BERT4Rec      | 0.2816 (-2.5%)   | 0.1703 (+5.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2813 (-2.6%)   | 0.1730 (+6.7%)  |                 |                 |                 |                 |
| HSTU          | 0.3273 (+13.3%)  | 0.1895 (+16.9%) | 0.5889 (+7.0%)  | 0.2473 (+12.5%) | 0.7952 (+3.8%)  | 0.2787 (+10.3%) |
| HSTU-large    | **0.3556 (+23.1%)**  | **0.2098 (+29.4%)** | **0.6143 (+11.6%)** | **0.2671 (+21.5%)** | **0.8074 (+5.4%)**  | **0.2965 (+17.4%)** |

**Amazon Reviews (Books)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------|---------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.0306           | 0.0164          | 0.0754          | 0.0260          | 0.1431          | 0.0362          |
| HSTU          | 0.0416 (+36.4%)  | 0.0227 (+39.3%) | 0.0957 (+27.1%) | 0.0344 (+32.3%) | 0.1735 (+21.3%) | 0.0461 (+27.7%) |
| HSTU-large    | **0.0478 (+56.7%)**  | **0.0262 (+60.7%)** | **0.1082 (+43.7%)** | **0.0393 (+51.2%)** | **0.1908 (+33.4%)** | **0.0517 (+43.2%)** |

for all three tables above, the ``SASRec`` rows are based on [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) but with the original binary cross entropy loss
replaced with sampled softmax losses proposed in [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039). These rows are reproducible with ``configs/*/sasrec-*-final.gin``.
The ``BERT4Rec`` and ``GRU4Rec`` rows are based on results reported by [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) -
note that the comparison slightly favors these two, due to them using full negatives whereas the other rows used 128/512 sampled negatives. The ``HSTU`` and ``HSTU-large`` rows are based on [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152); in particular, HSTU rows utilize identical configurations as SASRec. ``HSTU`` and ``HSTU-large`` results can be reproduced with ``configs/*/hstu-*-final.gin``.

### Synthetic Dataset / MovieLens-3B

We support generating synthetic dataset with fractal expansion introduced in https://arxiv.org/abs/1901.08910. This allows us to expand the current 20 million real-world ratings in ML-20M to 3 billion.

To download the pre-generated synthetic dataset:

```bash
pip3 install gdown
mkdir -p tmp/ && cd tmp/
gdown https://drive.google.com/uc?id=1-jZ6k0el7e7PyFnwqMLfqUTRh_Qdumt-
unzip ml-3b.zip && rm ml-3b.zip
```

To generate the synthetic dataset on your own:

```bash
python3 run_fractal_expansion.py --input-csv-file tmp/ml-20m/ratings.csv --write-dataset True --output-prefix tmp/ml-3b/
```

### Efficiency experiments

``ops/triton`` contains triton kernels needed for efficiency experiments. ``ops/cpp`` contains efficient CUDA kernels. In particular, ``ops/cpp/hstu_attention`` contains the attention implementation based on [FlashAttention V3](https://github.com/Dao-AILab/flash-attention) with state-of-the-art efficiency on H100 GPUs.

#### HSTU Ultra attention benchmark

``ops/hstu_ultra.py`` provides named, model-derived HSTU Ultra self-attention
configurations backed by ``ops/pytorch/pt_hstu_ultra.py``. The accompanying
benchmark can compare the PyTorch reference with the Triton forward and
backward kernels, including the semi-local attention configuration.

For the model architecture, see the
[HSTU Ultra paper](https://dl.acm.org/doi/10.1145/3770855.3818322).
``modules/hstu_ultra.py`` builds complete semi-local or target-aware layer
stacks from those configurations. Each layer includes input normalization, UVQK
projection, HSTU Ultra attention, output normalization and projection, and the
residual connection. Pass ``fp8_addmm_fwd=True`` to
``hstu_ultra_stack_configs`` to use FP8 for both projection GEMMs. For repeated
inference, call ``stack.prepare_fp8_weights()`` after moving or loading the
model to cache the row-wise quantized projection weights.

The end-to-end integration uses the target-aware HSTU Ultra stack inside
the existing DLRMv3 data, multitask-loss, optimizer, checkpoint, and eager
inference paths. Start a BF16 training run with:

```bash
python3 -m generative_recommenders.dlrm_v3.train.train_ranker \
  --dataset kuairand-1k --hstu-ultra
```

Add ``--hstu-ultra-fp8`` to use FP8 projection GEMMs (it also enables the
Ultra stack). Eager inference accepts the same flags and prepares the FP8
weight caches after loading the checkpoint:

```bash
python3 -m generative_recommenders.dlrm_v3.inference.main \
  --dataset debug --hstu-ultra-fp8
```

Cached incremental decoding and scripted inference export are not included yet.

The complete-stack benchmark measures all configured layers, including input
normalization, UVQK projection, attention, output projection, and residuals:

```bash
python3 -m generative_recommenders.ops.benchmarks.hstu_ultra_stack_bench \
  --config-name hstu_ultra_semi_local --provider all --sequence-lengths 512
```

Representative forward-only results for the complete configured stacks on one
NVIDIA H100 96 GB GPU are shown below. Each result is the median of three
trials with 10 warmup iterations and 100 timed iterations per trial.

| Configuration | Layers | Sequence length | BF16 | FP8 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| ``hstu_ultra_semi_local`` | 8 | 512 | 6.036 ms | 5.325 ms | 1.13x |
| ``hstu_ultra_semi_local`` | 8 | 1024 | 5.911 ms | 5.211 ms | 1.13x |
| ``hstu_ultra_semi_local`` | 8 | 2048 | 6.071 ms | 5.396 ms | 1.13x |
| ``hstu_ultra`` | 12 | 768 | 9.104 ms | 8.056 ms | 1.13x |
| ``hstu_ultra`` | 12 | 1024 | 9.375 ms | 7.768 ms | 1.21x |
| ``hstu_ultra`` | 12 | 1536 | 9.469 ms | 7.565 ms | 1.25x |

The FP8 provider uses cached weights, fused LayerNorm-to-FP8 quantization for
the UVQK projection, fused output normalization/gating/dropout-to-FP8
quantization, and a persistent Triton row-wise FP8 GEMM. The semi-local and
target-aware configurations are measured independently.

```bash
python3 -m generative_recommenders.ops.benchmarks.hstu_ultra_bench \
  --config-name hstu_ultra_semi_local --kernel triton --batch-size 1 --mode fwd
```

Use ``--config-name hstu_ultra`` for target-aware full causal
self-attention. Pass ``--sequence-lengths`` to override the conservative
reference defaults; long production lengths may require substantial GPU memory
because the PyTorch implementation materializes dense attention matrices.

Representative forward-pass results on one NVIDIA H100 96 GB GPU are shown
below. Measurements used BF16, batch size 1, a 1980 MHz clock, a 700 W power
cap, 10 warmup iterations, 100 timed iterations, and the median of three runs.

| Configuration | Sequence length | PyTorch | Triton | Speedup |
| --- | ---: | ---: | ---: | ---: |
| ``hstu_ultra_semi_local`` | 512 | 0.570 ms | 0.253 ms | 2.25x |
| ``hstu_ultra_semi_local`` | 1024 | 0.592 ms | 0.254 ms | 2.32x |
| ``hstu_ultra_semi_local`` | 2048 | 0.587 ms | 0.255 ms | 2.30x |
| ``hstu_ultra`` | 768 | 0.517 ms | 0.256 ms | 2.02x |
| ``hstu_ultra`` | 1024 | 0.540 ms | 0.252 ms | 2.15x |
| ``hstu_ultra`` | 1536 | 0.528 ms | 0.253 ms | 2.08x |

This benchmark measures attention only with precomputed BF16 Q/K/V tensors. It
does not include the UVQK or output projection GEMMs, so the FP8 addmm path is
not exercised by these numbers.

These figures are illustrative rather than release guarantees; rerun the
benchmark on the deployment hardware and software stack when making tuning
decisions.

#### FP8 linear reference and benchmark

``ops/fp8.py`` provides row-wise E4M3 quantization and FP8 addmm. It dispatches
to a persistent Triton implementation when requested and retains the portable
PyTorch ``torch._scaled_mm`` implementation and CPU fallback. Both paths
support on-the-fly or prequantized inputs and a full-precision backward for
training. HSTU's Triton path also fuses activation quantization into its input
LayerNorm and output postprocessing kernels.
Set ``fp8_in_addmm_fwd=True`` on ``hstu_compute_uqvk`` and
``hstu_compute_output`` to use the FP8 path for the HSTU input and output
projections. ``STULayerConfig(fp8_addmm_fwd=True)`` enables both projection
GEMMs for an HSTU layer.

```bash
python3 -m generative_recommenders.ops.benchmarks.fp8_addmm_bench \
  --provider all --m 4096 --k 512 --n 512
```

## DLRM-v3

We have created a DLRM model using HSTU and have developed benchmarks for both training and inference to faciliate production RecSys use cases.

#### Run model training with 4 GPUs

```bash
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/train/train_ranker.py --dataset debug --mode train
```

#### Run model inference with 4 GPUs

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python -m pip install .

LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/inference/main.py --dataset debug
```

### DLRM-v3 on Intel XPU (research harness)

`run_dlrm_v3_xpu.sh` starts a DLRM-v3 run on an Intel GPU, in training or
inference, and records what it ran on. It is a research harness for finding
what is still missing on XPU. It is not a CI gate and reports no performance
numbers. What it has found so far is in [FINDINGS.md](FINDINGS.md). The
one-off diagnostic scripts behind those findings are indexed in
[PROBES.md](PROBES.md).

#### Set up the environment

Run inside the XPU container. It provides a user-mode driver new enough for
torch 2.14 and oneAPI under `/opt/intel/oneapi` (FINDINGS A-07). Install the
four pinned components with `install_dlrmv3.sh`. It lives in the DLRM-v3
workspace next to `src/`, not in this repository. It installs torch
`2.14.1+xpu`, fbgemm-gpu-cpu `1.9.0`, fbgemm-xpu from torchlib-xpu, the TorchRec
fork, and this repository, and it downloads MovieLens-1M into `datasets/`.

The fork branches are not pushed. Move them to the test host as git bundles,
which keeps the commit SHAs:

```bash
# on the development host
git -C src/generative-recommenders bundle create /tmp/generative-recommenders.bundle \
    probes/dlrm-v3-xpu-fbgemmv1.9.0
scp /tmp/generative-recommenders.bundle <test-host>:/tmp/

# on the test host, inside the container
source /opt/intel/oneapi/setvars.sh
GENREC_BUNDLE=/tmp/generative-recommenders.bundle bash install_dlrmv3.sh
```

`TORCHREC_BUNDLE` and `TORCHLIB_XPU_BUNDLE` work the same way. The installer
checks out the pinned SHA and stops if it differs.

#### Run

From the root of this repository:

```bash
bash run_dlrm_v3_xpu.sh --mode train-eval     # training on one XPU (default: phase 1)
bash run_dlrm_v3_xpu.sh --mode infer          # unquantized inference on one XPU (MLPerf LoadGen)
```

| Option | Values |
| --- | --- |
| `--mode` | `train`, `eval`, `train-eval` (default), `streaming-train-eval`, or `infer` |
| `--phase` | `1` (default): one process. `2`: one XPU under `torchrun`. `3`: all XPUs over XCCL; refuses to start unless `DLRMV3_ALLOW_MULTI_TILE=1` (FINDINGS A-03) |
| `--device` | `xpu` (default), or `cpu` for the reference run (phase 1, training only) |
| `--gin`, `--dataset` | gin file (default: `train/gin/movielens_1m_xpu.gin` or `inference/gin/movielens_1m_xpu.gin`) and dataset (default `movielens-1m`) |

Environment: `DATA_DIR` (directory containing `data/ml-1m/`, default
`<repo>/datasets`), `RUN_DIR` (default `exps/xpu_runs/<timestamp>-...`),
`ONEAPI_ROOT`, `HSTU_EMBEDDING_DIM` (default 64) and `HASH_SIZE` (default
1,000,000).

Each run folder contains:

- `manifest.json`: the revisions of torchlib-xpu, FBGEMM, TorchRec and
  generative-recommenders, with dirty state, plus the runtime and environment.
  It is written before the run starts, so failed runs are recorded too.
- `run.gin` and `operative_config.gin`: the config the run was given and the
  config it actually used.
- `run.log`.
- `losses.csv` for training (loss per step), or the `mlperf_log_*` files for
  inference.

Before relying on the torchlib-xpu SHA in the manifest, read FINDINGS H-16.

#### Compare against CPU

```bash
bash run_dlrm_v3_xpu.sh --mode train --device cpu        # CPU reference, same model
python compare_loss_curves.py <xpu-run-dir> <cpu-run-dir> [--repeat <xpu-run-dir-2>]
```

It writes `loss_diff.txt` and `loss_diff.json` into the XPU run folder. It
exits 1 if a loss is not finite, or if the two runs were built from different
revisions or configs.

For inference, set `run.compute_eval = True` in a gin file that includes the
inference gin. Then score the predictions:

```bash
python generative_recommenders/dlrm_v3/inference/accuracy.py --path <run-dir>/mlperf_log_accuracy.json
```

## License
This codebase is Apache 2.0 licensed, as found in the [LICENSE](LICENSE) file.

## Contributors
The overall project is made possible thanks to the joint work from many technical contributors (listed in alphabetical order):

Adnan Akhundov, Bugra Akyildiz, Shabab Ayub, Alex Bao, Renqin Cai, Jennifer Cao, Xuan Cao, Guoqiang Jerry Chen, Lei Chen, Li Chen, Sean Chen, Xianjie Chen, Huihui Cheng, Weiwei Chu, Ted Cui, Shiyan Deng, Nimit Desai, Fei Ding, Shilin Ding, Francois Fagan, Lu Fang, Leon Gao, Zhaojie Gong, Fangda Gu, Liang Guo, Liz Guo, Jeevan Gyawali, Yuchen Hao, Daisy Shi He, Michael Jiayuan He, Yu He, Samuel Hsia, Jie Hua, Yanzun Huang, Hongyi Jia, Rui Jian, Jian Jin, Rafay Khurram, Rahul Kindi, Changkyu Kim, Yejin Lee, Fu Li, Han Li, Hong Li, Shen Li, Rui Li, Wei Li, Zhijing Li, Lucy Liao, Xueting Liao, Emma Lin, Hao Lin, Chloe Liu, Jingzhou Liu, Xing Liu, Xingyu Liu, Kai Londenberg, Yinghai Lu, Liang Luo, Linjian Ma, Matt Ma, Yun Mao, Bert Maher, Ajit Mathews, Matthew Murphy, Satish Nadathur, Min Ni, Jongsoo Park, Colin Peppler, Jing Qian, Lijing Qin, Jing Shan, Alex Singh, Timothy Shi,  Yu Shi, Dennis van der Staay, Xiao Sun, Colin Taylor, Shin-Yeh Tsai, Rohan Varma, Omkar Vichare, Alyssa Wang, Pengchao Wang, Shengzhi Wang, Wenting Wang, Xiaolong Wang, Yueming Wang, Zhiyong Wang, Wei Wei, Bin Wen, Carole-Jean Wu, Yanhong Wu, Eric Xu, Bi Xue, Hong Yan, Zheng Yan, Chao Yang, Junjie Yang, Wen-Yun Yang, Ze Yang, Zimeng Yang, Yuanjun Yao, Chunxing Yin, Daniel Yin, Yiling You, Jiaqi Zhai, Keke Zhai, Yanli Zhao, Zhuoran Zhao, Hui Zhang, Jingjing Zhang, Lu Zhang, Lujia Zhang, Na Zhang, Rui Zhang, Xiong Zhang, Ying Zhang, Zhiyun Zhang, Charles Zheng, Erheng Zhong, Zhao Zhu, Xin Zhuang.

For the initial paper describing the Generative Recommender problem formulation and the algorithms used, including HSTU and M-FALCON, please refer to ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations``([ICML'24 paper](https://dl.acm.org/doi/10.5555/3692070.3694484), [slides](https://icml.cc/media/icml-2024/Slides/32684.pdf)).
