# EraseStream

## 评测模型 EffectErase 在两个 benchmark 的表现，最终给我两个表格就行

### benchmark 下载指令
(1) ROSE-Bench <br>
```bash
hf download \
  --repo-type dataset \
  --include "Benchmark/*" \
  --local-dir ./Benchmark \
  Kunbyte/ROSE-Dataset
```
(2) VOR-Eval <br>
```bash
hf download \
  --repo-type dataset \
  --include "VOR-Eval.tar.gz.part_000" \
  --local-dir . \
  FudanCVL/EffectErase
```

### 评测脚本运行指令
```bash
python evaluate_ROSE-Bench.py
```
```bash
python evaluate_VOR-Eval.py
```

# Inference

### Installation
```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```

### Download Checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt --local-dir checkpoints
```

### Inference
```bash
python inference.py
```

# 制作视频 demo

可参考下面这个链接的 demo.mp4 (demo.py) ，做个类似差不多的，重点要展示我们生成的速度很快，超过实时生成
https://github.com/guandeh17/Self-Forcing

和用户互动的 demo 可参考下面这个链接，目前可以先去除掉 SAM 提取mask的环节，直接让用户选择我们默认提供的 mask
https://huggingface.co/spaces/jixin0101/ObjectClear<br>
https://github.com/sczhou/ProPainter

两个 demo 网页也可以做到一起，一个展示，一个互动

后续制作 mask 可参考下面这个链接
https://github.com/sakshamsingh1/sam3_mask_annotation_tool

