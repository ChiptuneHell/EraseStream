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

