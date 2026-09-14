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

