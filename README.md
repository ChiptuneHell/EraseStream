# EraseStream

## 评测模型 EffectErase 在两个 benchmark 的表现，最终给我两个表格就行
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
