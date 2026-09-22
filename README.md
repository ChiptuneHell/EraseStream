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
hf download zhuobai/StreamErase model.pt --local-dir model_weights
```

### Inference
```bash
python inference.py
```

### 测试时间
```bash
python time.py
```

# 制作视频 demo

可参考下面这个链接的 demo.mp4 (demo.py) ，做个类似差不多的，重点要展示我们生成的速度很快，超过实时生成
https://github.com/guandeh17/Self-Forcing

和用户互动的 demo 可参考下面这个链接，目前可以先去除掉 SAM 提取mask的环节，直接让用户选择我们默认提供的 mask
https://huggingface.co/spaces/jixin0101/ObjectClear<br>
https://github.com/sczhou/ProPainter

两个 demo 网页也可以做到一起，一个展示，一个互动

## 第一版 Web Demo

模型环境和权重准备好后，在项目根目录运行：

```bash
python web_demo.py
```

然后打开 `http://127.0.0.1:5001`。页面使用 `test_input/video` 和
`test_input/mask` 中的同名视频作为示例，选择视频后点击“开始生成”即可看到
原视频、mask、擦除结果和实时速度统计。结果文件会写入 `web/results/`。

后续制作 mask 可参考下面这个链接
https://github.com/sakshamsingh1/sam3_mask_annotation_tool

