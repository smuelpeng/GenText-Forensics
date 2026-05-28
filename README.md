# GenText-Forensics 最小调试分发包

这是一个可以单独拿出来调试的 300 条验证集小包，用于快速验证 GenText-Forensics 防御赛道流程。包里已经包含本地数据、评测脚本、最基础的 DocShield CCT baseline，以及基于 DashScope/Qwen-VL API 的运行框架。

说明：GitHub 仓库版本不包含 `data/` 目录，避免把 300 张图片和标注数据推到代码仓库。完整本地分发包中的 `data/` 目录需要单独复制或解压到本目录后，才能直接运行 API 推理和评测。

## 目录内容

```text
debug_distribution/
├── data/
│   ├── val_300.jsonl          # 300 条验证样本，路径已改成相对路径
│   ├── images/                # 复制出来的验证图片
│   ├── masks/                 # FORGED 样本对应的 mask，有则复制
│   └── reports/               # 复制出来的 GT 报告
├── baselines/DocShield/
│   ├── cct_prompt.py              # DocShield 风格的 6-stage 单 prompt baseline
│   ├── staged_prompts.py          # staged evidence-grounded prompts
│   ├── postprocess.py             # report/grounding 解析工具
│   ├── run_docshield_api.py       # 单 prompt DashScope API 推理脚本
│   └── run_staged_docshield_api.py # staged API 推理脚本
├── scripts/
│   └── eval_competition_aligned.py
├── outputs/
│   ├── raw/                   # API 推理输出
│   └── eval/                  # 本地评测结果
├── requirements.txt
├── environment.yml
├── run_api_debug.sh
└── eval_debug.sh
```

## 数据概况

- 样本数：300，来自 `gentext-forensics/data/splits/val_1000.jsonl`
- 标签分布：`{"AUTHENTIC": 148, "FORGED": 152}`
- 语言分布：`{"ar": 43, "en": 61, "id": 39, "ms": 43, "th": 52, "zh": 62}`
- 带 mask 的样本：152
- 图片、mask、GT report 都已经复制到 `data/` 目录下，不再依赖原始 `/mnt/pfs/...` 绝对路径。

## 环境安装

方式一：使用 `venv` 和 `pip`。

```bash
cd debug_distribution
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

方式二：使用 `conda`。

```bash
conda env create -f environment.yml
conda activate gentext-debug
```

说明：如果只是跑快速评测，`eval_debug.sh` 默认使用 `--skip-bertscore`，这时不强依赖 `bert-score`。如果要跑完整 BERTScore 版本，则需要安装 `bert-score`、`torch`、`transformers`。

如果系统 Python 缺少 `numpy` 等评测依赖，可以用本地虚拟环境运行：

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pillow tqdm
PYTHON=.venv/bin/python ./eval_debug.sh
```

## 运行 API Baseline

不要把 API key 提交到 git。可以任选一种方式提供 key：

- 在当前目录放一个 `api-key.txt`
- 设置环境变量 `DASHSCOPE_API_KEY`
- 运行时传入 `API_KEY_FILE=/path/to/key`

推荐直接用环境变量：

```bash
cd debug_distribution
DASHSCOPE_API_KEY=... ./run_api_debug.sh
```

常用参数可以通过环境变量覆盖。默认先跑 staged baseline 的 60 条样本：

```bash
MODEL=qwen3.6-35b-a3b NUM_WORKERS=8 MAX_TOKENS=8192 ./run_api_debug.sh
MAX_SAMPLES=3 ./run_api_debug.sh
```

默认使用：

- pipeline：`staged`
- 模型：`qwen3.6-35b-a3b`
- 并发：`8`。本轮全量实验中 `16` 并发在部分 staged donor 上触发过 429，默认保守使用 `8`
- 最大输出：`8192` tokens
- 样本数：`60`
- API key：`/Users/penpen/Desktop/api-key.txt`
- forged 默认风险阈值：`80`，可用 `FORGED_RISK_THRESHOLD` 覆盖
- Stage-1 语言感知风险阈值：`ar=70,id=75`，可用 `FORGED_RISK_THRESHOLDS` 覆盖；空字符串表示只使用默认阈值
- OCR layout 辅助：默认读取 `OCR_LAYOUT_CACHE_MODEL=qwen-vl-ocr` 的 native DashScope OCR cache，把文字框作为 CCT 的感知锚点；该 runner 不会在线调用 OCR，缺 cache 时只记录 cache miss
- OCR transcript 辅助模型：默认关闭；只在消融时使用 `OCR_TRANSCRIPT_MODEL=qwen-vl-ocr`，不让它承担逻辑推理、验证或最终判断；多语言纯文本补充应使用 native `multi_lan` cache
- 当前第一版 API key 已实测可调用 `qwen-vl-ocr`；`qwen-vl-ocr-latest` 仍会返回 `Model.AccessDenied`，不要作为默认模型
- OCR API key：默认复用 `API_KEY_FILE=/Users/penpen/Desktop/api-key.txt`；`OCR_TRANSCRIPT_API_KEY_FILE` 只作为受控消融覆盖项，正常运行不要设置
- OCR layout cache：默认 `outputs/cache/ocr_layouts/qwen-vl-ocr/`；固定测试集应先缓存 300 张 native OCR boxes，避免 staged CCT 运行时重复调用 OCR。可用 `REQUIRE_OCR_LAYOUT_CACHE=1` 强制缺 cache 即失败
- OCR transcript 注入范围：默认关闭 transcript；native OCR layout 默认只在 grounding/normalization 阶段作为 `auxiliary_ocr_spans` 定位锚点使用，不进入 Stage-1/2/3 的判别链。可用 `OCR_LAYOUT_TO_STAGE1=1` 做 Stage-1 OCR 注入消融，但当前 60 样本结果显示它会增加假阳性
- Stage-1 OCR 模型替换：`OCR_MODEL` 仅用于消融，不建议用 Qwen-OCR 直接替换主 Stage-1；此前 60 样本实验显示它会显著增加假阳性
- 坐标归一：模型阶段输出若被检测为常见的 `0-1000` 视觉坐标，会先投影到原图像素坐标，再进入 grounding/report 后处理
- grounding box 扩张：默认 `GROUNDING_BOX_SCALE_X=3.5`、`GROUNDING_BOX_SCALE_Y=4.0`，用于把模型偏紧的异常中心框扩展到更接近文本区域的定位框
- benign-error reviewer：默认开启。只在单个异常、低复杂度、解释主要来自 OCR/扫描/字体/排版等生产性瑕疵且缺少强篡改信号时，把低质量 forged 报告降级为 authentic。可用 `DISABLE_BENIGN_REVIEWER=1` 做消融。
- taxonomy prompt：默认关闭。`docs/forgery_taxonomy.md` 总结了 GT 聚合诊断得到的伪造/误判范式，可用 `ENABLE_TAXONOMY_PROMPTS=1` 作为实验开关注入各阶段 prompt；当前 60 样本消融低于默认 v9，因此不作为默认路径。

OCR 使用规则见 `docs/ocr_usage_rules.md`。默认先预热 native layout cache，供 CCT 使用：

```bash
.venv/bin/python baselines/DocShield/cache_ocr_layouts.py \
  --input-jsonl data/val_300.jsonl \
  --model qwen-vl-ocr \
  --api-mode dashscope-native \
  --ocr-task advanced_recognition \
  --api-key-file /Users/penpen/Desktop/api-key.txt \
  --num-workers 8
```

可选预热 native multilingual transcript cache，只用于 Stage-1 语言/文本补充或 transcript 消融；该结果只有纯文本，不含坐标，不承担逻辑推理：

```bash
.venv/bin/python baselines/DocShield/cache_ocr_transcripts.py \
  --input-jsonl data/val_300.jsonl \
  --model qwen-vl-ocr \
  --api-mode dashscope-native \
  --ocr-task multi_lan \
  --cache-dir outputs/cache/ocr_transcripts_multilan \
  --api-key-file /Users/penpen/Desktop/api-key.txt \
  --num-workers 8
```

如需在 staged pipeline 中只把 `multi_lan` transcript 作为 Stage-1 补充输入：

```bash
API_KEY_FILE=/Users/penpen/Desktop/api-key.txt \
OCR_TRANSCRIPT_MODEL=qwen-vl-ocr \
OCR_TRANSCRIPT_CACHE_DIR=outputs/cache/ocr_transcripts_multilan \
REQUIRE_OCR_TRANSCRIPT_CACHE=1 \
OCR_LAYOUT_CACHE_MODEL=qwen-vl-ocr \
REQUIRE_OCR_LAYOUT_CACHE=1 \
MAX_SAMPLES=60 NUM_WORKERS=8 ./run_api_debug.sh
```

Qwen-OCR layout cache 只保存 OCR 文本框，不读取 label、GT report、mask，也不做逻辑推理。坐标应优先使用 DashScope 原生 `advanced_recognition` 任务。该路径会返回官方 `ocr_result.words_info`，包含文字、四点 `location` 和 `rotate_rect`；脚本会转换成原图像素 `xyxy` 后再给 viewer 和 CCT 使用。`openai-prompt` 模式只作为消融和兼容路径，不作为 `qwen-vl-ocr` 坐标默认调用方式。

使用已缓存 native OCR layout 跑 staged pipeline：

```bash
API_KEY_FILE=/Users/penpen/Desktop/api-key.txt \
OCR_LAYOUT_CACHE_MODEL=qwen-vl-ocr \
REQUIRE_OCR_LAYOUT_CACHE=1 \
MAX_SAMPLES=60 NUM_WORKERS=8 ./run_api_debug.sh
```

查看 OCR/Layout 坐标和 grounding 坐标：

```bash
PYTHON=.venv/bin/python scripts/build_multi_result_viewer.sh
python3 -m http.server 8765 --bind 127.0.0.1
```

然后打开 `http://127.0.0.1:8765/tools/ocr_viewer/`。页面中的 `Qwen OCR` 是独立 OCR layout cache；`VLM layout` 是 staged Stage-1 输出，不再混称为 OCR。`Version Comparison` 表会横向列出 v9/v13/v16/v17/v18/v19 等已有结果的指标、判定、框数量和样本级 issue；`All final boxes` 可以把多个版本的最终定位框同时叠加到同一张图上。该页面会把模型常见的 `0-1000` OCR/Layout 坐标投影到原图像素坐标；GT report boxes 和 mask 只作为本地诊断 overlay，不会进入模型 prompt。

当前 best 路径在 base staged CCT 之后叠加 GT-blind 后处理：`scripts/review_false_positives_v2.py` 对低复杂度、弱证据 forged 报告做保守降级；`scripts/rescue_from_donor_reports.py` 只在 donor run 产生高置信、强触发证据时救回 base 漏检；`scripts/remap_crop_donor_report.py` 把 zoom-crop donor 的 grounding 坐标映射回原图。推理阶段不读取 label、GT report 或 mask；GT 只用于本地评测和误差诊断。

推理结果会写到：

```text
outputs/raw/staged_docshield_api_val_60.jsonl
```

每一行输出里：

- `raw_output`：最终 Markdown 鉴伪报告，用于评测
- `raw_output_full`：模型原始输出，方便排查格式问题
- `stage_outputs`：OCR/Layout、证据候选、交叉验证、grounding、最终报告等中间结果，只用于调试，不用于提交

当前 staged baseline 采用无训练的 DocShield/CCT proxy：

1. OCR/Layout
2. Evidence extraction
3. Cross-cue validation
4. Spatial grounding
5. Report synthesis

最终报告会做四个确定性后处理：补齐评测需要的报告结构标记；当模型判为 `FORGED` 但 `RISK_SCORE` 低于当前语言对应阈值时降级为 `AUTHENTIC`，用于降低低置信假阳性；执行 benign-error reviewer，过滤由 OCR/扫描/字体/排版瑕疵触发的低质量假阳性；对保留的 forged grounding box 做 GT-blind 扩张，缓解模型输出框偏紧、mask 覆盖不足的问题。默认语言阈值、reviewer 和 box 扩张都不读取 GT 语言、标签、mask 或报告文本。

如果需要回退到旧的单 prompt CCT baseline：

```bash
PIPELINE=prompt MODEL=qwen3-vl-flash ./run_api_debug.sh
```

如果 60 条结果明显更好，再跑完整 300 条：

```bash
MAX_SAMPLES=0 ./run_api_debug.sh
MAX_SAMPLES=0 ./eval_debug.sh
```

如果需要为实验保留不同输出文件，可以覆盖输出路径：

```bash
OUTPUT_JSONL=outputs/raw/my_experiment_60.jsonl ./run_api_debug.sh
RAW_JSONL=outputs/raw/my_experiment_60.jsonl \
OUT_JSON=outputs/eval/my_experiment_60.json \
OUT_CSV=outputs/eval/my_experiment_60.csv \
./eval_debug.sh
```

## 本地评测

跑完 API 推理后，执行快速本地评测：

```bash
cd debug_distribution
./eval_debug.sh
```

评测结果会写到：

```text
outputs/eval/staged_docshield_api_val_60.json
outputs/eval/staged_docshield_api_val_60.csv
```

如果本地依赖和模型缓存齐全，也可以跑包含 BERTScore 的版本：

```bash
python scripts/eval_competition_aligned.py \
  --gt-jsonl data/val_300.jsonl \
  --raw-jsonl outputs/raw/docshield_api_val_300.jsonl \
  --output outputs/eval/docshield_api_val_300_bertscore.json \
  --output-csv outputs/eval/docshield_api_val_300_bertscore.csv \
  --include-samples \
  --allow-empty-loc
```

## 快速检查流程

新同学第一次使用时，建议按这个顺序：

1. 确认能进入目录：`cd debug_distribution`
2. 安装环境：`pip install -r requirements.txt`
3. 设置 API key：`export DASHSCOPE_API_KEY=你的key`
4. 先跑 3 条烟测：`MAX_SAMPLES=3 ./run_api_debug.sh`
5. 本地评测：`./eval_debug.sh`
6. 确认 3 条无误后跑 60 条：`./run_api_debug.sh && ./eval_debug.sh`
7. 如果 60 条明显优于旧 baseline，再跑 300 条：`MAX_SAMPLES=0 ./run_api_debug.sh && MAX_SAMPLES=0 ./eval_debug.sh`

## 注意事项

- 这里的评测脚本是本地 proxy，不是 CodaBench 隐藏官方评测程序；适合用来迭代和做回归检查。
- 这里的 DocShield 是根据公开论文方法做的 proxy baseline，不包含官方未公开的代码和权重。
- staged baseline 推理时只把图片和模型阶段输出传给模型，不会把 `label`、`report_text`、`mask_path` 等 GT 字段传入 prompt。
- staged baseline 的最终 report 正文语言应跟随文档主语言；中文、泰语、阿拉伯语样本不应统一输出英文。
- `outputs/` 下的推理和评测结果可以随时删除重跑。
