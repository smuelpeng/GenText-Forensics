# GenText-Forensics 最小调试分发包

这是一个可以单独拿出来调试的 300 条验证集小包，用于快速验证 GenText-Forensics 防御赛道流程。包里已经包含本地数据、评测脚本、最基础的 DocShield CCT baseline，以及基于 DashScope/Qwen-VL API 的最小运行框架。

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
│   ├── cct_prompt.py          # DocShield 风格的 6-stage CCT prompt
│   ├── postprocess.py         # 提取 report，并把 Qwen 坐标缩放回原图尺寸
│   └── run_docshield_api.py   # 最小 DashScope API 推理脚本
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

常用参数可以通过环境变量覆盖：

```bash
MODEL=qwen3-vl-flash NUM_WORKERS=8 MAX_TOKENS=8192 ./run_api_debug.sh
PROMPT_MODE=simple ./run_api_debug.sh
```

默认使用：

- 模型：`qwen3-vl-flash`
- prompt：`cct`
- 并发：`4`
- 最大输出：`8192` tokens

推理结果会写到：

```text
outputs/raw/docshield_api_val_300.jsonl
```

每一行输出里：

- `raw_output`：最终 Markdown 鉴伪报告，用于评测
- `raw_output_full`：模型原始输出，方便排查格式问题
- `thinking`：抽取出来的 `<think>` 内容，只用于调试，不用于提交

## 本地评测

跑完 API 推理后，执行快速本地评测：

```bash
cd debug_distribution
./eval_debug.sh
```

评测结果会写到：

```text
outputs/eval/docshield_api_val_300.json
outputs/eval/docshield_api_val_300.csv
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
4. 先小并发跑完整 300 条：`NUM_WORKERS=2 ./run_api_debug.sh`
5. 本地评测：`./eval_debug.sh`
6. 查看 `outputs/eval/docshield_api_val_300.json`

## 注意事项

- 这里的评测脚本是本地 proxy，不是 CodaBench 隐藏官方评测程序；适合用来迭代和做回归检查。
- 这里的 DocShield 是根据公开论文方法做的 proxy baseline，不包含官方未公开的代码和权重。
- `postprocess.py` 会把 Qwen API 输出的 grounding 坐标缩放回原图尺寸，再交给评测脚本。
- `outputs/` 下的推理和评测结果可以随时删除重跑。
