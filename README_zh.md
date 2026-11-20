# eKNN 反应预测

eKNN 先用图神经网络从反应组分的 SMILES 预测目标值，再检索嵌入空间中相近的训练反应，用其残差加权修正预测。本仓库包含 v0.3.0 的模型实现、训练入口，以及 HiTEA、光氧化还原、PhotoCatDB 和 Michael 反应的研究脚本。

## 运行 CPU 示例

在仓库根目录执行以下命令。示例使用 Python 3.10 和 Bash。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-tested.txt -e .
python examples/quickstart.py
```

示例生成 40 条带人工标签的记录，其中 32 条用于训练、8 条用于测试。预测 CSV 和指标 JSON 位于 `results/synthetic_demo/model/` 下。人工标签只用于验证流程。再次运行时，可用 `--out-dir` 指定新的目录。

## 用自己的数据训练

准备训练集和测试集两个 CSV，两者使用相同的 SMILES 列和数值目标列。以下示例以反应物、催化剂和收率三列为例。

```bash
eknn-train --dataset my_reactions \
  --train-csv train.csv --test-csv test.csv \
  --smiles-cols reactant_smiles catalyst_smiles --target-col Yield \
  --backbone gcn --no-cache-graphs --out-dir results/my_reactions \
  --set train.device=cpu train.seed=42 data.seed=42
```

程序在内层验证集上选择近邻参数，再用完整训练集重新拟合模型，并用训练残差建立最终检索库。若要评估对新底物的泛化能力，应预先按底物划分外层训练集和测试集。内层分组验证可通过 `--group-col 列名 --group-val` 启用。注册数据集的本地目录可通过 `EKNN_DATA_DIR` 指定。[数据格式](docs/DATA.md)

## 研究脚本与测试

[模型与检索库](eknn_reaction/) 提供图编码、训练和残差修正。[任务脚本](scripts/) 包含 HiTEA 按筛选批次校准、光氧化还原反应按底物和论文划分、PhotoCatDB 预测，以及 Aho 和 Michael 反应的数据准备。任务脚本需要对应的外部输入，CPU 示例和单元测试会自行生成所用数据。

```bash
python -m pip install -c requirements-tested.txt -e '.[research,dev]'
python -m pytest -q
```

本项目采用 [MIT 许可证](LICENSE)。署名说明见 [NOTICE](NOTICE)。
