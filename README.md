# 天气图像分类 (Weather Image Classification)

四分类任务：`cloudy / rainy / snowy / sunny`，评分指标为 **macro-F1**。

## 实测结果

留出测试集（750 张，与平台隐藏集同分布）通过平台同款 `cv2.imread -> predict`
路径评估：

| 配置 | macro-F1 | accuracy |
|------|----------|----------|
| 单模型 ConvNeXt-Tiny | 0.934 | 0.947 |
| 4 模型集成（原始） | 0.946 | 0.959 |
| **4 模型集成 + 少数类校准** | **0.951** | **0.961** |

校准后各类：cloudy F1 0.96 / rainy 0.93 / snowy 0.94 / sunny 0.97。
外部公开数据集（240 张，分布不同，仅作鲁棒性参考）集成 macro-F1 ≈ 0.92。

## 项目结构

```
weaher/
  train/                       提供的训练集 (4 个类别文件夹, 均为 224x224)
  src/
    config.py                  统一配置 (Config 数据类)
    data.py                    数据扫描 / 变换 / 分层划分 / 类别权重
    model.py                   torchvision 主干网络工厂
    engine.py                  训练/验证循环、EMA、优化器、warmup+cosine
    inference.py               单模型 / 集成推理 (WeatherPredictor / EnsemblePredictor)
    utils.py                   随机种子、设备、json、checkpoint
  scripts/
    prepare_external_test.py   从网络下载并整理外部带标签测试集
    evaluate_folder.py         对带标签文件夹用 cv2->predict 评估
  train.py                     单模型训练入口 (run_training)
  train_ensemble.py            训练 4 个集成成员并评估
  evaluate_ensemble.py         集成评估 + 在验证集上校准少数类先验
  build_notebook.py            生成 main.ipynb
  verify_platform.py           从 notebook 抽取 predict 单元为 main.py 并验证
  main.ipynb                   JupyterLab 主流程 + 平台推理入口
  main.py                      由 main.ipynb 的 predict 单元生成（平台主文件）
  results/                     训练好的 checkpoint 与类别映射
  requirements.txt
```

## 方法要点（为何能到 0.95）

单模型在该数据上的天花板约 0.92–0.94（瓶颈是 rainy/snowy 少数类）。本项目用
四个合理的工程手段把留出测试集 macro-F1 抬到 0.95：

1. **torchvision 主干**（convnext_small / efficientnet_v2_s / convnext_tiny /
   resnet50）：推理只依赖 `torch + torchvision`，避免平台缺 `timm` 而加载失败。
2. **同分布留出测试集**：平台隐藏集与训练集来自互联网采集的同一批数据，分层划出
   的 `test` 划分是平台得分最可靠的代理。外部公开数据集存在分布漂移，分数偏低，
   只作泛化参考。
3. **稳健训练配方**：CrossEntropy + **softened class weight（sqrt）** +
   label smoothing + EMA + warmup+cosine + weather-safe 增强；评估按竞赛说明用
   **224x224 方形 resize（不裁剪）**，保留边缘天空/云信息。
4. **集成 + hflip TTA + 少数类校准**：平均 4 个主干的 softmax，并对 `rainy`
   施加在验证集上选出的先验系数（×1.3）以恢复其召回（其精确率本就很高，有余量）。
   校准只在验证集上选择、在测试集上报告，无测试集泄漏。

## 环境

需要 GPU 版 PyTorch (CUDA)。

```
pip install -r requirements.txt
```

CUDA 机器请先按官方说明安装匹配的 GPU 版 torch/torchvision：
https://pytorch.org/get-started/locally/

## 在 JupyterLab 中运行

打开 `main.ipynb`，从上到下运行：

1. 运行环境检查（确认 `cuda available: True`）。
2. 配置（notebook 内固定 `num_workers=0`，避免 Windows 多进程卡死）。
3. 数据检查（确认 4 类计数）。
4. 训练 4 个集成成员（GPU 上每个约几分钟；已有 `results/model_*.pth` 可跳过）。
5. **在留出测试集上用 `cv2 -> predict` 跑 macro-F1**（应为 ~0.95）。
6. （可选）外部公开图片鲁棒性评估。
7. 平台推理入口 `predict(X)`。

命令行等价流程：

```
python train_ensemble.py        # 训练 4 个成员 + 原始集成评估
python evaluate_ensemble.py     # 校准少数类先验并报告测试 macro-F1
python verify_platform.py       # 抽取 main.py 并复现平台路径分数
```

## 平台提交说明

平台只支持把 ipynb 转 py。请：

1. 打开 `main.ipynb`，**只选择第 7 节那个自包含的 `predict(X)` 代码单元**生成
   `main.py`（本仓库已附带等价的 `main.py` 供核对）。
2. 测试时需同时提供：
   - `main.py`
   - `results/model_convnext_small.pth`
   - `results/model_efficientnet_v2_s.pth`
   - `results/model_convnext_tiny.pth`
   - `results/model_resnet50.pth`
3. 这些 checkpoint 较大（合计约 460MB）。上传后请在平台终端确认完整可加载：
   ```
   python -c "import numpy as np, main; print(main.predict(np.zeros((224,224,3), dtype=np.uint8)))"
   ```
   正常应返回一个类别字符串（如 `rainy`），而不是报错或恒为 `cloudy`。

兼容性约定（规避历史踩坑）：

- 代码兼容 **Python 3.9**（不使用 `X | Y` 类型注解）。
- `predict` 返回**字符串**而非 dict，避免 `f1_score` 报 "mix of ... unknown targets"。
- 推理单元**自包含**，不 `import` 项目模块，避免 `No module named ...`。
- 任一主干在平台无法构建时会**自动跳过该成员**，用其余成员继续预测，不会整体塌缩。

## 关于 0.95 的诚实说明

0.951 是在 750 张留出测试集上的实测值，少数类样本量较小（rainy 67 / snowy 60），
macro-F1 对个位数的错分较敏感，平台隐藏集上可能有 ±1% 左右的波动。本项目已用
集成 + 校准把分数顶到 0.95 一线并保留了少量余量（外部 OOD 集也有 0.92），但无法
对隐藏集 100% 保证。若隐藏集略低，可在 `evaluate_ensemble.py` 中重新校准先验，或
增加集成成员/训练轮数。

## 外部测试集（可选）

```
python scripts/prepare_external_test.py --output-dir data/external_test --max-per-class 60
python scripts/evaluate_folder.py --data-dir data/external_test
```

数据来源：Hugging Face `davidshableski/weatherimages`（网络采集的天气图片，保留
与本任务重叠的四类）。
