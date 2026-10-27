# 商品评论评分预测（任务 2）

根据评论标题和正文预测 1～5 分评分，评价指标为 **RMSE，越小越好**。项目采用一套单模型方案：**Apache Spark 预处理 + DeBERTa-v3-base + 自主实现的回归头 + PyTorch DDP 训练**。

只使用 `title`、`comment` 作为模型输入，`rating` 为监督标签；`id` 仅用于生成提交文件。不做模型融合和用户／商品图建模。

## 1. 项目结构

```text
5434_project/
├── data/
│   ├── download.py                 # 已有的数据下载脚本
│   ├── clean_data.py               # 已有的独立清洗脚本
│   ├── raw_data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sampleSubmission.csv    # 注意实际文件名的大小写
│   └── cleaned_data/               # 已有的清洗结果
├── model/
│   ├── requirements.txt
│   ├── common.py                   # 文本清洗、硬件记录、JSON 工具
│   ├── prepare.py                  # Spark 并行处理、分组划分、分词缓存
│   ├── dataset.py                  # 内存映射、动态 padding、无重复分片
│   ├── network.py                  # 自定义池化和回归头
│   ├── runtime.py                  # CPU/单 GPU/多 GPU 运行环境
│   ├── train.py                    # 自定义训练循环和全局 RMSE
│   ├── predict.py                  # 最佳权重恢复、预测、提交检查
│   ├── test_pipeline.py            # 核心逻辑测试
│   ├── smoke_test.py               # 离线小模型端到端检查
│   └── artifacts/                 # 运行产物，已排除出 Git
└── README.md
```

以下命令均在项目根目录执行。路径基于脚本定位，服务器上的项目可以放在任意目录，不需要修改代码。

## 2. 环境安装

推荐正式环境：**Linux、Python 3.11、Java 17、7 张 NVIDIA RTX 3090**，至少 32 GB 内存，建议预留 30 GB 可用磁盘空间。也支持单卡，以及用于流程检查的 CPU。7 卡 DDP 的文档命令按单台服务器编写。

在 Ubuntu 上安装 Java（已安装 Java 17 可跳过）：

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless
java -version
```

创建 Python 环境，先安装 CUDA 12.1 版 PyTorch，再安装其他依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

这里固定 PyTorch、Transformers、Spark 等直接依赖版本。PyTorch CUDA wheel 需要服务器 NVIDIA 驱动兼容；CUDA Toolkit 的本地安装版本不等于 PyTorch wheel 使用的 CUDA 版本。

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPUs:', torch.cuda.device_count()); assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 7"
```

仅在 CPU/macOS 上检查流程时，不执行 CUDA wheel 的安装命令，直接安装 `requirements.txt`。Spark 仍然需要 Java 17。

## 3. 正式运行：三步

### 第一步：Spark 数据准备

将三个 CSV 放在 `data/raw_data/` 后运行：

```bash
python -m model.prepare
```

默认配置：`local[8]`、Spark driver 内存 8 GB、64 个分区、最大长度 256 tokens、约 95% 训练数据和 5% 验证数据、种子 42。第一次需要联网下载 Hugging Face 的 tokenizer；默认固定 DeBERTa commit `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`，实际 commit 会记录到 manifest，训练按相同 commit 加载主干。更换远程模型时需同时指定相应的 `--revision`。

`prepare.py` 直接从原始 CSV 开始，在 Spark 中统一处理训练和测试文本，包括移除 HTML 标签、解码实体、Unicode NFC 规范化、移除控制字符和合并空白。训练数据会移除非法评分、标题正文同时为空的样本及业务字段重复记录。测试样本全部保留，缺失文本转为空字符串。

按清洗后的“标题 + 正文”的哈希分组划分训练/验证集，因此重复文本、包括评分不同的相同文本，不会同时出现在训练集和验证集。比例为近似值，不是严格分层划分。数据会写成分片 Parquet，再一次性分词到磁盘缓存。后续训练按内存映射读取，避免将全部样本放入内存。

Spark 实际参与重新分区、并行清洗、去重、分组划分、统计和 Parquet 写出。原始单个多行 CSV 的读取并行度有限，读取后再分区执行后续任务。

成功完成后生成：

```text
model/artifacts/prepared/
├── manifest.json                  # 行数、模型 commit、配置、预处理耗时、硬件
├── parquet/{train,validation,test}/
├── tokenizer/
└── {train,validation,test}/
    ├── input_ids.npy              # int32 token ID；256 长度的全量缓存约 8 GB
    ├── lengths.npy
    ├── labels.npy
    └── ids.csv
```

最终 manifest 只在所有缓存写完后生成。中途失败时请给 `--output` 指定一个新的目录重新运行；程序拒绝覆盖非空产物目录。

### 第二步：7 卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc_per_node=7 \
  -m model.train --device cuda
```

默认训练 2 轮，每卡 batch size 为 16，全局 batch size 为 112，使用 FP16 AMP、AdamW、线性 warmup/衰减和梯度裁剪。编码器学习率 `2e-5`，回归头学习率 `1e-4`。不需要手动修改源码中的路径或 GPU 编号。

如果显存不足，使用以下命令替代上面的训练命令，保持全局 batch size 为 112：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc_per_node=7 \
  -m model.train --device cuda --batch-size 8 --accumulation-steps 2 \
  --gradient-checkpointing
```

每轮验证后按 RMSE 保存最好的权重：

```text
checkpoints/<run>/
├── best/
│   ├── encoder_config/config.json
│   └── weights.pt
└── training_report.json
```

单卡运行只需 `python -m model.train --device cuda`。首次训练会下载预训练主干；推理恢复已保存的 config 和完整权重，无需再次下载。训练中断后已保存的最佳权重仍可推理，但此实现不提供优化器断点续训；重训请指定新的 `--output`。

### 第三步：生成提交文件

测试集只有 10,000 条时，默认单 GPU 推理即可：

```bash
python -m model.predict --device cuda --checkpoint checkpoints/RUN_NAME/best
```

也可以使用 7 卡推理：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc_per_node=7 \
  -m model.predict --device cuda --checkpoint checkpoints/RUN_NAME/best
```

两种推理命令选择一种执行，输出为：

```text
model/artifacts/submission.csv
model/artifacts/submission.report.json
```

提交文件格式：

```csv
id,rating
0,4.621735
1,3.867201
```

上面仅为格式示例，不是真实预测。评分保留小数并裁剪到 `[1,5]`，不四舍五入为类别。程序按 `sampleSubmission.csv` 的 ID 顺序输出，并检查重复 ID、缺失 ID、样本数量、有限值及评分范围。**只向 Kaggle 上传 `submission.csv`。**

重新推理时可指定不同文件名，防止覆盖已有提交：

```bash
python -m model.predict --device cuda --checkpoint checkpoints/RUN_NAME/best --output model/artifacts/submission_v2.csv
```

## 4. 模型与自主实现内容

标题和正文作为 tokenizer 的两个输入段送入预训练 DeBERTa-v3-base。对最后一层输出进行带 attention mask 的均值池化，再经过以下自行编写的评分头：

```text
DeBERTa token 表示
→ mask 均值池化（padding 不参与）
→ Dropout → Linear(hidden,128) → GELU → Dropout → Linear(128,1)
→ 连续评分
```

训练优化 MSE：

```text
MSE = sum((prediction - rating)^2) / N
RMSE = sqrt(MSE)
```

训练中使用原始连续预测计算 MSE，验证与提交均将预测裁剪到 `[1,5]`。报告同时记录裁剪前 RMSE，避免评价口径不清楚。均值基线只使用训练部分的真实评分均值，在同一验证集上计算 RMSE。

第三方库提供预训练编码器、自动求导与分布式通信。项目自行实现了文本处理、无文本重叠划分、mask 池化、回归头、损失归一化、梯度累积、训练循环、全局 RMSE、最佳权重选择和提交关联检查；没有调用 Spark MLlib 的现成预测模型，也没有用 `Trainer` 一键训练。

多卡训练使用标准 `DistributedSampler`，为保持训练步数相同可能补齐少量样本；**验证和推理使用无补齐采样器，每条样本恰好计算一次**。验证先汇总所有进程的平方误差总和与样本数，再开根号，不平均各批或各 GPU 的 RMSE。

## 5. 运行报告与效率指标

程序自动记录实际执行机器的信息，不预填 3090 服务器的硬件信息或未测量的成绩。

| 要求 | 记录位置 / 字段 |
|---|---|
| Spark CPU、线程数和版本 | `prepared/manifest.json` 的 `hardware`、`spark_version`、`spark_master` |
| 服务器数量、CPU、线程数、GPU 型号和显存 | `run/training_report.json` 的 `hardware` |
| 每轮训练耗时 | `epochs[].train_seconds`，同步 GPU 后记录最慢进程时间 |
| 每轮验证耗时 | `epochs[].validation_seconds` |
| 每轮总耗时（含验证及权重保存） | `epochs[].epoch_wall_seconds` |
| 验证 RMSE、基线 RMSE、验证样本数 | `epochs[].validation` |
| 最佳验证 RMSE 与轮数 | `best_validation_rmse`、`best_epoch` |
| Spark + 分词缓存预处理耗时 | `preprocessing_seconds` |
| 完整训练阶段耗时 | `training_total_seconds` |
| 总离线耗时：预处理 + 训练 | `total_offline_seconds` |
| 全测试集推理总耗时及样本数 | `submission.report.json` 的 `inference_total_seconds`、`test_samples` |

预处理耗时包含 Spark 启停和首次 tokenizer 下载，训练总耗时包含主干加载/下载、验证和保存。推理总耗时包含模型加载、全部样本预测、分片合并和提交文件写出。数据下载脚本的网络下载时间不计入离线训练时间。报告中的总离线耗时针对这一次数据准备和训练运行，不把人工排错或多次实验时间混在一起。

## 6. 运行前快速检查

先检查核心逻辑，再用离线随机小模型验证端到端流程；小模型测试不下载正式预训练权重：

```bash
python -m unittest model.test_pipeline -v
python -m model.smoke_test
```

检查 2 进程 DDP 的验证聚合和分片推理：

```bash
python -m model.smoke_test --distributed
```

这些检查使用合成数据和随机初始化的小型 DeBERTa，只证明代码流程可运行，不代表全量训练效果或 GPU 性能。Spark 测试仍需 Java 17。需要保留测试日志时，添加 `--directory /tmp/rating-smoke-check`，该目录须为空。

也可以在正式数据的少量样本上试跑主模型，测试下载、显存与速度：

```bash
python -m model.prepare --limit-train 10000 --output model/artifacts/prepared_small
python -m model.train --device cuda --data-dir model/artifacts/prepared_small \
  --output model/artifacts/run_small --epochs 1
python -m model.predict --device cuda --checkpoint checkpoints/RUN_NAME/best --data-dir model/artifacts/prepared_small \
  --checkpoint model/artifacts/run_small/best --output model/artifacts/submission_small.csv
```

小规模样本按 ID 排序选取，只用于排错，不用于报告最终效果。正式训练请使用完整数据默认命令。

## 7. 可复现性与常见问题

- 随机种子默认 42，数据划分按文本哈希确定；实际模型 commit、训练参数和硬件会保存到报告。数据缓存一旦生成可复用。不同 GPU、CUDA 或依赖环境下仍可能有浮点差异，不承诺逐位一致。
- 推理严格检查缓存 manifest 与训练时是否相同。不要将另一次预处理得到的缓存与现有权重混用；新缓存需重新训练。
- `JAVA_GATEWAY_EXITED`：先确认 `java -version` 为 Java 17，必要时设置 `JAVA_HOME`。Spark worker 必须使用与 driver 相同的 Python；脚本默认将 `PYSPARK_PYTHON` 设为当前解释器。
- CUDA OOM：使用上面的 batch size 8、累积 2 步及梯度检查点命令；推理可降低 `--batch-size`。
- GPU 利用率低：缓存应放在本地 SSD；每个进程默认 2 个 DataLoader worker，可根据 CPU 核数调整 `--workers`，避免 7 个进程争抢过多线程。
- Hugging Face 下载失败：确认服务器能访问模型仓库，或将完整的本地预训练模型目录传给 `prepare --model-name /path/to/model`。本地目录在后续训练时仍须可读。
- 仅文本模型不会利用用户/商品 ID。项目效果以真实验证集 RMSE 和 Kaggle 得分为准，不使用流程测试成绩冒充正式结果。

## 8. 参考资料

- [DeBERTa-v3-base 官方模型说明](https://huggingface.co/microsoft/deberta-v3-base)
- [DeBERTaV3 论文](https://arxiv.org/abs/2111.09543)
- [Spark 3.5 CSV 数据源文档](https://spark.apache.org/docs/3.5.7/sql-data-sources-csv.html)
- [PyTorch DDP 官方教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch 历史版本安装命令](https://pytorch.org/get-started/previous-versions/)


## 服务器实测与 W&B（2026-10-26）

训练默认输出为项目根目录 `checkpoints/<UTC时间戳>/`；可以通过 `--output checkpoints/实验名` 指定空目录。最佳模型为其下 `best/weights.pt` 和 `best/encoder_config/`，训练报告与 wandb 数据也保存在同一实验目录。只保存最佳模型权重，尚不支持恢复优化器断点续训。推理必须使用 `--checkpoint checkpoints/实验名/best`。

W&B 只由 rank 0 初始化与记录，避免 7 个重复实验。默认 `--wandb-mode offline`，无需登录，离线数据保存在实验目录的 `wandb/`。记录超参数、全局 batch、每轮全局训练 RMSE、验证 RMSE/原始 RMSE/基线、样本数、训练和验证耗时、最佳轮次及本地权重路径；批次间隔指标 `train/rank0_running_rmse` 明确仅代表 rank 0。权重保存在本地，不自动上传。

在线记录先运行 `wandb login`，然后加 `--wandb-mode online --wandb-project 5434_project`，可选 `--wandb-entity 团队名 --wandb-name 实验名`。离线结果以后可用 `wandb sync checkpoints/实验名/wandb/offline-run-*` 上传。完全关闭可用 `--wandb-mode disabled`。


7 张 RTX 3090 的真实 NCCL/DDP 随机小模型测试已通过：57 条训练样本、17 条验证样本，每卡 batch 2、梯度累积 2，验证全局样本数为 17。结果保存在 `checkpoints/ddp-smoke-7gpu/`，日志为 `checkpoints/ddp-smoke-7gpu.log`。该结果仅验证多卡链路，不代表正式模型质量或正式 batch 的显存可行性。


7 卡流程复查命令（使用新的输出目录）：

```bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate 5434_project
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run --standalone --nproc_per_node=7 -m model.train \
  --device cuda --data-dir checkpoints/ddp-smoke-fixture --output checkpoints/ddp-recheck \
  --epochs 1 --batch-size 2 --eval-batch-size 2 --accumulation-steps 2 \
  --workers 0 --log-every 1 --gradient-checkpointing --wandb-mode offline
```
