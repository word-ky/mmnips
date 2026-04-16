# Rex-Omni Anchor Task — 项目交接文档

> 最后更新: 2026-04-16

## 一、项目目标

在 Rex-Omni（基于 Qwen2.5-VL 的多任务视觉模型）上新增 **Semantic Anchor 检测任务**。

原始 Rex-Omni 支持 detection（输出 bounding box `<x1><y1><x2><y2>`）、pointing 等任务。Anchor 任务要求模型输出 **5 元组** `<coord_id><x_grid><y_grid><scale_id><ratio_id>`，表示物体在预定义网格上的语义锚点，用于替代传统 bbox 的目标表示。

---

## 二、相较于原始 Rex-Omni 的代码变更

### 2.1 修改的文件（10 个）

| 文件 | 变更内容 |
|---|---|
| `finetuning/train.py` | 新增 `unfreeze_last_n_layers` 支持（冻结大部分 LLM 层，只解冻最后 N 层 + lm_head 进行格式对齐微调） |
| `finetuning/engine/argument.py` | 新增 `unfreeze_last_n_layers` 参数（默认 4） |
| `finetuning/dataset/__init__.py` | 注册 `AnchorTSVDataset` |
| `finetuning/dataset/task_fns/__init__.py` | 注册 `AnchorTaskFn` |
| `rex_omni/tasks.py` | 新增 `TaskType.ANCHOR` 及其 prompt 模板 |
| `rex_omni/parser.py` | 新增 `parse_anchor_prediction()` 函数，解析 anchor 输出 |
| `rex_omni/wrapper.py` | 在推理管线中接入 anchor 任务（prompt 构建 + 结果解析） |
| `app.py` | Gradio demo 中增加 anchor 任务入口 |
| `finetuning/tools/convert_json_data_to_tsv.py` | 小修改适配 anchor 数据 |
| `.gitignore` | 排除权重/checkpoint/数据目录 |

### 2.2 新增的文件（11 个）

| 文件 | 用途 |
|---|---|
| `finetuning/dataset/anchor_tsv_dataset.py` | Anchor 数据集加载器，继承自 `GroundingTSVDataset` |
| `finetuning/dataset/task_fns/anchor_task.py` | 训练时的数据处理：将 anchor 标注转换为模型训练目标格式 |
| `finetuning/dataset/task_fns/task_prompts/anchor_task.py` | Anchor 任务的 prompt 模板（4 种变体） |
| `finetuning/tools/build_anchor_labels.py` | 从 COCO bbox 标注生成 anchor 标签的工具脚本 |
| `finetuning/configs/sft_anchor_1k.py` | 训练配置（基础版） |
| `finetuning/configs/sft_anchor_1k_refine.py` | 训练配置（续训版） |
| `finetuning/configs/sft_anchor_boxtoken.py` | **当前使用的训练配置**（box token 策略） |
| `finetuning/configs/anchor_label.yaml` | Anchor 标签生成配置 |
| `finetuning/configs/anchor_label.local.yaml` | Anchor 标签生成配置（本地路径版） |
| `finetuning/dataset/anchor_train_1k.stats.json` | 1K 训练数据的统计信息 |
| `tutorials/anchor_example.py` | Anchor 推理测试脚本 |

---

## 三、技术方案演进

### 3.1 方案 v1 — 新特殊 token `<|anchor_start|>` / `<|anchor_end|>`（失败）

- 添加新 token，从 `<|box_start|>` 复制 embedding 初始化
- 冻结除最后 4 层以外的 LLM 层
- **问题**：模型输出混用 box 和 anchor 格式，只输出 4 值而非 5 值

### 3.2 方案 v2 — 短名 token `<|anc_s|>` / `<|anc_e|>` + 更好的初始化（失败）

- 换用更短的 token 名，从语义更接近的 `<|quad_start|>` 初始化
- 解冻 8 层，30 epochs，constant_with_warmup LR
- **问题**：新 token embedding 学不动，输出 degenerate

### 3.3 方案 v3 — 解绑 lm_head + 单独训练（失败）

- 发现 `tie_word_embeddings=True` 导致 lm_head 和 embed_tokens 共享权重（300M+ 参数），1K 数据根本推不动
- 手动解绑 lm_head，冻结 embed_tokens，只训练 lm_head
- **问题**：输出稍有改善但仍然 degenerate，且破坏了原有 box 检测能力

### 3.4 方案 v4（当前）— 复用 `<|box_start|>` / `<|box_end|>`（成功）

**核心思路**：放弃新 token，直接复用模型已学好的 `<|box_start|>` / `<|box_end|>` 来包裹 anchor 输出。模型通过 **prompt 指令** 区分两种任务：
- Detection prompt → 输出 4 值 `<x1><y1><x2><y2>`
- Anchor prompt → 输出 5 值 `<coord_id><x><y><scale_id><ratio_id>`

**优势**：
- 不改变 vocabulary，无需 `resize_token_embeddings`
- 不改变 `lm_head` / `embed_tokens` 的权重结构
- 模型已经熟练使用这对 token，学习成本极低
- `train.py` 极简，只需标准的 partial-layer SFT

---

## 四、当前效果

### 4.1 训练

- 数据：1K 样本，30 epochs（实际跑到 ~26 epoch 后因 checkpoint 保存时中断）
- Loss 曲线：4.87 → 0.002（收敛良好）
- 最新可用 checkpoint：`checkpoint-1612`（~epoch 26）

### 4.2 推理结果

**格式对齐 ✅**：模型学会了在 anchor prompt 下输出 5 值格式
```
<|object_ref_start|>man<|object_ref_end|><|box_start|><0><33><41><5><1>,...<|box_end|>
```

**存在的问题**：
| 问题 | 原因 |
|---|---|
| `repetition_penalty=1.05` 时无限重复 | 1K 数据过拟合 |
| 部分 anchor 值中混入中文/乱码 | 过拟合 + 数据不足导致 hallucination |
| 原有 box 检测能力部分退化 | 共用 box token，anchor 微调改变了 box 输出分布 |
| `cup` 类别输出过多 anchor | 训练数据中该类别分布不均 |

---

## 五、待解决问题与后续建议

### 5.1 立即可做

| 措施 | 预期效果 |
|---|---|
| **减少训练 epochs 至 10** | 避免过拟合，loss 在 0.05~0.1 时泛化更好 |
| **增加训练数据至 5K+** | 更多样本 = 更好的泛化，减少 hallucination |

### 5.2 中期建议

| 措施 | 预期效果 |
|---|---|
| **混合训练**：anchor 数据 + 原始 box 检测数据 | 防止 catastrophic forgetting，保留 box 检测能力 |
| **数据增强**：多样化的类别组合和图片场景 | 提升模型在 unseen 场景的泛化能力 |
| **调整 repetition_penalty** 或使用 length penalty | 控制输出长度，避免重复 |

### 5.3 长期方向

- 评估 anchor 表示与 bbox 表示在下游任务中的精度对比
- 探索 anchor + bbox 联合输出的可能性

---

## 六、关键文件速查

### 训练
```bash
# 训练命令
cd finetuning && deepspeed --num_gpus=1 train.py \
    --config configs/sft_anchor_boxtoken.py \
    --deepspeed scripts/zero2.json \
    --bf16 True \
    --output_dir work_dirs/sft_anchor_boxtoken \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 5 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --data_flatten False
```

### 推理测试
```bash
python tutorials/anchor_example.py \
    --model_path finetuning/work_dirs/sft_anchor_boxtoken/checkpoint-1612 \
    --image_path tutorials/detection_example/test_images/cafe.jpg
```

注意：checkpoint 目录需要包含 `preprocessor_config.json`（从原始模型 `hf_ckpt/Rex-Omni/` 复制）。

### 数据生成
```bash
# 从 COCO 标注生成 anchor 标签
python finetuning/tools/build_anchor_labels.py --config finetuning/configs/anchor_label.yaml

# 转为 TSV 格式
python finetuning/tools/convert_json_data_to_tsv.py \
    --jsonl_file dataset/anchor_train_1k.jsonl \
    --output_dir dataset/anchor_tsv_1k
```

### 代码核心链路
```
Prompt 构建:  tasks.py (TaskType.ANCHOR) → wrapper.py (_build_messages)
训练目标:     anchor_task.py (_compose_answer_from_anchors) → <|box_start|>5值<|box_end|>
推理解析:     wrapper.py → parser.py (parse_anchor_prediction) → 按数值个数区分 5=anchor / 4=box
```

---

## 七、环境信息

- 服务器：AutoDL vGPU-32GB
- Python 3.12 + PyTorch + Transformers + DeepSpeed
- 基础模型：Rex-Omni (Qwen2.5-VL-3B-Instruct 架构, 151936 vocab)
- 原始模型路径：`hf_ckpt/Rex-Omni/`
- 训练 checkpoint：`finetuning/work_dirs/sft_anchor_boxtoken/checkpoint-1612`
