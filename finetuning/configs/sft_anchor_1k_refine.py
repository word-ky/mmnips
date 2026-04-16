from dataset import AnchorTSVDataset, ConcatDataset, DataCollatorForSupervisedDataset
from dataset.task_fns import AnchorTaskFn
from dataset.task_fns.task_prompts.anchor_task import ANCHOR_PROMPTS

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# 从你已经训好的 e2 checkpoint 继续
model_name_or_path = "/autodl-fs/data/mllm检测/Rex-Omni/finetuning/work_dirs/sft_anchor_lmhead_align120"

anchor_data = dict(
    type=AnchorTSVDataset,
    img_tsv_file="/root/autodl-fs/mllm检测/Rex-Omni/finetuning/dataset/anchor_tsv_1k/anchor_train_1k.images.tsv",
    ann_tsv_file="/root/autodl-fs/mllm检测/Rex-Omni/finetuning/dataset/anchor_tsv_1k/anchor_train_1k.annotations.tsv",
    ann_lineidx_file="/root/autodl-fs/mllm检测/Rex-Omni/finetuning/dataset/anchor_tsv_1k/anchor_train_1k.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=AnchorTaskFn,
        task_prompts=ANCHOR_PROMPTS,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="anchor_train_1k_refine",
)

train_dataset = dict(type=ConcatDataset, datasets=[anchor_data])
data_collator = dict(type=DataCollatorForSupervisedDataset)
