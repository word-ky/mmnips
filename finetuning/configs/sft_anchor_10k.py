from dataset import AnchorTSVDataset, ConcatDataset, DataCollatorForSupervisedDataset
from dataset.task_fns import AnchorTaskFn
from dataset.task_fns.task_prompts.anchor_task import ANCHOR_PROMPTS

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "/root/autodl-fs/mllm检测/Rex-Omni/hf_ckpt/Rex-Omni"
unfreeze_last_n_layers = 8

anchor_data = dict(
    type=AnchorTSVDataset,
    img_tsv_file="/root/autodl-fs/mllm检测/Rex-Omni/finetuning/dataset/anchor_tsv_10k/anchor_train_10k.images.tsv",
    ann_tsv_file="/root/autodl-fs/mllm检测/Rex-Omni/finetuning/dataset/anchor_tsv_10k/anchor_train_10k.annotations.tsv",
    ann_lineidx_file="/root/autodl-fs/mllm检测/Rex-Omni/finetuning/dataset/anchor_tsv_10k/anchor_train_10k.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=AnchorTaskFn,
        task_prompts=ANCHOR_PROMPTS,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="anchor_train_10k",
)

train_dataset = dict(type=ConcatDataset, datasets=[anchor_data])
data_collator = dict(type=DataCollatorForSupervisedDataset)
