from dataset.task_fns import GroundingTaskFn, PointingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from dataset.task_fns.task_prompts.pointing_task import POINTING_POINT
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28


grounding_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_data.images.tsv",
    anno_tsv_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_data.annotations.tsv",
    anno_idx_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_data.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_grounding_data",
    reward_name="box_iou",
)


pointing_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_point_data.images.tsv",
    anno_tsv_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_point_data.annotations.tsv",
    anno_idx_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_point_data.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=PointingTaskFn,
        task_prompts=POINTING_POINT,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_pointing_data",
    reward_name="point_in_box",
)

train_dataset = [
    grounding_data,
]
