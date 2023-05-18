import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.laion_dataset import LaionDataset
from video_llama.datasets.datasets.llava_instruct_dataset import Instruct_Dataset
from video_llama.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset

@registry.register_builder("instruct")
class Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/instruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token 
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name 
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'


        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token = num_video_query_token,
            tokenizer_name = tokenizer_name,
            data_type = self.config.data_type
        )

        return datasets

@registry.register_builder("webvid_instruct")
class WebvidInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }

@registry.register_builder("webvid_instruct_zh")
class WebvidInstruct_zh_Builder(Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }



@registry.register_builder("llava_instruct")
class LlavaInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/llava_instruct.yaml",
    }

