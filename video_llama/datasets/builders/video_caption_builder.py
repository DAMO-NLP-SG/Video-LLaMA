import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.webvid_datasets import WebvidDataset

@registry.register_builder("webvid")
class WebvidBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebvidDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/webvid/defaults.yaml"}
    
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
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets