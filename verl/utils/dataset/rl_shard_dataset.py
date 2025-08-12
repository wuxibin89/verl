# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
import multiprocessing as mp
import os
import re
import threading
import time

import datasets
import ray
import torch
from omegaconf import DictConfig, ListConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@ray.remote
class ShardDataLoader:
    """Shard data loader in each node, load dataset in range [start, end)
    and put image/video data to ray object store.

    Args:
        data_files (list[str]): List of data files.
        start (int): Start index of the dataset.
        end (int): End index of the dataset.
        image_key (str): Key for image data in the dataset.
        video_key (str): Key for video data in the dataset.
    """

    def __init__(
        self,
        data_files: list[str],
        start: int,
        end: int,
        image_key: str,
        video_key: str,
    ):
        self.data_files = data_files
        self.start = start
        self.end = end
        self.image_key = image_key
        self.video_key = video_key

        self.image_and_video: dict[int, tuple] = {}

    def load_shard_data(self):
        """Load shard dataset in range [start, end)"""
        from verl.utils.dataset.vision_utils import process_image, process_video

        t_start = time.time()
        print(f"Load shard data from {self.data_files} in range [{self.start}, {self.end})")
        dataset = datasets.load_dataset("parquet", data_files=self.data_files, split="train", streaming=True)
        shard = dataset.skip(self.start).take(self.end - self.start)

        for i, data in enumerate(shard):
            image_refs, video_refs = [], []
            # load images
            if self.image_key in data and data[self.image_key] is not None:
                images = [process_image(image) for image in data[self.image_key]]
                image_refs = [ray.put(image) for image in images]

            # load videos
            if self.video_key in data and data[self.video_key] is not None:
                videos = [process_video(video) for video in data[self.video_key]]
                video_refs = [ray.put(video.numpy()) for video in videos]

            self.image_and_video[i + self.start] = (image_refs, video_refs)

        print(f"Load shard [{self.start}, {self.end}) done, cost {time.time() - t_start}")

    def get_image_and_video(self, index: int) -> tuple[list[ray.ObjectRef], list[ray.ObjectRef]]:
        """Get image and video object_ids by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple[list[ray.ObjectRef], list[ray.ObjectRef]]: A tuple containing lists of image and video object_ids.
        """
        assert self.start <= index < self.end, f"index {index} not in shard range [{self.start}, {self.end})"
        return self.image_and_video[index]


class ShardRLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from parquet files, the image and video are sharded load in each node.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        num_shards (int, optional): Number of shards to split the dataset, testing only.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: ProcessorMixin = None,
        num_shards: int = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.num_shards = num_shards

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.use_shm = config.get("use_shm", False)

        self._download()
        self._read_files()

        # enable shard dataloader if processor is not None
        if self.processor is not None:
            self._init_shard_dataloader()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files(self):
        # don't load image and video columns
        dataset = datasets.load_dataset("parquet", data_files=self.data_files[0], split="train", streaming=True)
        columns = set(dataset.features.keys())
        columns.discard(self.image_key)
        columns.discard(self.video_key)

        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file, split="train", columns=list(columns))
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

    def _init_shard_dataloader(self):
        # start shard dataloader in each node
        nodes = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"]["CPU"] > 0]
        if self.num_shards is not None:
            nodes = [nodes[i % len(nodes)] for i in range(self.num_shards)]

        self.shard_offsets = self._calculate_shard_offsets(len(self.dataframe), len(nodes))
        self.loaders: list[ShardDataLoader] = []
        for node_rank, node_id in enumerate(nodes):
            start, end = self.shard_offsets[node_rank]
            self.loaders.append(
                ShardDataLoader.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(
                    self.data_files, start, end, self.image_key, self.video_key
                )
            )
        ray.get([loader.load_shard_data.remote() for loader in self.loaders])

        # NOTE: Dataloader subprocess worker can not remote call ShardDataLoader directly,
        # so we use multiprocessing queue to pass the index back to main process, and use
        # threads to delegate remote call.
        num_workers = self.config.dataloader_num_workers
        self.in_queues = [mp.Queue() for _ in range(num_workers)]
        self.out_queues = [mp.Queue() for _ in range(num_workers)]
        self.threads = [
            threading.Thread(target=self._worker_thread, args=(worker_rank,), daemon=True)
            for worker_rank in range(num_workers)
        ]
        for thread in self.threads:
            thread.start()

        # cache object_id to object_ref, since object_ref can't be pickled and sent to subprocess
        self.object_id_to_refs: dict[str, ray.ObjectRef] = {}

    def _calculate_shard_offsets(self, m, n):
        shard_offsets = []
        shard_size = m // n
        remainder = m % n
        start = 0
        for i in range(n):
            end = start + shard_size + (1 if i < remainder else 0)
            shard_offsets.append((start, end))
            start = end
        return shard_offsets

    def _find_shard_index(self, index: int):
        for i, (start, end) in enumerate(self.shard_offsets):
            if start <= index < end:
                return i
        raise ValueError(f"index {index} not in any shard range")

    def _worker_thread(self, worker_rank: int):
        while True:
            index = self.in_queues[worker_rank].get()
            shard_index = self._find_shard_index(index)
            image_and_video_refs = ray.get(self.loaders[shard_index].get_image_and_video.remote(index))

            # convert object_ref to object_id to send through multiprocessing queue
            for object_refs in image_and_video_refs:
                for i, object_ref in enumerate(object_refs):
                    self.object_id_to_refs[object_ref.hex()] = object_ref
                    object_refs[i] = object_ref.hex()

            self.out_queues[worker_rank].put(image_and_video_refs)

    def _convert_object_id_to_ref(self, batch: dict[str, list]) -> dict[str, list]:
        """Convert image/video object_id to object_ref"""
        multi_modal_data = batch.get("multi_modal_data", None)
        if multi_modal_data is None:
            return batch

        for item in multi_modal_data:
            if "image" in item:
                item["image"] = [self.object_id_to_refs.pop(object_id) for object_id in item["image"]]
            if "video" in item:
                item["video"] = [self.object_id_to_refs.pop(object_id) for object_id in item["video"]]

        return batch

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        row_dict["raw_prompt"] = messages

        # multi-modal data
        if self.processor is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id
            self.in_queues[worker_id].put(item)
            image_refs, video_refs = self.out_queues[worker_id].get()
            multi_modal_data = {}
            if image_refs:
                multi_modal_data["image"] = image_refs
            if video_refs:
                multi_modal_data["video"] = video_refs
            row_dict["multi_modal_data"] = multi_modal_data

        return row_dict


def process_batch(dataloader: StatefulDataLoader):
    is_shard = isinstance(dataloader.dataset, ShardRLHFDataset)
    for batch in dataloader:
        if is_shard:
            batch = dataloader.dataset._convert_object_id_to_ref(batch)
        yield batch
