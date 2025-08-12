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
import os

import numpy as np
import ray
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor

from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.dataset.rl_shard_dataset import ShardRLHFDataset, process_batch


def test_shard_dataset_image():
    ray.init()

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose("ppo_trainer")

    model_path = "/mnt/hdfs/wuxibin_wl/model/Qwen2.5-VL-3B-Instruct"
    local_folder = os.path.expanduser("~/data/geo3k")
    data_files = [os.path.join(local_folder, "train.parquet")]
    hf_dataset = load_dataset("parquet", data_files=data_files, split="train")

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    dataset = ShardRLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=processor,
        num_shards=4,
    )

    batch_size = 128
    sampler = create_rl_sampler(config.data, dataset)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    for batch in process_batch(dataloader):
        assert len(batch["raw_prompt"]) == batch_size

        images = ray.get(batch["multi_modal_data"][0]["image"])
        extra_info = batch["extra_info"][0]
        truth_images = hf_dataset[extra_info["index"]]["images"]
        for img1, img2 in zip(images, truth_images, strict=False):
            np.array_equal(np.array(img1), np.array(img2))

    print("Test passed!")
    ray.shutdown()


def test_shard_dataset_text():
    ray.init()

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose("ppo_trainer")

    model_path = "/mnt/hdfs/wuxibin_wl/model/Qwen2.5-VL-3B-Instruct"
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    data_files = [os.path.join(local_folder, "train.parquet")]

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    dataset = ShardRLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
        num_shards=4,
    )

    batch_size = 128
    sampler = create_rl_sampler(config.data, dataset)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    for batch in process_batch(dataloader):
        assert len(batch["raw_prompt"]) == batch_size

    print("Test passed!")
    ray.shutdown()
