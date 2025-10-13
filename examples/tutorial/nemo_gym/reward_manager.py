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
import asyncio
import logging
import os
from collections import defaultdict
from typing import Any

import aiohttp
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# Just a dummy function to make the register work
def compute_score():
    raise NotImplementedError


@register("google_search")
class GoogleSearchRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str,
        resource_server_url: str,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.resource_server_url = resource_server_url

    async def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]

            # extract last assistant turn message
            last_assistant_turn_end = response_mask.nonzero()[-1].item()
            nnz = (1 - response_mask[: last_assistant_turn_end + 1]).nonzero()
            last_tool_turn_end = nnz[-1].item() if nnz.numel() > 0 else -1
            response_str = self.tokenizer.decode(
                response_ids[last_tool_turn_end + 1 : last_assistant_turn_end + 1], skip_special_tokens=True
            )

            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            try:
                score = await self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            except Exception as e:
                logger.error(f"Error in computing score: {e}")
                score = 0.0

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            print(f"response_str: {response_str}, ground_truth: {ground_truth}, reward: {reward}")
            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    async def compute_score(self, data_source, solution_str, ground_truth, extra_info=None):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.resource_server_url}/verify",
                json={  # GoogleSearchVerifyRequest
                    "responses_create_params": {
                        "input": [
                            {
                                "role": "user",
                                "content": "",
                            }
                        ],
                    },
                    "response": {
                        "id": "",
                        "created_at": 0,
                        "model": "",
                        "object": "response",
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                        "output": [
                            {
                                "role": "assistant",
                                "content": solution_str,
                            }
                        ],
                    },
                    "expected_answer": ground_truth,
                    "task_difficulty_qwen3_32b_avg_8": 0,
                },
            ) as resp:
                resp_json = await resp.json()
                return resp_json["reward"]


async def main(resource_server_url: str):
    tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/wuxibin_hldy/model/Qwen3-4B")
    rmg = GoogleSearchRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=None,
        reward_fn_key="data_source",
        resource_server_url=resource_server_url,
    )

    data_source = "google_search"
    solution_str = "The answer is \\boxed{D}\n."
    ground_truth = "D"
    extra_info = {"task_difficulty_qwen3_32b_avg_8": 3.5}

    reward = await rmg.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    print(reward)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_server_url", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.resource_server_url))
