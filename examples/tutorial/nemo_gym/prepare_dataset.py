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
import argparse

import pandas as pd


def map_fn(row):
    item = {
        "data_source": "google_search",
        "prompt": [
            {
                "role": "user",
                "content": row["responses_create_params"]["instructions"] + row["responses_create_params"]["input"],
            },
        ],
        "reward_model": {
            "ground_truth": row["expected_answer"],
        },
        "task_difficulty_qwen3_32b_avg_8": row["task_difficulty_qwen3_32b_avg_8"],
    }
    return item


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to Nemo-Gym dataset in jsonl format")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save processed dataset in parquet format"
    )
    args = parser.parse_args()

    dataset = pd.read_json(args.input_path, lines=True)
    rl_dataset = dataset.apply(map_fn, axis=1, result_type="expand")
    rl_dataset.to_parquet(args.output_path, index=False)
