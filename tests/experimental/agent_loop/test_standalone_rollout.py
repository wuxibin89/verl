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
import os

import pytest
import ray
from omegaconf import DictConfig
from openai import AsyncOpenAI

from verl.experimental.agent_loop import AsyncLLMServerManager, SingleTurnAgentLoop
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.rollout_server import get_rollout_server_class


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = 2
    config.trainer.nnodes = 2
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-1.5B-Instruct"
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.load_format = "auto"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    config.actor_rollout_ref.rollout.num_standalone_rollouts = 2

    return config


@pytest.mark.asyncio
async def test_standalone_rollout_token(init_config):
    """Test standalone rollout with token-in-token-out."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    init_config.actor_rollout_ref.rollout.mode = "async"

    # create standalone rollout server
    rollout_config: RolloutConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.rollout)
    model_config: HFModelConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.model)
    rollout_server_class = get_rollout_server_class(rollout_config.name)
    rollout_servers = [
        rollout_server_class(dp_rank=dp_rank, config=rollout_config, model_config=model_config)
        for dp_rank in range(rollout_config.num_standalone_rollouts)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == rollout_config.num_standalone_rollouts
    assert len(server_addresses) == rollout_config.num_standalone_rollouts

    # agent loop call standalone rollout server
    server_manager = AsyncLLMServerManager(init_config, server_handles=server_handles)
    agent_loop = SingleTurnAgentLoop(
        trainer_config=init_config,
        server_manager=server_manager,
        tokenizer=model_config.tokenizer,
        processor=model_config.processor,
    )

    kwargs = {
        "raw_prompt": [{"role": "user", "content": "How are you?"}],
        "data_source": "openai/gsm8k",
        "reward_model": {"style": "rule", "ground_truth": "1.0"},
    }
    output = await agent_loop.run(sampling_params={}, **kwargs)
    prompt = model_config.tokenizer.decode(output.prompt_ids)
    response = model_config.tokenizer.decode(output.response_ids)

    print("================ prompt ==================")
    print(f"{prompt}")
    print("================ response ==================")
    print(f"{response}")

    ray.shutdown()


@pytest.mark.asyncio
async def test_standalone_rollout_message(init_config):
    """Test standalone rollout with openai chat completion api."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # FIXME(@wuxibin): we should add a new option to switch between
    # token-in-token-out and openai chat completion api.
    init_config.actor_rollout_ref.rollout.mode = "sync"

    # create standalone rollout server
    rollout_config: RolloutConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.rollout)
    model_config: HFModelConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.model)
    rollout_server_class = get_rollout_server_class(rollout_config.name)
    rollout_servers = [
        rollout_server_class(dp_rank=dp_rank, config=rollout_config, model_config=model_config)
        for dp_rank in range(rollout_config.num_standalone_rollouts)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == rollout_config.num_standalone_rollouts
    assert len(server_addresses) == rollout_config.num_standalone_rollouts

    client = AsyncOpenAI(
        api_key="",
        base_url=f"http://{server_addresses[0]}/v1",
    )

    completion = await client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": "你好啊"}],
    )
    print(completion.choices[0].message.content)

    ray.shutdown()
