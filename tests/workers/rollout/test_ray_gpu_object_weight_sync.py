import ipdb

import ray
import ray.experimental.collective
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def init_worker_groups(config: DictConfig):
    # create role => resource_pool mapping
    role_worker_mapping = {
        Role.Actor: ray.remote(ActorRolloutRefWorker),
        Role.Rollout: ray.remote(ActorRolloutRefWorker),
    }
    # TODO: create sperate resource_pool for each rollout instance.
    resource_pool_spec = {
        "actor": [4] * 1,
        "rollout": [4] * 1,
    }
    mapping = {
        Role.Actor: "actor",
        Role.Rollout: "rollout",
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor
    resource_pool = resource_pool_manager.get_resource_pool(Role.Actor)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.Actor], config=config.actor_rollout_ref, role="actor"
    )
    resource_pool_to_cls[resource_pool]["actor"] = actor_rollout_cls

    # create rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.Rollout)
    rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.Rollout], config=config.actor_rollout_ref, role="rollout"
    )
    resource_pool_to_cls[resource_pool]["rollout"] = rollout_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    for group_name, worker_group in all_wg.items():
        print(f"Init model for {group_name}")
        worker_group.init_model()

    return all_wg


if __name__ == "__main__":
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

    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-1.5B-Instruct"
    config.actor_rollout_ref.actor.strategy = "fsdp2"

    # 1. create actor and rollout worker group
    all_wg = init_worker_groups(config)
    actor_group = all_wg["actor"]
    rollout_group = all_wg["rollout"]

    # 2. build weight sync group
    group = ray.experimental.collective.create_collective_group(
        actor_group._workers + rollout_group._workers, backend="nccl", name="weight_sync"
    )

    # 3. sync weight from actor to rollout
    actor_state_dicts = actor_group.state_dict()
    ray.get(rollout_group.load_state_dict(actor_state_dicts))
    ipdb.set_trace()
