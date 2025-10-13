# NVIDIA-NeMo/Gym: Google Search Agent Training

This is an tutorial for training a Google Search Agent using [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym).

## Setup NeMo-Gym Resource Server
Please follow the instructions in [resources_servers/google_search](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/google_search) to setup the resource server.

```bash
config_paths="resources_servers/google_search/configs/google_search.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

Get the resource server url from stdout:
```txt
[1] google_search (resources_servers/google_search)
{
    'process_name': 'google_search',
    'server_type': 'resources_servers',
    'name': 'google_search',
    'dir_path': '/opt/tiger/Gym/resources_servers/google_search',
    'entrypoint': 'app.py',
    'host': '127.0.0.1',
    'port': 42613,
    'pid': 77241,
    'config_path': 'google_search',
    'url': 'http://127.0.0.1:42613',    <- resource server url
}
```

To verify the resource server is working, you can run the following command:
```bash
python examples/tutorial/nemo_gym/google_search_tool.py --resource_server_url <url>
```

## Train the Google Search Agent

1. Convert Nemo-Gym dataset to verl dataset format
```bash
python examples/tutorial/nemo_gym/prepare_dataset.py \
    --input_path resources_servers/google_search/data/example.jsonl \
    --output_path example.parquet
```

2. Start training
```bash
RESOURCE_SERVER_URL=<url> bash examples/tutorial/nemo_gym/run_qwen3_4b.sh
```
