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
import asyncio
import json

import aiohttp
from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse


class GoogleSearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.resource_server_url = config["resource_server_url"]

    async def search(self, query: str) -> str:
        """Search Google for a query and return up to 10 search results. Use browse() to retrieve full content
        from relevant URL(s), or refine your search query if results aren't relevant enough.

        Args:
            query: The term to search for.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.resource_server_url}/search",
                json={"query": query},
            ) as response:
                data = await response.json()
                return data["search_results"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.search)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = await self.search(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


class BrowseTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.resource_server_url = config["resource_server_url"]

    async def browse(self, url: str) -> str:
        """Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.

        Args:
            url: The url of the page to get the content of.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.resource_server_url}/browse",
                json={"url": url},
            ) as response:
                data = await response.json()
                return data["page_content"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.browse)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = await self.browse(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


async def main(resource_server_url: str):
    search_tool = GoogleSearchTool(config={"resource_server_url": resource_server_url}, tool_schema=None)
    search_results = await search_tool.execute(instance_id="", parameters={"query": "What is the capital of France?"})
    print(search_results)

    browse_tool = BrowseTool(config={"resource_server_url": resource_server_url}, tool_schema=None)
    browse_results = await browse_tool.execute(
        instance_id="", parameters={"url": "https://en.wikipedia.org/wiki/Paris"}
    )
    print(browse_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_server_url", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.resource_server_url))
