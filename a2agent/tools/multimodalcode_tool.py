# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
import requests
import logging
import os
import json
import threading
from contextlib import ExitStack
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import cv2
import numpy as np
from math import ceil
from typing import Union
import ray
import ray.actor
from qwen_vl_utils import fetch_image, fetch_video

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

T = TypeVar("T")

class MultiModalCode_tool:
    """A tool for zooming in on an image by cropping it based on a bounding box.

    This tool provides a zoom-in functionality by cropping a region from an image,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the zoom-in operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "multimodalcode_tool",
                "description": (
                    "Perform deep research for image and video via code excution"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items":{"type":"number"},
                            "minItems":4,
                            "maxItems":4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional).",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # Worker and rate limiting configuration
        self.timeout = config.get("timeout", 60)
        self.name = "multimodalcode"
        self.code_sandbox_url = config.get("code_sandbox_url", "http://0.0.0.0:16259")
        self.headers = {"Content-Type": "application/json"}
        logger.info("[MultiModalCode_tool] 实例化MultiModalCode_tool")
        logger.info(f"Initialized multimodalcode_tool with config: {config}")


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for multimodal code server.
        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        logger.info(f"[MultiModalCode_Tool] 进入工具create, kwargs={kwargs}")

        try:
            code = kwargs.get("code", "")
            if len(code)==0:
                logger.error(f"[DEBUG] Code为空!")
            payload = {
                "code": code,
                "timeout": self.timeout + 2
            }
            response = requests.post(self.code_sandbox_url, headers=self.code_sandbox_url, data=json.dumps(payload))
            # {
            # "task_id":"5e44a605-f85d-427e-9db8-90d2c72f96ba",
            # "status":"pending",
            # "message":"任务已提交，正在处理",
            # "created_at":1767551607.5667443,
            # "check_status_url":"/task/5e44a605-f85d-427e-9db8-90d2c72f96ba/status",
            # "get_result_url":"/task/5e44a605-f85d-427e-9db8-90d2c72f96ba/result"
            # }
            instance_id = self.code_sandbox_url + response["get_result_url"]
            logger.info(f"[DEBUG] 工具发送Code成功, instance_id = {instance_id}, response = {response}, payload={payload}")
        except Exception as e:
            logger.error(f"[DEBUG] 工具发送Code错误:{e}")


        return instance_id, ToolResponse()

    # def resize_min_image_opencv(self, image: np.ndarray) -> np.ndarray:
    #     """
    #     Qwen-VL raises an error for images with height or width less than 32 pixels.
    #     使用OpenCV格式处理图像。
        
    #     Args:
    #         image: OpenCV格式的图像，类型为numpy.ndarray，形状为(H, W, C)或(H, W)
    #             其中H为高度，W为宽度，C为通道数(1, 3或4)
    #             数据类型应为uint8或float32/float64
            
    #     Returns:
    #         调整大小后的图像，类型与输入相同，为numpy.ndarray
    #     """
    #     # 类型检查
    #     if not isinstance(image, np.ndarray):
    #         raise TypeError(f"Input image must be a numpy.ndarray, got {type(image)}")
        
    #     # 维度检查
    #     if image.ndim not in [2, 3]:
    #         raise ValueError(f"Image must have 2 or 3 dimensions, got {image.ndim}")
        
    #     # 获取图像尺寸 (OpenCV中是先高后宽)
    #     height, width = image.shape[:2]
        
    #     # 处理长宽比过大的情况
    #     if max(height, width) / min(height, width) > 200:
    #         max_val = max(height, width)
    #         min_val = min(height, width)

    #         old_scale = max_val / min_val
    #         max_ratio = min(150, old_scale / 2)
    #         target_max = int(min_val * max_ratio)

    #         if height > width:
    #             new_height = target_max
    #             new_width = int(width * old_scale / max_ratio)
    #         else:
    #             new_width = target_max
    #             new_height = int(height * old_scale / max_ratio)
            
    #         # 确保尺寸为正数
    #         new_width = max(1, new_width)
    #         new_height = max(1, new_height)
            
    #         # 使用OpenCV的resize函数
    #         image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    #         height, width = image.shape[:2]

    #     # 处理最小边小于32像素的情况
    #     if min(height, width) >= 32:
    #         return image

    #     ratio = 32 / min(height, width)
    #     new_height = ceil(height * ratio)
    #     new_width = ceil(width * ratio)
        
    #     # 确保尺寸为正数
    #     new_width = max(1, new_width)
    #     new_height = max(1, new_height)
        
    #     # 使用OpenCV的resize函数
    #     new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    #     return new_image

    async def execute(self, instance_id: str, **kwargs) -> tuple[ToolResponse, float, dict]:
        '''
        Docstring for execute
            instance_id: url for code result

        :rtype: tuple[ToolResponse, float, dict]
        '''
        try:
            resjson = requests.get(instance_id)
            # {"success":true,
            # "stdout":"视频文件大小: 1348854 bytes\nFPS: 24.0, 总帧数: 657\n第20秒帧位置: 480\n帧尺寸: 480x270\n保存裁剪帧: /project/peilab/qjl/CODE/SERVER/tmp/tmp3xi_ezcy/frame_20s.jpg\n文件存在: True\n",
            # "stderr":"",
            # "returncode":0,
            # "execution_time":1.5079731941223145,
            # "tmp_path": absolute_dir}
            stdout = resjson['stdout']
            stderr = resjson['stderr']
            tool_response_text = f"[Sand_Box_Server]: {{success:{resjson['success']} stdout:{stdout}, stderr:{stderr}}}"
            logger.info(f"[DEBUG] tool_response_text={tool_response_text}")
            tmp_path = resjson['tmp_path']
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
            # 获取所有文件路径
            image_list = []
            video_list = []
            for root, dirs, files in os.walk(tmp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_list.append(file_path)
                    elif ext in video_extensions:
                        video_list.append(file_path)
            return ToolResponse(text=tool_response_text, image=image_list, video=video_list),   0.0 if resjson['success'] else -0.05,  {"success": resjson['success']}
# return ToolResponse(text=tool_response_text, image=image_list if image_list else None, video=video_list if video_list else None), 0.0 if resjson['success'] else -0.05, {"success": resjson['success']}

        except Exception as err:
            tool_response_text = f' [ERROR code] Request to Sand_Box_Server failed: {err}'
            logger.error(f"[DEBUG] tool_response_text={tool_response_text}")
            return ToolResponse(text=tool_response_text, image=image_list, video=video_list),   0.0 if resjson['success'] else -0.05,    {"success": resjson['success']}

        
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]