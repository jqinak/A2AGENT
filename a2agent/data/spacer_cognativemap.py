# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 Reallm Labs Ltd. or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
import torch
import datasets

from verl.utils.hdfs_io import copy, makedirs

import a2agent.data as d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default='/nvme/data-pool1/qjl/A2VL_DATA/data/spacer_c_map', help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/a2agent/data/spacer_c_map",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()



    local_dataset_path = args.local_dataset_path
    dataset = datasets.load_dataset(local_dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        # Assuming you want to work with a single split, e.g., "train"
        # Or you can concatenate all splits
        dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    
    print(dataset[0])  # 查看第一条数据

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    import random
    random.shuffle(indices)
    
    train_size = int(0.97 * dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:dataset_size]
    
    # Select subsets using Hugging Face Dataset's select method
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    train_dataset = dataset
    SYSTEM_PROMPT_1="""You are the model performing "Multi-Modal Chain-of-Thought with Code" reasoning paradiam in the conceptual flowchart. Reasoning according to it."""
    SYSTEM_PROMPT_2 = "<image> The diagram above presents the concept of Multi-Modal Chain-of-Thought with Code.\n"
          
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            video_path = example.pop("video_path")
            map = example.pop('cognitive_map')
            object_list = example.pop("object_list")
            USER_PROMPT = f""""object_list": {object_list}"""
            # problem = SYSTEM_PROMPT_2 + "What is your cognitive map about the objects in this video ? "+ USER_PROMPT + f"If needed, the video/image's path: {video_path} " + "<video>"
            problem = "<video> What is your cognitive map about objects in this video ? "+ USER_PROMPT + f"If needed, the video/image's path: {video_path} " + "<video>"
            THINK = f"<think>\nUmm, the user asks me to response the cognitive map of the video. The object_list is provided so I only need to focus on the listed objected when they appear in the video. I plan to estimate first and write code to visualize and verify the cognitive map. \n</think>\n"
            ESTIMATE = f"<estimate>\n{map}\n</estimate>\n"
            CODE = f"""<code>\n```python\nimport matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Given cognitive map
cognitive_map = {map}
object_list = [
    "backpack", "bed", "bicycle", "ceiling", "clock", "coffee table",
    "couch", "desk", "dish rack", "doorframe", "guitar", "guitar case",
    "kitchen counter", "laundry basket", "microwave", "mirror", "nightstand",
    "refrigerator", "scale", "shelf", "shower", "sink", "table", "tissue box",
    "toaster", "toaster oven", "toilet", "trash can", "tv", "window"
]

# Create a colormap for objects
colors = plt.cm.tab20(np.linspace(0, 1, len(object_list)))
color_map = dict(zip(object_list, colors))

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 10)
ax.set_aspect('equal')
ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

# Draw grid
for i in range(11):
    ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
    ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

# Plot each object
for obj in object_list:
    if obj in cognitive_map:
        for x, y in cognitive_map[obj]:
            # Draw a square at (x, y)
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color_map[obj], edgecolor='black')
            ax.add_patch(rect)

# Add legend
handles = []
labels = []
for obj in object_list:
    if obj in cognitive_map:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=color_map[obj]))
        labels.append(obj)

# Only show unique objects
legend = ax.legend(handles, labels, title="Objects", bbox_to_anchor=(1.05, 1), loc='upper left')
legend.get_title().set_fontsize(12)

# Title
ax.set_title("Prediction", fontsize=24, fontweight='bold')

# Remove axes ticks
ax.set_xticks([])
ax.set_yticks([])

# Save or show
plt.tight_layout()
plt.savefig("./cognitive_map.png", dpi=300, bbox_inches='tight')\n```\n</code>"""
            response1 = THINK + ESTIMATE + CODE
            video_path =os.path.basename(video_path)
            data = {
                "messages": [
                    # {
                    #     "role": "system",
                    #     "content": SYSTEM_PROMPT_1,
                    # },
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT_2 + problem,
                    },
                    {
                        "role": "assistant",
                        "content": response1,
                    }
                ],
                "images": [{
                    "type": "image",
                    "image": "/home/jincai_guo/ICML2025_JIE/QJL/Evaluation/SYATEM.png"
                }],
                "videos": [{
                    "type": "video",
                    "video": f"/nvme/data-pool1/qjl/A2VL_DATA/spatialvideo/{video_path}",
                    "fps": 2,
                    "min_frames": 1,
                    "max_frames": 128
                }],
            }
            
            return data

        return process_fn
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    # train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    train_dataset.to_json(os.path.join(local_save_dir, "train.jsonl"), orient="records", lines=True)
    test_dataset.to_json(os.path.join(local_save_dir, "test.jsonl"), orient="records", lines=True)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
