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
    parser.add_argument("--local_dataset_path", default='/nvme/data-pool1/qjl/A2VL_DATA/data/segmentationimage', help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/a2agent/data/segmentationimage",
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
            image = example.pop("images")
            message =  example.pop("messages")
            assert len(message)>=2
            problem = message[0]["content"]
            answer = message[1]["content"]
            remove_text = ", as segmentation hints.\nPlease answer this question based on the visual content.Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.At the end, you must output the final answer in the format:\n<answer><your_answer_here></answer>\nThis task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\nPlease provide ONE bounding box, 3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) within the <answer>...</answer> tags.\nChoose informative points that help distinguish object vs. background. Prefer negatives on clear non-object pixels INSIDE the box when safe; otherwise place them just outside on obvious background. Negatives must NEVER be on the object or on its boundary.\nExample:\n<answer>{\"boxes\": [x1, y1, x2, y2], \"positive_points\": [[x,y],[x,y],[x,y]], \"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
            to_text = "."
            cleaned_problem = SYSTEM_PROMPT_2 + problem.replace(remove_text, to_text).strip()
            image_list = [{
                    "type": "image",
                    "image": "/home/jincai_guo/ICML2025_JIE/QJL/Evaluation/SYATEM.png"
                }]
            for img in image:
                if str(image).startswith("/nvme/data-pool1/qjl/OneThinker-train-data/"):
                    img = str(img).replace("/nvme/data-pool1/qjl/OneThinker-train-data/","")
                elif str(image).startswith("./"):
                    img = str(img).replace("./","")
                image_list.append({
                        "type": "image",
                        "image": f"/nvme/data-pool1/qjl/A2VL_DATA/{img}" 
                    })
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_1,
                    },
                    {
                        "role": "user",
                        "content": cleaned_problem,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                ],
                "images": image_list,
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
