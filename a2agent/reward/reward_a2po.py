import re
import math

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    计算格式奖励和答案奖励的总分。
    
    参数:
        data_source: 数据源标识（未在本函数中使用，但保留参数）
        solution_str: 待评分的回答字符串
        ground_truth: 标准答案（用于内容比较）
        extra_info: 额外信息（可选，包含problem_type等字段）
        
    返回:
        total_score: 总分（格式奖励 + 答案奖励）
    """
    print(f"[DEBUG] compute_score  data_source, solution_str, ground_truth, extra_info:{data_source, solution_str, ground_truth, extra_info}")
    
    # 1. 定义格式匹配的正则表达式
    pattern = r'^.*?(?:<think>.*?</think><answer>.*?</answer><code>.*?</code>)*<think>.*?</think><answer>.*?</answer>$'
    
    # 2. 计算格式奖励（最高10分）
    format_reward = 0
    if re.match(pattern, solution_str, re.DOTALL):
        format_reward = 1  # 格式完全符合要求
        # 额外奖励：如果包含至少一个完整的think-answer-code序列（除了结尾的think-answer）
        # if re.search(r'<code>.*?</code>', solution_str):
            # format_reward += 2  # 额外奖励2分
    else:
        # 部分匹配尝试：至少以think-answer结尾
        if re.search(r'<think>.*?</think><answer>.*?</answer>$', solution_str, re.DOTALL):
            format_reward = 1  # 部分格式正确
        elif re.search(r'<answer>.*?</answer>$', solution_str, re.DOTALL):
            format_reward = 0.5
    
    # 3. 处理ground_truth_clean
    ground_truth_clean = str(ground_truth)
    # 去除<answer></answer>包裹
    if ground_truth_clean.startswith('<answer>') and ground_truth_clean.endswith('</answer>'):
        ground_truth_clean = ground_truth_clean[8:-9]  # 去除标签
    
    # 4. 计算答案奖励（最高20分）
    answer_reward = 0
    solution_clean = extract_last_answer_content(solution_str)
    
    # 检查extra_info中的problem_type
    problem_type = None
    if extra_info and isinstance(extra_info, dict):
        problem_type = extra_info.get('problem_type')
    
    if problem_type == "multichoice":
        # 选择题：检查solution是否包含ground_truth_clean
        if ground_truth_clean.lower() in solution_clean.lower():
            answer_reward = 1 # 选择题完全正确得满分
        else:
            answer_reward = 0   # 选择题错误不得分
    
    elif problem_type == "numeric":
        # 数值题：根据差距大小计算分数
        try:
            # 尝试从solution中提取数值
            solution_numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', solution_clean)
            if solution_numbers:
                solution_value = float(solution_numbers[-1])  # 取最后一个数值
                ground_truth_value = float(ground_truth_clean)
                
                # 计算相对误差
                if ground_truth_value == 0:
                    # 如果标准答案是0，使用绝对误差
                    if solution_value == 0:
                        diff = 0
                    else:
                        diff = min(1.0, abs(solution_value)) 
                else:
                    diff = min(1.0, abs(solution_value - ground_truth_value) / abs(ground_truth_value)) 
                
                #  [0,1]
                answer_reward = 1 - diff 

            else:
                answer_reward = 0  # 未找到数值
        except (ValueError, IndexError):
            answer_reward = 0  # 数值转换失败
    
    else:
        # 其他类型或未指定类型：使用默认的字符串包含检查
        if ground_truth_clean.lower() in solution_clean.lower():
            answer_reward = 1 # 基础奖励
        else:
            answer_reward = 0
    
    total_score = format_reward*0.2 + answer_reward*0.8
    
    return total_score


def extract_last_answer_content(solution_str):
    """
    提取最后一个<answer>标签中的内容
    
    参数:
        solution_str: 回答字符串
        
    返回:
        最后一个<answer>标签中的内容，如果找不到则返回空字符串
    """
    # 使用正则表达式查找所有<answer>标签内容
    answer_matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    
    if answer_matches:
        # 返回最后一个<answer>标签的内容
        return answer_matches[-1].strip()
    else:
        # 如果没有找到<answer>标签，返回空字符串
        return ""

# 示例用法
if __name__ == "__main__":
    # 测试用例1：选择题类型
    test_solution = "问题分析<think>这是思考过程</think><answer>最终答案是B</answer><code>最终答案</code><think>这是思考过程</think><answer>最终答案是C</answer>"
    test_ground_truth = "<answer>C</answer>"
    extra_info1 = {"problem_type": "multichoice"}
    score1 = compute_score("test", test_solution, test_ground_truth, extra_info1)
    print(f"测试1（选择题）得分: {score1}")
    
    # 测试用例2：数值题类型（完全正确）
    test_solution2 = "<think>计算过程</think><answer>42.0</answer>"
    test_ground_truth2 = "42"
    extra_info2 = {"problem_type": "numeric"}
    score2 = compute_score("test", test_solution2, test_ground_truth2, extra_info2)
    print(f"测试2（数值题，完全正确）得分: {score2}")
    
    # 测试用例3：数值题类型（有误差）
    test_solution3 = "<think>计算过程</think><answer>44.1</answer>"
    test_ground_truth3 = "42"
    extra_info3 = {"problem_type": "numeric"}
    score3 = compute_score("test", test_solution3, test_ground_truth3, extra_info3)
    print(f"测试3（数值题，有误差）得分: {score3}")
    
    # 测试用例4：未指定类型
    test_solution4 = "问题分析这是思考过程</think><answer>最终答案是42</answer>"
    test_ground_truth4 = "42"
    score4 = compute_score("test", test_solution4, test_ground_truth4)
    print(f"测试4（未指定类型）得分: {score4}")
    
    # 测试用例5：包含code块的复杂回答
    test_solution5 = """首先分析<think>初步思考</think><answer>初步答案</answer><code>print("hello")</code>
    <think>进一步思考</think><answer>最终答案是42</answer>"""
    score5 = compute_score("test", test_solution5, test_ground_truth4, extra_info2)
    print(f"测试5（包含code块）得分: {score5}")