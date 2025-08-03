import json
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 初始化 DeepSeek 接口
client = openai.OpenAI(
    api_key="sk-EUS-FAKi1Huycf-acQWqJQ",
    base_url="https://llmapi.paratera.com/v1/"
)

# 文件路径
input_json = "/root/shared-nvme/lichong/WCA-main/prompts/food101/cupl.json"
output_json = "food-qwen.json"

# 读取现有描述
with open(input_json, 'r', encoding='utf-8') as f:
    existing_descriptions = json.load(f)

updated_descriptions = {}

def evaluate_description(category, description):
    """使用大模型评估描述质量"""
    prompt = f"""Evaluate this visual description for {category}:

Description: "{description}"

Score 1-5 based on:
1. Accuracy (scientifically correct)
2. Specificity (detailed visual features)
3. Usefulness (helps distinguish from similar categories)
4. Clarity (unambiguous wording)

Output ONLY the numeric score (1-5) and nothing else:"""
    try:
        response = client.chat.completions.create(
            model="DeepSeek-R1-Distill-Qwen-7B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2
        )
        score = int(response.choices[0].message.content.strip())
        return score >= 4
    except:
        return False

def generate_new_descriptions(category, preserved_descs, request_count):
    """基于固定语义结构生成细粒度描述，利于CLIP分类"""
    prompt = f"""
The category "{category}" is often misclassified in CLIP-based zero-shot classification.
To improve classification accuracy, generate {request_count} unique, discriminative descriptions using the following 5 prompt forms:

(1) "Describe what a(n) {category} looks like."
(2) "How can you identify a(n) {category}?"
(3) "What does a(n) {category} look like?"
(4) "A caption of an image of a(n) {category}."
(5) "Describe an image from the internet of a(n) {category}."

Requirements:
- Each output should follow one of the 5 forms above, in natural language.
- Focus on fine-grained visual traits that help CLIP distinguish this class from similar ones.
- Include discriminative features: color, shape, texture, structure, context, etc.
- Use concrete visual language. Avoid vagueness.
- Do NOT repeat content from these existing descriptions:
{chr(10).join(preserved_descs[:3])}

Output ONLY the 40 generated descriptions, each on a new line:
"""

    response = client.chat.completions.create(
        model="DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048
    )
    raw_lines = response.choices[0].message.content.split('\n')
    return [line.strip() for line in raw_lines if len(line.strip()) >= 15]


def process_category(category, desc_list):
    try:
        # —— 1. 并行评估原始描述 —— 
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(evaluate_description, category, d): d for d in desc_list}
            preserved = []
            for future in as_completed(futures):
                if future.result():
                    preserved.append(futures[future])

        # —— 2. 生成新描述 —— 
        final = preserved.copy()
        num_needed = 50 - len(final)
        attempt = 0
        while num_needed > 0 and attempt < 5:
            new_descs = generate_new_descriptions(category, final, num_needed * 3)
            final.extend(new_descs[:num_needed])  # 添加至 final
            num_needed = 50 - len(final)
            attempt += 1
            print(f"[{category}] Round {attempt}: got {len(new_descs)} new, total = {len(final)}")
            if num_needed > 0:
                time.sleep(1)

        # —— 3. 不足时重复补齐 —— 
        if len(final) < 50:
            pad = (final * ((50 - len(final)) // len(final) + 1))[:50 - len(final)]
            final.extend(pad)

        # —— 4. 截断 —— 
        return category, final[:50]

    except Exception as e:
        print(f"Error processing {category}: {e}")
        return category, desc_list[:50]

# —— 主流程：对每类并行处理 —— 
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(process_category, cat, descs)
        for cat, descs in existing_descriptions.items()
    ]
    for future in tqdm(as_completed(futures), total=len(futures)):
        cat, descs = future.result()
        updated_descriptions[cat] = descs

# —— 保存结果 —— 
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(updated_descriptions, f, ensure_ascii=False, indent=2)

print(f"✅ 处理完成，结果保存至 {output_json}")
