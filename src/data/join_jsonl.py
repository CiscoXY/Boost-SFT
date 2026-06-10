'''
该文件主要作用是制作llm_data --- mix的数据, join k个不同的jsonl
支持每个文件随机抽取指定数量样本（k参数控制）
'''
import json
import random
import os
from typing import List, Dict, Set, Tuple


def load_jsonl_file(file_path: str) -> Tuple[List[Dict], int]:
    """
    读取单个JSONL文件，返回数据列表和样本数
    :param file_path: JSONL文件路径（字符串）
    :return: (数据列表, 样本数)
    """
    data = []
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"⚠️  警告：文件 {file_path} 第 {line_num} 行JSON格式错误，已跳过：{e}")
        return data, line_count
    except Exception as e:
        print(f"❌ 错误：读取文件 {file_path} 失败：{e}")
        raise


def check_field_consistency(all_data: List[Tuple[List[Dict], str]]) -> Set[str]:
    """
    校验所有文件的字段一致性
    :param all_data: 列表，每个元素是(文件数据, 文件路径)
    :return: 统一的字段集合
    """
    if not all_data:
        raise ValueError("没有有效数据文件需要处理")

    # 获取第一个非空文件的第一个样本的字段作为基准
    base_fields = None
    for data, file_path in all_data:
        if data:
            base_fields = set(data[0].keys())
            print(f"📋 以文件 {file_path} 的字段作为基准：{sorted(base_fields)}")
            break

    if not base_fields:
        raise ValueError("所有输入文件都没有有效数据")

    # 校验所有文件的所有样本
    for data, file_path in all_data:
        for idx, sample in enumerate(data):
            sample_fields = set(sample.keys())
            if sample_fields != base_fields:
                missing = base_fields - sample_fields
                extra = sample_fields - base_fields
                error_msg = f"❌ 字段不一致：文件 {file_path} 第 {idx+1} 行"
                if missing:
                    error_msg += f" 缺少字段：{sorted(missing)}"
                if extra:
                    error_msg += f" 多余字段：{sorted(extra)}"
                raise ValueError(error_msg)

    print("✅ 所有文件字段一致性校验通过")
    return base_fields


def merge_jsonl_files(
    input_paths: List[str], 
    output_path: str, 
    seed: int = 42,
    k: int = 0  # 新增参数：每个文件抽取的样本数，0表示全部抽取
) -> None:
    """
    合并多个JSONL文件，支持每个文件随机抽取指定数量样本
    :param input_paths: 输入JSONL文件路径列表（字符串）
    :param output_path: 输出合并后的JSONL文件路径（字符串）
    :param seed: shuffle随机种子（保证可复现）
    :param k: 每个文件抽取的样本数，0表示不抽样（取全部样本），正数表示随机抽取对应数量
    """
    # 验证k参数有效性
    if k < 0:
        print(f"❌ 错误：k值不能为负数（当前k={k}）")
        return
    
    # 1. 读取所有文件并根据k值抽样
    all_data = []
    file_stats = []  # 存储 (文件名, 原始样本数, 抽取后样本数)
    total_samples = 0
    random.seed(seed)  # 抽样使用统一种子，保证可复现

    print("📥 开始读取输入文件并抽样...")
    for path in input_paths:
        if not os.path.exists(path):
            print(f"⚠️  警告：文件 {path} 不存在，已跳过")
            continue
        if not os.path.isfile(path):
            print(f"⚠️  警告：{path} 不是文件，已跳过")
            continue
        
        data, original_count = load_jsonl_file(path)
        file_name = os.path.basename(path)
        
        # 根据k值进行抽样
        if k == 0:
            sampled_data = data
            sampled_count = original_count
        else:
            # 当k大于文件总样本数时，取全部样本
            if k >= original_count:
                sampled_data = data
                sampled_count = original_count
                print(f"   注意：{file_name} 原始样本数（{original_count}）≤ k（{k}），将使用全部样本")
            else:
                sampled_data = random.sample(data, k=k)
                sampled_count = k
        
        all_data.append((sampled_data, path))
        file_stats.append((file_name, original_count, sampled_count))
        total_samples += sampled_count
        print(f"   已处理 {file_name}：原始{original_count}条 → 抽取{sampled_count}条")

    if not all_data:
        print("❌ 错误：没有找到有效可读取的JSONL文件")
        return

    # 2. 字段一致性校验
    try:
        base_fields = check_field_consistency(all_data)
    except ValueError as e:
        print(f"❌ 字段校验失败：{e}")
        return

    # 3. 合并所有数据
    merged_data = []
    for data, _ in all_data:
        merged_data.extend(data)

    # 4. 数据shuffle（合并后整体shuffle）
    print(f"🔀 开始对合并后的 {total_samples} 条样本进行shuffle...")
    random.shuffle(merged_data)  # 已设置seed，保证可复现
    print("✅ Shuffle完成")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 已创建输出目录：{output_dir}")

    # 5. 写入输出文件
    print(f"📤 开始写入输出文件 {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in merged_data:
                # 按字段顺序排序输出（增强可读性）
                sorted_sample = dict(sorted(sample.items()))
                f.write(json.dumps(sorted_sample, ensure_ascii=False) + '\n')
        print(f"✅ 输出文件已保存至：{output_path}")
    except Exception as e:
        print(f"❌ 写入输出文件失败：{e}")
        return

    # 6. 输出统计报告
    print("\n📊 合并统计报告")
    print("-" * 60)
    print(f"输入文件总数：{len(file_stats)} 个")
    print(f"抽样参数k：{k}（0表示全部抽取）")
    print("各文件处理情况：")
    for file_name, original_count, sampled_count in file_stats:
        print(f"  - {file_name}: 原始{original_count}条 → 抽取{sampled_count}条")
    print(f"合并后总样本量：{total_samples} 条")
    print(f"字段集合：{sorted(base_fields)}")
    print(f"字段数量：{len(base_fields)} 个")
    print(f"Shuffle种子：{seed}")
    print(f"输出文件路径：{output_path}")
    print("-" * 60)


if __name__ == "__main__":
    # 配置参数
    INPUT_FILES = [
        "/mnt/data/userseq/train/Books_llm.jsonl",
        "/mnt/data/sid_understand/sid_understand_Books.jsonl",
    ]
    OUTPUT_FILE = "/mnt/data/llm_data/mix/Books/train_mix_200_samples.jsonl"
    #OUTPUT_FILE = "/mnt/data/llm_data/mix/test/pipeline_test.jsonl"
    SHUFFLE_SEED = 42
    K = 200  # 每个文件抽取k个样本（k=0时取全部）

    print("🚀 开始执行JSONL文件合并任务...")
    merge_jsonl_files(
        input_paths=INPUT_FILES,
        output_path=OUTPUT_FILE,
        seed=SHUFFLE_SEED,
        k=K  # 传入抽样参数
    )
    print("\n🎉 任务执行完毕！")