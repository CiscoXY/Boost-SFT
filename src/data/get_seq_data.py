import pandas as pd
import argparse
import os
import ast
from typing import Dict, List, Optional
import logging

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_mapping_dict(mapping_csv_path: str) -> Dict[int, str]:
    """
    加载映射csv，构建item_id到sid的映射字典
    
    Args:
        mapping_csv_path: 上一步生成的主csv路径（包含item_id和sid列）
        
    Returns:
        映射字典：key为int类型的item_id，value为对应的sid字符串
    """
    # 加载csv
    try:
        df_mapping = pd.read_csv(mapping_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"加载映射csv失败: {e}")
    
    # 验证必要列是否存在
    required_columns = ["item_id", "sid"]
    missing_cols = [col for col in required_columns if col not in df_mapping.columns]
    if missing_cols:
        raise KeyError(f"映射csv缺少必要列: {missing_cols}")
    
    # 验证item_id为整数类型（处理可能的字符串格式）
    df_mapping["item_id"] = pd.to_numeric(df_mapping["item_id"], errors="coerce")
    if df_mapping["item_id"].isna().any():
        logging.warning(f"映射csv中有 {df_mapping['item_id'].isna().sum()} 行的item_id无法转换为整数，已跳过")
        df_mapping = df_mapping.dropna(subset=["item_id"])
    
    # 转换为int类型并去重（保留第一个出现的映射）
    df_mapping["item_id"] = df_mapping["item_id"].astype(int)
    df_mapping = df_mapping.drop_duplicates(subset=["item_id"], keep="first")
    
    # 构建映射字典
    mapping_dict = df_mapping.set_index("item_id")["sid"].to_dict()
    logging.info(f"成功构建映射字典，共包含 {len(mapping_dict)} 个有效item_id->sid映射")
    
    return mapping_dict

def load_userseq_csv(userseq_csv_path: str) -> pd.DataFrame:
    """
    加载userseq.csv并解析history_item_id字段
    
    Args:
        userseq_csv_path: userseq.csv文件路径
        
    Returns:
        解析后的DataFrame，history_item_id为list[int]类型
    """
    # 加载csv
    try:
        df_userseq = pd.read_csv(userseq_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"加载userseq.csv失败: {e}")
    
    # 验证必要列是否存在
    required_columns = ["history_item_id", "item_id"]
    missing_cols = [col for col in required_columns if col not in df_userseq.columns]
    if missing_cols:
        raise KeyError(f"userseq.csv缺少必要列: {missing_cols}")
    
    # 解析history_item_id（字符串格式的列表 -> 实际列表）
    def parse_history(history_str: str) -> Optional[List[int]]:
        try:
            # 使用ast.literal_eval安全解析字符串列表
            history_list = ast.literal_eval(history_str)
            if not isinstance(history_list, list):
                return None
            # 转换为整数列表，过滤非整数元素
            history_int = [int(x) for x in history_list if isinstance(x, (int, float)) and pd.notna(x)]
            return history_int if history_int else None
        except:
            return None
    
    logging.info("开始解析history_item_id字段...")
    df_userseq["history_item_list"] = df_userseq["history_item_id"].apply(parse_history)
    
    # 过滤解析失败或空列表的行
    before_filter = len(df_userseq)
    df_userseq = df_userseq.dropna(subset=["history_item_list"])
    df_userseq = df_userseq[df_userseq["history_item_list"].apply(lambda x: len(x) > 0)]
    
    # 验证并转换item_id为整数
    df_userseq["item_id"] = pd.to_numeric(df_userseq["item_id"], errors="coerce")
    df_userseq = df_userseq.dropna(subset=["item_id"])
    df_userseq["item_id"] = df_userseq["item_id"].astype(int)
    
    after_filter = len(df_userseq)
    logging.info(f"userseq.csv解析完成：原始 {before_filter} 行，有效 {after_filter} 行（过滤了解析失败/空历史/无效item_id的行）")
    
    return df_userseq[["history_item_list", "item_id"]]

def convert_userseq_data(
    df_userseq: pd.DataFrame,
    mapping_dict: Dict[int, str],
    missing_id_handling: str = "skip"
) -> pd.DataFrame:
    """
    将userseq中的int类型id转换为sid
    
    Args:
        df_userseq: 解析后的userseq DataFrame
        mapping_dict: item_id->sid映射字典
        missing_id_handling: 缺失id处理方式，"skip"跳过该行，"error"抛出异常
        
    Returns:
        转换后的DataFrame，包含history_sid_str（逗号分隔）和output_sid
    """
    logging.info("开始进行id->sid映射转换...")
    
    # 转换history_item_list为sid列表，再拼接为字符串
    def map_history_to_sid(history_list: List[int]) -> Optional[str]:
        # 过滤出存在映射的sid
        sid_list = [mapping_dict.get(item_id) for item_id in history_list]
        valid_sid_list = [sid for sid in sid_list if sid is not None]
        
        # 根据缺失处理方式判断
        if missing_id_handling == "error" and len(valid_sid_list) != len(history_list):
            raise ValueError(f"历史item_id中存在未映射的id：{[item_id for item_id, sid in zip(history_list, sid_list) if sid is None]}")
        
        return ",".join(valid_sid_list) if valid_sid_list else None
    
    # 转换history_item_list
    df_userseq["history_sid_str"] = df_userseq["history_item_list"].apply(map_history_to_sid)
    
    # 转换item_id
    df_userseq["output_sid"] = df_userseq["item_id"].map(mapping_dict)
    
    # 过滤无效数据（history_sid_str为空或output_sid为空）
    before_filter = len(df_userseq)
    df_userseq = df_userseq.dropna(subset=["history_sid_str", "output_sid"])
    after_filter = len(df_userseq)
    
    logging.info(f"映射转换完成：原始 {before_filter} 行，有效 {after_filter} 行（过滤了无有效映射的行）")
    
    # 只保留需要的列
    return df_userseq[["history_sid_str", "output_sid"]]

def save_converted_csv(df_converted: pd.DataFrame, converted_csv_path: str) -> None:
    """
    保存转换后的数据到指定路径
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(converted_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存csv
    df_converted.to_csv(converted_csv_path, index=False, encoding="utf-8")
    logging.info(f"成功保存转换后的数据到：{converted_csv_path}")
    logging.info(f"转换后的数据共 {len(df_converted)} 行，包含列：{list(df_converted.columns)}")

def convert_to_jsonl(
    df_converted: pd.DataFrame,
    jsonl_path: str,
    instruction: str,
    mode: str = "train"
) -> None:
    """
    将转换后的数据转换为jsonl格式并保存
    
    Args:
        df_converted: 转换后的DataFrame
        jsonl_path: 输出jsonl文件路径
        instruction: 指定的instruction字符串
        mode: train和test两种可选, train的话只会生成包含三个和核心字段的data, test的话会生成2个文件, 
            一个只包含instruction和input , 一个全部包含
    """
    logging.info("开始转换为jsonl格式...")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(jsonl_path)
    jsonl_basename = os.path.splitext(os.path.basename(jsonl_path))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 逐行生成json并写入文件
    if mode == "train":
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df_converted.iterrows():
                json_line = {
                    "instruction": instruction,
                    "input": row["history_sid_str"],
                    "output": row["output_sid"]
                }
                # 使用json.dumps确保格式正确
                import json
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    elif mode == "test":
        ground_truth_path = os.path.join(output_dir , jsonl_basename + "_groundtruth.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df_converted.iterrows():
                json_line = {
                    "instruction": instruction,
                    "input": row["history_sid_str"],
                }
                # 使用json.dumps确保格式正确
                import json
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        with open(ground_truth_path, "w", encoding="utf-8") as f:
            for _, row in df_converted.iterrows():
                json_line = {
                    "instruction": instruction,
                    "input": row["history_sid_str"],
                    "output": row["output_sid"]
                }
                # 使用json.dumps确保格式正确
                import json
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    
    logging.info(f"成功保存jsonl文件到：{jsonl_path}")
    logging.info(f"jsonl文件共 {len(df_converted)} 条数据")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理userseq数据：id->sid映射转换并生成jsonl训练数据")
    parser.add_argument("--mapping_csv", default="/mnt/data/sid/Books/checkpoint_120000/itemid_to_sid.csv", help="上一步生成的映射csv路径（包含item_id和sid列）")
    parser.add_argument("--userseq_csv", default="/mnt/data/userseq/valid/Books.csv", help="输入的userseq.csv路径")
    parser.add_argument("--converted_csv_path", default="/mnt/data/userseq/valid/Books_temp.csv", help="转换后中间数据的保存路径（csv格式）")
    parser.add_argument("--jsonl_path", default="/mnt/data/userseq/valid/Books_llm.jsonl", help="输出jsonl文件的路径")
    parser.add_argument("--mode", default="train", help="生成用户行为data的模式 , 'train' or 'test' ")
    parser.add_argument("--instruction", default="You are a professional search recommendation engine. Please predict the semantic id of the next item to be visited based on the user's historical visit records. The following are the historical records:", help="jsonl中的instruction字段内容")
    parser.add_argument(
        "--missing_id_handling",
        choices=["skip", "error"],
        default="error",
        help="处理未找到映射的item_id的方式：skip=跳过该行，error=抛出异常（默认：skip）"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件存在
    for path in [args.mapping_csv, args.userseq_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"输入文件不存在：{path}")
    
    # 执行全流程
    try:
        logging.info("="*60)
        logging.info("开始加载映射字典...")
        mapping_dict = load_mapping_dict(args.mapping_csv)
        
        logging.info("\n" + "="*60)
        logging.info("开始加载并解析userseq.csv...")
        df_userseq = load_userseq_csv(args.userseq_csv)
        
        logging.info("\n" + "="*60)
        df_converted = convert_userseq_data(
            df_userseq=df_userseq,
            mapping_dict=mapping_dict,
            missing_id_handling=args.missing_id_handling
        )
        
        logging.info("\n" + "="*60)
        save_converted_csv(df_converted, args.converted_csv_path)
        
        logging.info("\n" + "="*60)
        convert_to_jsonl(df_converted, args.jsonl_path, args.instruction , mode = args.mode)
        
        logging.info("\n" + "="*60)
        logging.info("所有流程执行完成！")
        logging.info(f"最终生成：")
        logging.info(f"  - 中间转换数据：{args.converted_csv_path}（{len(df_converted)} 行）")
        logging.info(f"  - JSONL训练数据：{args.jsonl_path}（{len(df_converted)} 条）")
    except Exception as e:
        logging.error(f"\n流程执行失败：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    
    

'''
帮我修改上述代码，要求如下
不需要test模式，只有train模式，同时三个字段保留，但是从instruction、input、output改成
instruction、input、chosen
同时增加rejected字段
该字段根据以下三种不同的规则进行选择(三种规则通过参数传入)
1. rejected的sid的sid1和chosen就不同
2. rejected的sid的sid1和chosen相同，剩下的不同
3. rejected的sid1和sid2都和chosen相同，但是sid3不同

注意: itemid_to_sid映射表包含五列，item_id , sid , sid1 , sid2 , sid3，其中sid就是sid1，2，3的concat，sid形如<a_123><b_112><c_212>这样
'''