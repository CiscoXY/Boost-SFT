import pandas as pd
import os
import json
import logging
from typing import Dict, List, Optional
import random

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_mapping_dict(mapping_csv_path: str) -> Dict[int, str]:
    """
    加载映射csv，构建item_id到sid的映射字典
    """
    try:
        df_mapping = pd.read_csv(mapping_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"加载映射csv失败: {e}")
    
    # 验证必要列
    required_columns = ["item_id", "sid"]
    missing_cols = [col for col in required_columns if col not in df_mapping.columns]
    if missing_cols:
        raise KeyError(f"映射csv缺少必要列: {missing_cols}")
    
    df_mapping["item_id"] = pd.to_numeric(df_mapping["item_id"], errors="coerce")
    df_mapping = df_mapping.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"], keep="first")
    df_mapping["item_id"] = df_mapping["item_id"].astype(int)
    
    mapping_dict = df_mapping.set_index("item_id")["sid"].to_dict()
    logging.info(f"成功构建映射字典，包含 {len(mapping_dict)} 个有效item_id->sid映射")
    return mapping_dict

def parse_truncate_config(truncate_config: Dict[str, int]) -> Dict[str, int]:
    """
    验证并处理字段截断配置（过滤无效值）
    """
    valid_truncate = {}
    for field, num in truncate_config.items():
        try:
            truncate_num = int(num)
            if truncate_num < 0:
                logging.warning(f"字段 {field} 的截断数不能为负数，设为0（不截断）")
                truncate_num = 0
            valid_truncate[field] = truncate_num
        except ValueError:
            logging.warning(f"字段 {field} 的截断数必须为整数，跳过该配置")
    return valid_truncate

def load_and_process_item_meta(
    meta_path: str,
    mapping_dict: Dict[int, str],
    target_fields: List[str],
    truncate_dict: Dict[str, int],
    sample_ratio: float = 1.0
) -> List[Dict[str, str]]:
    """
    加载item_meta.jsonl，处理字段截断、过滤无效item、抽样
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"item_meta.jsonl不存在：{meta_path}")
    
    # 验证抽样比例
    if not (0 < sample_ratio <= 1):
        raise ValueError(f"抽样比例必须在(0,1]之间，当前为：{sample_ratio}")
    
    logging.info(f"开始加载item_meta.jsonl，目标字段：{target_fields}，抽样比例：{sample_ratio}")
    
    processed_items = []
    total_count = 0
    valid_count = 0
    missing_field_count = 0
    missing_sid_count = 0
    
    # 逐行读取jsonl，避免内存占用过大
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logging.debug(f"无效的JSON格式（跳过）：{line[:50]}...")
                continue
            
            # 提取并验证item_id
            item_id = item.get("item_id")
            if item_id is None:
                continue
            try:
                item_id = int(item_id)
            except (ValueError, TypeError):
                continue
            
            # 检查是否有对应的sid
            sid = mapping_dict.get(item_id)
            if sid is None:
                missing_sid_count += 1
                continue
            
            # 提取目标字段并处理截断
            field_infos = []
            has_missing_field = False
            for field in target_fields:
                field_value = str(item.get(field, "")).strip()
                if not field_value:
                    has_missing_field = True
                    field_text = f"{field}:无"
                else:
                    # 按单词截断（空格分割）
                    truncate_num = truncate_dict.get(field, 0)
                    if truncate_num > 0:
                        words = field_value.split()
                        truncated_words = words[:truncate_num]
                        field_value = " ".join(truncated_words)
                    field_text = f"{field}:{field_value}"
                field_infos.append(field_text)
            
            # 拼接item_info
            item_info = ", ".join(field_infos) + " "
            
            # 统计缺失字段
            if has_missing_field:
                missing_field_count += 1
            
            processed_items.append({
                "item_info": item_info,
                "sid": sid,
                "item_id": item_id
            })
            valid_count += 1
    
    if sample_ratio < 1.0:
        sample_size = int(len(processed_items) * sample_ratio)
        random.seed(42)  # 固定随机种子，结果可复现
        processed_items = random.sample(processed_items, sample_size)
        logging.info(f"抽样完成：从 {valid_count} 个有效item中抽取 {sample_size} 个")
    else:
        logging.info(f"不抽样：共 {valid_count} 个有效item")
    
    logging.info(f"item_meta处理统计：")
    logging.info(f"  - 总记录数：{total_count}")
    logging.info(f"  - 有效记录数（含item_id+sid）：{valid_count}")
    logging.info(f"  - 无对应sid的记录数：{missing_sid_count}")
    logging.info(f"  - 存在缺失字段的有效记录数：{missing_field_count}")
    logging.info(f"  - 最终处理后item数：{len(processed_items)}")
    
    return processed_items

def construct_alignment_data(
    processed_items: List[Dict[str, str]],
    instruction1: str = "根据物品的元数据信息生成对应的sid",
    instruction2: str = "根据sid生成对应的物品元数据信息"
) -> List[Dict[str, str]]:
    """
    构造双向对齐数据：每个item生成2条数据（item info→sid 和 sid→item info）
    """
    logging.info("开始构造双向对齐数据...")
    
    alignment_data = []
    for item in processed_items:
        item_info = item["item_info"]
        sid = item["sid"]
        
        # 第一条：item info -> sid
        data1 = {
            "instruction": instruction1,
            "input": item_info,
            "output": sid
        }
        alignment_data.append(data1)
        
        # 第二条：sid -> item info
        data2 = {
            "instruction": instruction2,
            "input": f"sid:{sid}. ",
            "output": item_info
        }
        alignment_data.append(data2)
    
    logging.info(f"数据构造完成：共生成 {len(alignment_data)} 条记录（{len(processed_items)} 个item × 2）")
    return alignment_data

def save_jsonl(data: List[Dict[str, str]], output_path: str) -> None:
    """
    保存数据到jsonl文件
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + "\n")
    
    logging.info(f"成功保存jsonl文件到：{output_path}")
    logging.info(f"文件大小：{len(data)} 条记录")

def main(config: Dict):
    """
    主函数：执行全流程
    """
    try:
        logging.info("="*60)
        logging.info("开始执行文本对齐数据处理流程")
        logging.info("="*60)
        
        # 1. 加载映射字典
        mapping_dict = load_mapping_dict(config["mapping_csv"])
        
        logging.info("\n" + "="*60)
        # 2. 处理字段截断配置
        truncate_dict = parse_truncate_config(config["truncate_config"])
        logging.info(f"字段截断配置：{truncate_dict if truncate_dict else '无'}")
        
        logging.info("\n" + "="*60)
        # 3. 加载并处理item_meta
        processed_items = load_and_process_item_meta(
            meta_path=config["item_meta_path"],
            mapping_dict=mapping_dict,
            target_fields=config["target_fields"],
            truncate_dict=truncate_dict,
            sample_ratio=config["sample_ratio"]
        )
        
        logging.info("\n" + "="*60)
        # 4. 构造对齐数据
        alignment_data = construct_alignment_data(
            processed_items=processed_items,
            instruction1=config["instruction1"],
            instruction2=config["instruction2"]
        )
        
        logging.info("\n" + "="*60)
        # 5. 保存jsonl
        save_jsonl(alignment_data, config["output_jsonl"])
        
        logging.info("\n" + "="*60)
        logging.info("所有流程执行完成！")
        logging.info(f"最终输出：{config['output_jsonl']}（{len(alignment_data)} 条双向对齐数据）")
        
    except Exception as e:
        logging.error(f"\n流程执行失败：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # ====================== 配置参数（直接修改这里即可）======================
    CONFIG = {
        # 文件路径配置
        "mapping_csv": "/mnt/data/sid/Toys_and_Games/checkpoint_120000/itemid_to_sid.csv",  # itemid<->sid映射csv
        "item_meta_path": "/mnt/data/Toys & Games_items.jsonl",  # item_meta.jsonl路径
        "output_jsonl": "/mnt/data/sid_understand/sid_understand_Toys_and_Games.jsonl", 
        
        # 数据构造配置
        "target_fields": ["title","features", "description"],  # 参与构造的字段列表（根据实际meta字段修改）
        "truncate_config": {  # 字段截断配置：key=字段名，value=截断数（0表示不截断）
            "title": 0,     
            "description": 300,  # toys and games
            "features": 150,
        },
        "sample_ratio": 0.5,
        
        # 指令文本配置（可自定义）
        "instruction1": "Please generate the corresponding semantic id based on the given item information",
        "instruction2": "Please restore the item information based on the given semantic id."
    }
    
    # 执行主流程
    main(CONFIG)