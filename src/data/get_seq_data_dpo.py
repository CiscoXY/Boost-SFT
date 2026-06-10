import pandas as pd
import argparse
import os
import ast
import json
from typing import Dict, List, Optional, Tuple
import logging
import random
from tqdm import tqdm

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_mapping_dict(mapping_csv_path: str) -> Tuple[Dict[int, str], Dict[int, Tuple[str, str, str]], List[Tuple[str, str, str, str]], List[str], List[str], List[str], set[str]]:
    """
    加载映射csv,构建多类映射关系,新增收集sid1/sid2/sid3池和现有有效sid集合
    
    Args:
        mapping_csv_path: 包含item_id、sid、sid1、sid2、sid3的映射csv路径
        
    Returns:
        1. item_id->sid的映射字典
        2. item_id->(sid1, sid2, sid3)的映射字典
        3. 所有有效sid的候选池：[(sid, sid1, sid2, sid3), ...]
        4. 所有唯一sid1的集合(转列表)
        5. 所有唯一sid2的集合(转列表)
        6. 所有唯一sid3的集合(转列表)
        7. 所有现有有效sid的集合(用于判断新组合sid是否非法)
    """
    # 加载csv
    try:
        df_mapping = pd.read_csv(mapping_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"加载映射csv失败: {e}")
    
    # 验证必要列是否存在
    required_columns = ["item_id", "sid", "sid1", "sid2", "sid3"]
    missing_cols = [col for col in required_columns if col not in df_mapping.columns]
    if missing_cols:
        raise KeyError(f"映射csv缺少必要列: {missing_cols}")
    
    # 验证item_id为整数类型(处理可能的字符串格式)
    df_mapping["item_id"] = pd.to_numeric(df_mapping["item_id"], errors="coerce")
    if df_mapping["item_id"].isna().any():
        logging.warning(f"映射csv中有 {df_mapping['item_id'].isna().sum()} 行的item_id无法转换为整数,已跳过")
        df_mapping = df_mapping.dropna(subset=["item_id"])
    
    # 验证sid1/sid2/sid3非空
    sid_cols = ["sid1", "sid2", "sid3"]
    for col in sid_cols:
        df_mapping[col] = df_mapping[col].astype(str).str.strip()
        empty_count = df_mapping[col].isin(["", "nan", None]).sum()
        if empty_count > 0:
            logging.warning(f"映射csv中 {col} 列有 {empty_count} 行空值,已跳过")
            df_mapping = df_mapping[df_mapping[col].notna() & (df_mapping[col] != "")]
    
    # 转换为int类型并去重(保留第一个出现的映射)
    df_mapping["item_id"] = df_mapping["item_id"].astype(int)
    df_mapping = df_mapping.drop_duplicates(subset=["item_id"], keep="first")
    
    # 构建各类映射字典
    item_to_sid = df_mapping.set_index("item_id")["sid"].to_dict()
    item_to_sid_parts = df_mapping.set_index("item_id")[sid_cols].apply(tuple, axis=1).to_dict()
    
    # 构建候选池(包含sid和对应的三个子sid)
    candidate_pool = df_mapping[["sid", "sid1", "sid2", "sid3"]].apply(tuple, axis=1).tolist()
    
    # 新增：收集唯一的sid1/sid2/sid3池(用于rule 0随机组合)
    sid1_pool = df_mapping["sid1"].unique().tolist()
    sid2_pool = df_mapping["sid2"].unique().tolist()
    sid3_pool = df_mapping["sid3"].unique().tolist()
    
    # 新增：收集所有现有有效sid的集合(用于判断新sid是否非法)
    existing_sids = set(df_mapping["sid"].tolist())
    
    logging.info(f"成功构建映射关系,共包含 {len(item_to_sid)} 个有效item_id")
    logging.info(f"sid候选池大小：{len(candidate_pool)}")
    logging.info(f"sid1池大小：{len(sid1_pool)} | sid2池大小：{len(sid2_pool)} | sid3池大小：{len(sid3_pool)}")
    logging.info(f"现有有效sid总数：{len(existing_sids)}")
    
    return item_to_sid, item_to_sid_parts, candidate_pool, sid1_pool, sid2_pool, sid3_pool, existing_sids

def load_userseq_csv(userseq_csv_path: str) -> pd.DataFrame:
    """
    加载userseq.csv并解析history_item_id字段
    
    Args:
        userseq_csv_path: userseq.csv文件路径
        
    Returns:
        解析后的DataFrame,history_item_id为list[int]类型
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
    
    # 解析history_item_id(字符串格式的列表 -> 实际列表)
    def parse_history(history_str: str) -> Optional[List[int]]:
        try:
            # 使用ast.literal_eval安全解析字符串列表
            history_list = ast.literal_eval(history_str)
            if not isinstance(history_list, list):
                return None
            # 转换为整数列表,过滤非整数元素
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
    logging.info(f"userseq.csv解析完成：原始 {before_filter} 行,有效 {after_filter} 行(过滤了解析失败/空历史/无效item_id的行)")
    
    return df_userseq[["history_item_list", "item_id"]]

def convert_userseq_data(
    df_userseq: pd.DataFrame,
    item_to_sid: Dict[int, str],
    item_to_sid_parts: Dict[int, Tuple[str, str, str]],
    missing_id_handling: str = "skip"
) -> pd.DataFrame:
    """
    将userseq中的int类型id转换为sid及对应的sid1/sid2/sid3
    
    Args:
        df_userseq: 解析后的userseq DataFrame
        item_to_sid: item_id->sid映射字典
        item_to_sid_parts: item_id->(sid1, sid2, sid3)映射字典
        missing_id_handling: 缺失id处理方式,"skip"跳过该行,"error"抛出异常
        
    Returns:
        转换后的DataFrame,包含history_sid_str、chosen、sid1、sid2、sid3
    """
    logging.info("开始进行id->sid映射转换...")
    
    # 转换history_item_list为sid列表,再拼接为字符串
    def map_history_to_sid(history_list: List[int]) -> Optional[str]:
        # 过滤出存在映射的sid
        sid_list = [item_to_sid.get(item_id) for item_id in history_list]
        valid_sid_list = [sid for sid in sid_list if sid is not None]
        
        # 根据缺失处理方式判断
        if missing_id_handling == "error" and len(valid_sid_list) != len(history_list):
            raise ValueError(f"历史item_id中存在未映射的id：{[item_id for item_id, sid in zip(history_list, sid_list) if sid is None]}")
        
        return ",".join(valid_sid_list) if valid_sid_list else None
    
    # 转换history_item_list
    df_userseq["history_sid_str"] = df_userseq["history_item_list"].apply(map_history_to_sid)
    
    # 转换item_id到chosen(sid)和对应的sid1/sid2/sid3
    df_userseq["chosen"] = df_userseq["item_id"].map(item_to_sid)
    df_userseq["sid_parts"] = df_userseq["item_id"].map(item_to_sid_parts)
    
    # 拆分sid_parts为单独列
    df_userseq[["sid1", "sid2", "sid3"]] = pd.DataFrame(
        df_userseq["sid_parts"].tolist(),
        index=df_userseq.index
    )
    
    # 过滤无效数据(history_sid_str为空、chosen为空或sid_parts为空)
    before_filter = len(df_userseq)
    df_userseq = df_userseq.dropna(subset=["history_sid_str", "chosen", "sid1", "sid2", "sid3"])
    after_filter = len(df_userseq)
    
    logging.info(f"映射转换完成：原始 {before_filter} 行,有效 {after_filter} 行(过滤了无有效映射的行)")
    
    # 只保留需要的列
    return df_userseq[["history_sid_str", "chosen", "sid1", "sid2", "sid3"]]

def generate_rejected_sid(
    chosen_sid1: str,
    chosen_sid2: str,
    chosen_sid3: str,
    chosen_sid: str,
    candidate_pool: List[Tuple[str, str, str, str]],
    rule: int,
    sid1_pool: Optional[List[str]] = None,
    sid2_pool: Optional[List[str]] = None,
    sid3_pool: Optional[List[str]] = None,
    existing_sids: Optional[set[str]] = None,
    sid_separator: str = ""  # sid拼接分隔符,需与现有sid格式一致
) -> Optional[str]:
    """
    根据指定规则生成rejected sid,新增rule 0(非法sid随机组合)
    
    Args:
        chosen_sid1: chosen对应的sid1
        chosen_sid2: chosen对应的sid2
        chosen_sid3: chosen对应的sid3
        chosen_sid: chosen本身的sid(用于避免rejected和chosen重复)
        candidate_pool: sid候选池,每个元素为(sid, sid1, sid2, sid3)
        rule: 生成规则(0/1/2/3)
        sid1_pool: sid1候选池(仅rule 0使用)
        sid2_pool: sid2候选池(仅rule 0使用)
        sid3_pool: sid3候选池(仅rule 0使用)
        existing_sids: 现有有效sid集合(仅rule 0使用)
        sid_separator: sid拼接分隔符(默认下划线,需与现有sid格式匹配)
        
    Returns:
        符合规则的rejected sid,无有效结果时返回None
    """
    # 新增：rule 0 - 随机组合sid1/sid2/sid3生成非法sid(不存在于existing_sids)
    if rule == 0:
        # 验证必要参数是否传入
        if not all([sid1_pool, sid2_pool, sid3_pool, existing_sids]):
            logging.error("使用rule 0时,必须传入sid1_pool、sid2_pool、sid3_pool和existing_sids参数")
            return None
        
        max_attempts = 100  # 最大尝试次数,避免无限循环
        attempt = 0
        
        while attempt < max_attempts:
            # 从各池随机选择一个元素
            rand_sid1 = random.choice(sid1_pool)
            rand_sid2 = random.choice(sid2_pool)
            rand_sid3 = random.choice(sid3_pool)
            
            # 拼接成新sid
            new_sid = f"{rand_sid1}{sid_separator}{rand_sid2}{sid_separator}{rand_sid3}"
            
            # 判断是否为非法sid：1. 不在现有有效sid中 2. 不等于chosen_sid
            if new_sid not in existing_sids and new_sid != chosen_sid:
                return new_sid
            
            attempt += 1
        
        # 超过最大尝试次数仍未找到
        logging.warning(f"尝试{max_attempts}次后仍未生成符合要求的非法sid(chosen：{chosen_sid})")
        return None
    
    # 原有规则1/2/3逻辑
    valid_candidates = []
    vc_length = 0
    for cand_sid, cand_sid1, cand_sid2, cand_sid3 in candidate_pool:
        # 确保rejected和chosen不重复
        if cand_sid == chosen_sid:
            continue
        
        # 根据不同规则筛选
        if rule == 1:
            # 规则1：rejected的sid1和chosen不同
            if cand_sid1 != chosen_sid1:
                valid_candidates.append(cand_sid)
                vc_length += 1
        elif rule == 2:
            # 规则2：rejected的sid1和chosen相同,sid2不同
            if cand_sid1 == chosen_sid1 and cand_sid2 != chosen_sid2:
                valid_candidates.append(cand_sid)
                vc_length += 1
        elif rule == 3:
            # 规则3：rejected的sid1、sid2和chosen相同,sid3不同
            if cand_sid1 == chosen_sid1 and cand_sid2 == chosen_sid2 and cand_sid3 != chosen_sid3:
                valid_candidates.append(cand_sid)
                vc_length += 1
        if vc_length >= 50:
            break
    # 随机返回一个有效候选(如果有)
    if valid_candidates:
        return random.choice(valid_candidates)
    else:
        return None

def save_converted_csv(df_converted: pd.DataFrame, converted_csv_path: str) -> None:
    """
    保存转换后的数据到指定路径
    """
    # 创建输出目录(如果不存在)
    output_dir = os.path.dirname(converted_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存csv
    df_converted.to_csv(converted_csv_path, index=False, encoding="utf-8")
    logging.info(f"成功保存转换后的数据到：{converted_csv_path}")
    logging.info(f"转换后的数据共 {len(df_converted)} 行,包含列：{list(df_converted.columns)}")

def convert_to_jsonl(
    df_converted: pd.DataFrame,
    jsonl_path: str,
    instruction: str,
    candidate_pool: List[Tuple[str, str, str, str]],
    rejected_rule: int,
    sid1_pool: Optional[List[str]] = None,
    sid2_pool: Optional[List[str]] = None,
    sid3_pool: Optional[List[str]] = None,
    existing_sids: Optional[set[str]] = None,
    sid_separator: str = "_"
) -> None:
    """
    将转换后的数据转换为jsonl格式(仅train模式)并保存,包含instruction/input/chosen/rejected字段
    新增支持rule 0的参数传递
    
    Args:
        df_converted: 转换后的DataFrame
        jsonl_path: 输出jsonl文件路径
        instruction: 指定的instruction字符串
        candidate_pool: sid候选池
        rejected_rule: rejected生成规则(0/1/2/3)
        sid1_pool: sid1候选池(仅rule 0使用)
        sid2_pool: sid2候选池(仅rule 0使用)
        sid3_pool: sid3候选池(仅rule 0使用)
        existing_sids: 现有有效sid集合(仅rule 0使用)
        sid_separator: sid拼接分隔符(默认下划线)
    """
    logging.info(f"开始按规则 {rejected_rule} 生成rejected字段并转换为jsonl格式...")
    
    # 创建输出目录(如果不存在)
    output_dir = os.path.dirname(jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成jsonl数据并过滤无有效rejected的样本
    valid_samples = []
    for _, row in tqdm(df_converted.iterrows()):
        rejected_sid = generate_rejected_sid(
            chosen_sid1=row["sid1"],
            chosen_sid2=row["sid2"],
            chosen_sid3=row["sid3"],
            chosen_sid=row["chosen"],
            candidate_pool=candidate_pool,
            rule=rejected_rule,
            sid1_pool=sid1_pool,
            sid2_pool=sid2_pool,
            sid3_pool=sid3_pool,
            existing_sids=existing_sids,
            sid_separator=sid_separator
        )
        if rejected_sid:
            json_line = {
                "instruction": instruction,
                "input": row["history_sid_str"],
                "chosen": row["chosen"],
                "rejected": rejected_sid
            }
            valid_samples.append(json_line)
        else:
            logging.warning(f"无符合规则 {rejected_rule} 的rejected sid,跳过该样本(chosen：{row['chosen']})")
    
    # 写入文件
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logging.info(f"成功保存jsonl文件到：{jsonl_path}")
    logging.info(f"jsonl文件共 {len(valid_samples)} 条有效数据(过滤了无有效rejected的样本)")

def main():
    parser = argparse.ArgumentParser(description="处理userseq数据：id->sid映射转换并生成含chosen/rejected的jsonl训练数据")
    parser.add_argument("--mapping_csv", default="/mnt/data/sid/Books/checkpoint_120000/itemid_to_sid.csv", help="包含item_id、sid、sid1、sid2、sid3的映射csv路径")
    parser.add_argument("--userseq_csv", default="/mnt/data/userseq/train/Books.csv", help="输入的userseq.csv路径")
    parser.add_argument("--converted_csv_path", default="/mnt/data/userseq/train/Books_temp.csv", help="转换后中间数据的保存路径(csv格式)")
    parser.add_argument("--jsonl_path", default="/mnt/data/userseq/train/Books_llm_dpo_stage0.jsonl", help="输出jsonl文件的路径")
    parser.add_argument("--instruction", default="You are a professional search recommendation engine. Please predict the semantic id of the next item to be visited based on the user's historical visit records. The following are the historical records:", help="jsonl中的instruction字段内容")
    parser.add_argument(
        "--missing_id_handling",
        choices=["skip", "error"],
        default="skip",
        help="处理未找到映射的item_id的方式 : skip=跳过该行, error=抛出异常(默认 : skip)"
    )
    parser.add_argument(
        "--rejected_rule",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="rejected字段生成规则 : 0=非法sid(sid1/sid2/sid3随机组合,非现有sid) : 1=sid1不同 : 2=sid1相同、sid2不同 : 3=sid1/sid2相同、sid3不同"
    )
    parser.add_argument(
        "--sid_separator",
        type=str,
        default="",
        help="sid拼接分隔符(仅rule 0使用,需与现有sid格式一致,默认：空)"
    )
    
    args = parser.parse_args()
    
    for path in [args.mapping_csv, args.userseq_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"输入文件不存在：{path}")
    
    try:
        logging.info("="*60)
        logging.info("开始加载映射字典及候选池...")
        
        item_to_sid, item_to_sid_parts, candidate_pool, sid1_pool, sid2_pool, sid3_pool, existing_sids = load_mapping_dict(args.mapping_csv)
        
        logging.info("\n" + "="*60)
        logging.info("开始加载并解析userseq.csv...")
        df_userseq = load_userseq_csv(args.userseq_csv)
        
        logging.info("\n" + "="*60)
        df_converted = convert_userseq_data(
            df_userseq=df_userseq,
            item_to_sid=item_to_sid,
            item_to_sid_parts=item_to_sid_parts,
            missing_id_handling=args.missing_id_handling
        )
        
        logging.info("\n" + "="*60)
        save_converted_csv(df_converted, args.converted_csv_path)
        
        logging.info("\n" + "="*60)
        convert_to_jsonl(
            df_converted=df_converted,
            jsonl_path=args.jsonl_path,
            instruction=args.instruction,
            candidate_pool=candidate_pool,
            rejected_rule=args.rejected_rule,
            sid1_pool=sid1_pool,
            sid2_pool=sid2_pool,
            sid3_pool=sid3_pool,
            existing_sids=existing_sids,
            sid_separator=args.sid_separator
        )
        
        logging.info("\n" + "="*60)
        logging.info("所有流程执行完成！")
        # 统计最终有效jsonl样本数
        with open(args.jsonl_path, "r", encoding="utf-8") as f:
            jsonl_count = len(f.readlines())
        logging.info(f"最终生成：")
        logging.info(f"  - 中间转换数据：{args.converted_csv_path}({len(df_converted)} 行)")
        logging.info(f"  - JSONL训练数据：{args.jsonl_path}({jsonl_count} 条有效样本)")
    except Exception as e:
        logging.error(f"\n流程执行失败：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()