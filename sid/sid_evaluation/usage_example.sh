python /path/to/case_extract.py --csv data.csv --jsonl meta.jsonl -m 5 -k 3 --output ./result

python /path/to/case_extract.py --csv data.csv --jsonl meta.jsonl -m 5 -k 3 --at-least --output ./result

python /path/to/case_extract.py --csv /path/to/data/dataset_A/checkpoint_XXXXX/itemid_to_sid.csv --jsonl /path/to/data/dataset_A_items.jsonl -m 2 -k 5 --seed 2020 \
    --output /path/to/output



python /path/to/case_extract.py --csv /path/to/data/dataset_A/checkpoint_XXXXX/itemid_to_sid.csv --jsonl /path/to/data/dataset_A_items.jsonl -m 10 -k 5 --seed 2020 \
    --output /path/to/output


python /path/to/case_extract.py --csv /path/to/data/dataset_B/checkpoint_XXXXX/itemid_to_sid.csv --jsonl /path/to/data/dataset_B_items.jsonl -m 10 -k 5 --seed 2020 \
    --output /path/to/output
