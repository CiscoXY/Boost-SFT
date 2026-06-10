python /mnt/graduation_paper/sid/sid效果评测/case_extract.py --csv data.csv --jsonl meta.jsonl -m 5 -k 3 --output ./result

python /mnt/graduation_paper/sid/sid效果评测/case_extract.py --csv data.csv --jsonl meta.jsonl -m 5 -k 3 --at-least --output ./result

python /mnt/graduation_paper/sid/sid效果评测/case_extract.py --csv /mnt/data/sid/Beauty/checkpoint_120000/itemid_to_sid.csv --jsonl /mnt/data/Beauty_items.jsonl -m 2 -k 5 --seed 2020 \
    --output /mnt/graduation_paper/sid/sid效果评测/抽样结果/2个item的



python /mnt/graduation_paper/sid/sid效果评测/case_extract.py --csv /mnt/data/sid/Beauty/checkpoint_120000/itemid_to_sid.csv --jsonl /mnt/data/Beauty_items.jsonl -m 10 -k 5 --seed 2020 \
    --output /mnt/graduation_paper/sid/sid效果评测/抽样结果/10个item的


python /mnt/graduation_paper/sid/sid效果评测/case_extract.py --csv /mnt/data/sid/Sports_Outdoors/checkpoint_120000/itemid_to_sid.csv --jsonl /mnt/data/Sports_Outdoors_items.jsonl -m 10 -k 5 --seed 2020 \
    --output /mnt/graduation_paper/sid/sid效果评测/抽样结果/Sports_Outdoors/10个item的