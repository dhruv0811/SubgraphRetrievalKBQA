'''构建 Local KB 中 relation 集合以及 entity 集合 '''
import os
import json
from tqdm import tqdm
from utils import load_jsonl
from loguru import logger
from config import cfg

def run():
    load_data_path = cfg.retrieve_subgraph["dump_data_folder"]

    # train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple.json"))
    # test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))
    dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple_top20_min0.json"))

    entity_set = set()
    relation_set = set()
    out_entity_set_filename = os.path.join(load_data_path, 'entities.txt')
    out_relation_set_filename = os.path.join(load_data_path, 'relations.txt')

    for dataset in [dev_dataset]: # train_dataset, test_dataset, 
        for json_obj in tqdm(dataset):
            answers = {ans_json_obj#["kb_id"]
                    for ans_json_obj in json_obj["answers"]}
            subgraph_entities = set(json_obj["subgraph"]["entities"]) | set(json_obj["topic_entities"])
            subgraph_relations = {r for h, r, t in json_obj["subgraph"]["tuples"]}
            entity_set = entity_set | answers | subgraph_entities
            relation_set = relation_set | subgraph_relations

    def dump_list_to_txt(mylist, outname):
        with open(outname, 'w') as f:
            for item in mylist:
                print(item, file=f)

    entity_set = sorted(entity_set)
    relation_set = sorted(relation_set)

    dump_list_to_txt(entity_set, out_entity_set_filename)
    dump_list_to_txt(relation_set, out_relation_set_filename)
