import os
import json
import argparse
from tqdm import tqdm

from config import cfg
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase

def load_webqsp():
    load_data_path = cfg.preprocessing["step0"]["load_data_path"]
    dump_data_path = cfg.preprocessing["step0"]["dump_data_path"]
    folder_path = cfg.preprocessing["step0"]["dump_data_folder"]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(load_data_path, "r") as f:
        train_dataset = json.loads(f.read())
        data_list = []
        for json_obj in train_dataset["Questions"]:
            question = json_obj["ProcessedQuestion"]
            for parse in json_obj["Parses"]:
                topic_entities = [parse["TopicEntityMid"]]
                answers = []
                for answer_json_obj in parse["Answers"]:
                    if answer_json_obj["AnswerType"] == "Entity":
                        answers.append(answer_json_obj["AnswerArgument"])
                if len(answers) == 0:
                    continue
                data_list.append({
                    "question": question,
                    "topic_entities": topic_entities,
                    "answers": answers,
                })
        with open(dump_data_path, "w") as f:
            for json_obj in data_list:
                f.write(json.dumps(json_obj) + "\n")

def clean_string(date_string):
    # Split the string at "^^" and take only the first part
    if '^^http://' in date_string:
        # Already formatted as a typed literal
        parts = date_string.replace('#', '^^').split('^^')
        return f'"{parts[0]}"^^<{parts[1]}#{parts[2]}>'
        
    # Regular entity case - just use the namespace prefix
    return date_string

def load_grailqa():
    load_data_path = cfg.preprocessing["step0"]["load_data_path"]
    dump_data_path = cfg.preprocessing["step0"]["dump_data_path"]
    folder_path = cfg.preprocessing["step0"]["dump_data_folder"]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(load_data_path, "r") as f:
        train_dataset = json.loads(f.read())
        data_list = []
        for json_obj in tqdm(train_dataset, desc="Load Grailqa"):
            # Get Question
            question = json_obj["question"]

            # Get Topic Entities
            topic_classes = []
            for node_json in json_obj['graph_query']['nodes']:
                if node_json['node_type'] == 'literal' or node_json['node_type'] == 'entity':
                    topic_classes.append(node_json['class'])

            # Get Answers
            answers = []
            for node_json in json_obj['graph_query']['nodes']:
                if node_json['question_node'] == 1:
                    answers.append(node_json['class'])

            data_list.append({
                "question": question,
                "topic_entities": list(set(topic_classes)),
                "answers": list(set(answers)),
            })

        with open(dump_data_path, "w") as f:
            for json_obj in data_list:
                f.write(json.dumps(json_obj) + "\n")