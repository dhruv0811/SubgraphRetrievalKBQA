import argparse
import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import sys

# Add parent directory to import path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge_graph.knowledge_graph_ontology import KnowledgeBaseOntology

# Parse arguments
parser = argparse.ArgumentParser(description='Analyze relation score distribution')
parser.add_argument('--model_path', type=str, default='/data/user_data/dhruvgu2/model_ckpt/SimBERT', help='Path to the BERT model checkpoint')
parser.add_argument('--data_path', type=str, default='/home/dhruvgu2/SubgraphRetrievalKBQA/data/GrailQA/data/grailqa_v1.0_dev.json', help='Path to the JSON file with questions and gold relations')
parser.add_argument('--output_dir', type=str, default='/home/dhruvgu2/SubgraphRetrievalKBQA/src/retrieve_subgraph/analysis_plots/', help='Directory to save plots')
parser.add_argument('--theta', type=float, default=0.07, help='Scaling factor for similarity scores')
parser.add_argument('--top_k', type=int, default=10, help='Number of paths to consider')
parser.add_argument('--max_hop', type=int, default=2, help='Maximum hop distance')
args = parser.parse_args()

# Constants
END_REL = "END OF HOP"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Load model and tokenizer
print(f"Using device: {device}")
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path)
model = model.to(device)
model.eval()

# Load knowledge graph
print(f"Loading knowledge graph...")
kg = KnowledgeBaseOntology()

# Load data with questions and gold relations
print(f"Loading data from {args.data_path}...")
# Load data - supports both JSON and JSONL formats
print(f"Loading data from {args.data_path}...")
with open(args.data_path, 'r') as f:
    # Try to load as JSON first
    try:
        data = json.load(f)
    except json.JSONDecodeError:
        # If that fails, try to load as JSONL
        f.seek(0)
        data = []
        for line in f:
            if line.strip():
                data.append(json.loads(line))

# Function to get embeddings (same as in the original code)
@torch.no_grad()
def get_texts_embeddings(texts):
    inputs = tokenizer(texts, padding=True,
                       truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, output_hidden_states=True,
                       return_dict=True).pooler_output
    return embeddings

# Function to get candidate relations (same as in the original code)
def path_to_candidate_relations(topic_entity, path):
    """Get candidate relations from a topic entity and path.
    
    Args:
        topic_entity (str): The topic entity ID (e.g., "m.0f8l9c")
        path (list): A list of relations forming a path
        
    Returns:
        list: List of candidate relations
    """
    try:
        # Make sure topic_entity is a string
        if not isinstance(topic_entity, str):
            raise TypeError(f"topic_entity must be a string, got {type(topic_entity)}: {topic_entity}")
            
        # Make sure path is a list
        if not isinstance(path, list):
            raise TypeError(f"path must be a list, got {type(path)}: {path}")
            
        new_relations = kg.deduce_leaves_relation_by_path(topic_entity, path)
        # filter relation
        candidate_relations = [r for r in new_relations if r.split(".")[0] not in ["kg", "common"]]
        return list(candidate_relations)
    except Exception as e:
        print(f"Error getting candidate relations for {topic_entity}, {path}: {str(e)}")
        return []

# Function to score relations (adapted from the original code)
@torch.no_grad()
def score_relations(question, path, relations, theta=0.07):
    if not relations:
        return {}
        
    query = '#'.join([question] + path)
    q_emb = get_texts_embeddings([query]).unsqueeze(1)  # [1, 1, D]
    rel_emb = get_texts_embeddings(relations).unsqueeze(0)  # [1, L, D]
    sim_scores = torch.cosine_similarity(q_emb, rel_emb, dim=2) / theta  # [1, L]
    
    scores = {}
    for i, relation in enumerate(relations):
        scores[relation] = float(sim_scores[0, i])
    
    return scores

# Collect scores for gold and non-gold relations
gold_scores = []
non_gold_scores = []
relation_to_scores = defaultdict(list)  # To track scores by relation
question_to_scores = defaultdict(lambda: {'gold': [], 'non_gold': []})  # To track scores by question

print("Computing similarity scores...")
for item in tqdm(data):
    question = item.get('question', '')

    topic_entity = []
    for node in item['graph_query']['nodes']:
        if node['node_type'] in ['literal', 'entity']:
            topic_entity.append(node['class'])

    gold_relations = []
    for edge in item['graph_query']['edges']:
        gold_relations.append(edge['relation'])
    
    # Also check for alternate formats from your dataset structure
    if not topic_entity and 'entities' in item:
        # If topic_entity not provided directly but entities are available
        entities = item.get('entities', [])
        if entities:
            # Use the first entity as topic_entity
            if isinstance(entities[0], dict) and 'kb_id' in entities[0]:
                topic_entity = entities[0]['kb_id']
            elif isinstance(entities[0], str):
                topic_entity = entities[0]
    
    # Similarly, check for gold relations in alternate formats
    if not gold_relations and 'answers' in item:
        # Try to extract gold relations from the answers and subgraph
        answers = item.get('answers', [])
        subgraph = item.get('subgraph', {})
        tuples = subgraph.get('tuples', [])
        
        # If we have answer entities and tuples, extract relations leading to answers
        if answers and tuples:
            answer_ids = set()
            for ans in answers:
                if isinstance(ans, dict) and 'kb_id' in ans:
                    answer_ids.add(ans['kb_id'])
                elif isinstance(ans, str):
                    answer_ids.add(ans)
            
            # Extract relations from tuples that lead to answers
            for t in tuples:
                if isinstance(t, list) or isinstance(t, tuple):
                    # If tuple format is [h, r, t]
                    if len(t) == 3 and t[2] in answer_ids:
                        gold_relations.append(t[1])
                elif isinstance(t, dict):
                    # If tuple has dict format
                    head = t.get('head', {})
                    rel = t.get('relation', {})
                    tail = t.get('tail', {})
                    
                    head_id = head.get('kb_id', '') if isinstance(head, dict) else head
                    rel_id = rel.get('rel_id', '') if isinstance(rel, dict) else rel
                    tail_id = tail.get('kb_id', '') if isinstance(tail, dict) else tail
                    
                    if tail_id in answer_ids:
                        gold_relations.append(rel_id)
    
    if not question or not topic_entity:
        print(f"Warning: Missing question or topic entity in item: {item}")
        continue
        
    # Deduplicate gold relations
    gold_relations = list(set(gold_relations))
    
    # We'll consider paths of various lengths
    # First, make sure topic_entity is a string, not a list
    if isinstance(topic_entity, list):
        topic_entity = topic_entity[0]  # Take the first entity if it's a list
        
    # Get candidate relations for an empty path
    initial_relations = path_to_candidate_relations(topic_entity, [])
    
    paths_to_try = [
        [],  # Empty path (0-hop)
    ]
    
    # Add some 1-hop paths (using top 5 relations for efficiency)
    if initial_relations:
        for r in initial_relations[:5]:
            paths_to_try.append([r])
    
    for path in paths_to_try:
        # Get candidate relations for this path
        candidate_relations = path_to_candidate_relations(topic_entity, path)
        
        if not candidate_relations:
            continue
            
        # Compute similarity scores
        relation_scores = score_relations(question, path, candidate_relations, theta=args.theta)
        
        # Collect scores
        for relation, score in relation_scores.items():
            is_gold = relation in gold_relations
            
            if is_gold:
                gold_scores.append(score)
                question_to_scores[question]['gold'].append((relation, score))
            else:
                non_gold_scores.append(score)
                question_to_scores[question]['non_gold'].append((relation, score))
            
            relation_to_scores[relation].append((question, score, is_gold))

# Ensure we have basic NumPy arrays without pandas dependencies
print(f"Processing collected scores. Gold scores: {len(gold_scores)}, Non-gold scores: {len(non_gold_scores)}")
print(f"Type of gold_scores: {type(gold_scores)}")

# Convert to simple Python lists first, then to NumPy arrays
gold_scores_list = [float(score) for score in gold_scores]
non_gold_scores_list = [float(score) for score in non_gold_scores]

# Now convert to NumPy arrays
gold_scores_array = np.array(gold_scores_list, dtype=np.float64)
non_gold_scores_array = np.array(non_gold_scores_list, dtype=np.float64)

print(f"After conversion - Gold scores array shape: {gold_scores_array.shape}, Non-gold scores array shape: {non_gold_scores_array.shape}")

# Plot histograms using matplotlib directly (avoid seaborn)
plt.figure(figsize=(10, 6))
if len(gold_scores_array) > 0:
    plt.hist(gold_scores_array, bins=30, alpha=0.7, label='Gold Relations', color='gold', density=True)
if len(non_gold_scores_array) > 0:
    plt.hist(non_gold_scores_array, bins=30, alpha=0.7, label='Non-Gold Relations', color='blue', density=True)

plt.title('Distribution of Similarity Scores')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(args.output_dir, 'score_distribution.png'))
plt.close()

# Plot CDF using matplotlib directly
plt.figure(figsize=(10, 6))
if len(gold_scores_array) > 0:
    sorted_data = np.sort(gold_scores_array)
    yvals = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, yvals, label='Gold Relations', color='gold')
if len(non_gold_scores_array) > 0:
    sorted_data = np.sort(non_gold_scores_array)
    yvals = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, yvals, label='Non-Gold Relations', color='blue')
plt.title('Cumulative Distribution of Similarity Scores')
plt.xlabel('Score')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(args.output_dir, 'score_cdf.png'))
plt.close()

# Print statistics using the NumPy arrays
if len(gold_scores_array) > 0:
    print(f"Gold Relations: Count={len(gold_scores_array)}, Mean={gold_scores_array.mean():.4f}, Median={np.median(gold_scores_array):.4f}, Std={gold_scores_array.std():.4f}")
else:
    print("No gold relation scores collected")

if len(non_gold_scores_array) > 0:
    print(f"Non-Gold Relations: Count={len(non_gold_scores_array)}, Mean={non_gold_scores_array.mean():.4f}, Median={np.median(non_gold_scores_array):.4f}, Std={non_gold_scores_array.std():.4f}")
else:
    print("No non-gold relation scores collected")

# Plot top relations by gold mean score
relation_stats = []
for relation, scores in relation_to_scores.items():
    gold_instances = [(q, s) for q, s, is_gold in scores if is_gold]
    non_gold_instances = [(q, s) for q, s, is_gold in scores if not is_gold]
    
    if gold_instances:
        gold_mean = np.mean([s for _, s in gold_instances])
        gold_count = len(gold_instances)
        relation_stats.append((relation, gold_mean, gold_count))

if relation_stats:
    # Sort by gold mean score
    relation_stats.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(relation_stats))
    relations = [r.split('.')[-1][:15] + '...' if len(r.split('.')[-1]) > 15 else r.split('.')[-1] 
                for r, _, _ in relation_stats[:top_n]]
    gold_means = [g for _, g, _ in relation_stats[:top_n]]
    gold_counts = [c for _, _, c in relation_stats[:top_n]]
    
    plt.bar(range(len(relations)), gold_means, color='blue', alpha=0.7)
    plt.xlabel('Relations')
    plt.ylabel('Mean Gold Score')
    plt.title('Top Relations by Gold Mean Score')
    plt.xticks(range(len(relations)), relations, rotation=45, ha='right')
    
    # Add count annotations
    for i, count in enumerate(gold_counts):
        plt.annotate(str(count), xy=(i, gold_means[i]), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top_gold_relations.png'))
    plt.close()

# Save detailed stats to CSV
with open(os.path.join(args.output_dir, 'relation_stats.csv'), 'w') as f:
    f.write('relation,gold_mean,gold_count\n')
    for relation, gold_mean, gold_count in relation_stats:
        f.write(f'"{relation}",{gold_mean:.4f},{gold_count}\n')

# Save scores to file for further analysis
with open(os.path.join(args.output_dir, 'scores.json'), 'w') as f:
    json.dump({
        'gold_scores': gold_scores_list,  # Use the list directly
        'non_gold_scores': non_gold_scores_list,  # Use the list directly
    }, f)

print("Analysis complete. Plots and statistics saved to", args.output_dir)