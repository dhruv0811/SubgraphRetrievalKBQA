import json
from knowledge_graph_ontology import KnowledgeBaseOntology

def enhance_subgraph(entry, ontology):
    """
    Enhance the subgraph by adding all possible relations between existing entities.
    """
    # Make a copy of the entry to avoid modifying the original
    enhanced_entry = entry.copy()
    
    # Extract the existing subgraph
    subgraph = enhanced_entry.get("subgraph", {})
    entities = subgraph.get("entities", [])
    existing_tuples = set(tuple(t) for t in subgraph.get("tuples", []))
    
    # For each pair of entities, find all direct relations
    new_tuples = set()
    for i, src in enumerate(entities):
        for j, tgt in enumerate(entities):
            if i != j:  # Skip self-relations
                # Find all direct relations from src to tgt
                direct_relations = ontology.search_one_hop_relaiotn(src, tgt)
                
                # Add the new relations to the subgraph
                for relation in direct_relations:
                    new_tuple = (src, relation, tgt)
                    new_tuples.add(new_tuple)
    
    # Combine existing and new tuples
    all_tuples = existing_tuples.union(new_tuples)
    
    # Update the subgraph with the enhanced tuples
    enhanced_entry["subgraph"] = {
        "tuples": [list(t) for t in all_tuples],
        "entities": entities
    }
    
    return enhanced_entry

def main():    
    # Path to input and output JSONL files
    INPUT_JSONL = "/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa/dev_simple_top20_min0.jsonl"
    OUTPUT_JSONL = "/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa/dev_simple_top20_min0_dense_enhanced_PARTIAL_TYPE.jsonl"
    
    # Initialize the ontology
    ontology = KnowledgeBaseOntology()
    
    # Process the input JSONL file
    with open(INPUT_JSONL, 'r') as f_in, open(OUTPUT_JSONL, 'w') as f_out:
        for line in f_in:
            entry = json.loads(line.strip())
            if entry.get('topic_entities') and any("type." in entity for entity in entry.get('topic_entities')):
                enhanced_entry = enhance_subgraph(entry, ontology)
                f_out.write(json.dumps(enhanced_entry) + '\n')
            else:
                print(entry.get('question'))

if __name__ == "__main__":
    main()