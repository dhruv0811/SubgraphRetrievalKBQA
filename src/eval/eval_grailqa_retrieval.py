import json
import os
from collections import defaultdict

def load_original_file(file_path):
    """Load the original JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_retrieved_file(file_path):
    """Load the retrieved JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def calculate_overlap_and_size():
    """Calculate overlap percentages and subgraph sizes."""
    
    # Load the files
    original_data = load_original_file("/home/dhruvgu2/SubgraphRetrievalKBQA/data/GrailQA/data/grailqa_v1.0_dev.json")
    # retrieved_data = load_retrieved_file("/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa/dev_simple.jsonl")
    retrieved_data = load_retrieved_file("/home/dhruvgu2/SubgraphRetrievalKBQA/results/reader_data/grailqa/dev_simple_top20_min0_dense_enhanced_PARTIAL_TYPE.jsonl")
    
    # Create a dictionary to map questions to retrieved data for faster lookup
    retrieved_dict = {item['question']: item for item in retrieved_data}
    
    results = []
    level_stats = defaultdict(lambda: {
        'count': 0,
        'relation_overlap_count': 0,
        'total_original_relations': 0,
        'entity_overlap_count': 0,
        'total_original_entities': 0,
        'original_subgraph_relations': 0,
        'original_subgraph_entities': 0,
        'retrieved_subgraph_relations': 0,
        'retrieved_subgraph_entities': 0
    })
    
    # Process each item in the original data
    for original_item in original_data:
        question = original_item['question']
        level = original_item.get('level', 'unknown')
        
        # Skip if the question is not in the retrieved data
        if question not in retrieved_dict:
            print(f"Question not found in retrieved data: {question}")
            continue
        
        retrieved_item = retrieved_dict[question]
        
        # ---- Relations Analysis ----
        # Extract relations from original data (edge['relation'])
        original_relations = set()
        if 'graph_query' in original_item and 'edges' in original_item['graph_query']:
            for edge in original_item['graph_query']['edges']:
                original_relations.add(edge['relation'])
        
        # Extract relations from retrieved data (middle element of tuples in subgraph)
        retrieved_relations = set()
        if 'subgraph' in retrieved_item and 'tuples' in retrieved_item['subgraph']:
            for tuple_item in retrieved_item['subgraph']['tuples']:
                if len(tuple_item) >= 2:  # Ensure tuple has at least 2 elements
                    retrieved_relations.add(tuple_item[1])
        
        # Calculate relation overlap
        relation_overlap_count = 0
        for relation in original_relations:
            if relation in retrieved_relations:
                relation_overlap_count += 1
        
        if len(original_relations) > 0:
            relation_overlap_percentage = (relation_overlap_count / len(original_relations)) * 100
        else:
            relation_overlap_percentage = 0  # Avoid division by zero
        
        # ---- Entities Analysis ----
        # Extract entities from original data (node['id'] or node['class'])
        original_entities = set()
        if 'graph_query' in original_item and 'nodes' in original_item['graph_query']:
            for node in original_item['graph_query']['nodes']:
                if node['node_type'] == 'entity':
                    original_entities.add(node['class'])
        
        # Extract entities from retrieved data (from 'entities' list and from tuples)
        retrieved_entities = set()
        if 'entities' in retrieved_item:
            retrieved_entities.update(retrieved_item['entities'])
        
        if 'subgraph' in retrieved_item and 'entities' in retrieved_item['subgraph']:
            retrieved_entities.update(retrieved_item['subgraph']['entities'])
        
        # Also get entities from tuples
        if 'subgraph' in retrieved_item and 'tuples' in retrieved_item['subgraph']:
            for tuple_item in retrieved_item['subgraph']['tuples']:
                if len(tuple_item) >= 3:  # Ensure tuple has all elements
                    retrieved_entities.add(tuple_item[0])  # First element
                    retrieved_entities.add(tuple_item[2])  # Third element
        
        # Calculate entity overlap
        entity_overlap_count = 0
        for entity in original_entities:
            if entity in retrieved_entities:
                entity_overlap_count += 1
        
        if len(original_entities) > 0:
            entity_overlap_percentage = (entity_overlap_count / len(original_entities)) * 100
        else:
            entity_overlap_percentage = 0  # Avoid division by zero
        
        # ---- Subgraph Size Analysis ----
        # Original subgraph size
        original_subgraph_relations_count = len(original_relations)
        original_subgraph_entities_count = len(original_entities)
        
        # Retrieved subgraph size
        retrieved_subgraph_relations_count = len(retrieved_relations)
        retrieved_subgraph_entities_count = len(retrieved_entities)
        
        # Calculate the ratio of retrieved to original subgraph size
        relation_size_ratio = 0
        if original_subgraph_relations_count > 0:
            relation_size_ratio = retrieved_subgraph_relations_count / original_subgraph_relations_count
        
        entity_size_ratio = 0
        if original_subgraph_entities_count > 0:
            entity_size_ratio = retrieved_subgraph_entities_count / original_subgraph_entities_count
        
        # Add to result
        result = {
            'question': question,
            'level': level,
            'original_relations': list(original_relations),
            'retrieved_relations': list(retrieved_relations),
            'relation_overlap_count': relation_overlap_count,
            'total_original_relations': len(original_relations),
            'relation_overlap_percentage': relation_overlap_percentage,
            'original_entities': list(original_entities),
            'retrieved_entities': list(retrieved_entities),
            'entity_overlap_count': entity_overlap_count,
            'total_original_entities': len(original_entities),
            'entity_overlap_percentage': entity_overlap_percentage,
            'original_subgraph_size': {
                'relations': original_subgraph_relations_count,
                'entities': original_subgraph_entities_count
            },
            'retrieved_subgraph_size': {
                'relations': retrieved_subgraph_relations_count,
                'entities': retrieved_subgraph_entities_count
            },
            'subgraph_size_ratio': {
                'relations': relation_size_ratio,
                'entities': entity_size_ratio
            }
        }
        
        results.append(result)
        
        # Update level statistics
        level_stats[level]['count'] += 1
        level_stats[level]['relation_overlap_count'] += relation_overlap_count
        level_stats[level]['total_original_relations'] += len(original_relations)
        level_stats[level]['entity_overlap_count'] += entity_overlap_count
        level_stats[level]['total_original_entities'] += len(original_entities)
        level_stats[level]['original_subgraph_relations'] += original_subgraph_relations_count
        level_stats[level]['original_subgraph_entities'] += original_subgraph_entities_count
        level_stats[level]['retrieved_subgraph_relations'] += retrieved_subgraph_relations_count
        level_stats[level]['retrieved_subgraph_entities'] += retrieved_subgraph_entities_count
    
    # Calculate overall statistics
    total_relation_overlap = sum(result['relation_overlap_count'] for result in results)
    total_original_relations = sum(result['total_original_relations'] for result in results)
    total_entity_overlap = sum(result['entity_overlap_count'] for result in results)
    total_original_entities = sum(result['total_original_entities'] for result in results)
    
    total_original_subgraph_relations = sum(result['original_subgraph_size']['relations'] for result in results)
    total_original_subgraph_entities = sum(result['original_subgraph_size']['entities'] for result in results)
    total_retrieved_subgraph_relations = sum(result['retrieved_subgraph_size']['relations'] for result in results)
    total_retrieved_subgraph_entities = sum(result['retrieved_subgraph_size']['entities'] for result in results)
    
    # Calculate averages and percentages for each level
    level_summaries = {}
    for level, stats in level_stats.items():
        # Calculate relation and entity overlap percentages
        relation_percentage = 0
        if stats['total_original_relations'] > 0:
            relation_percentage = (stats['relation_overlap_count'] / stats['total_original_relations']) * 100
        
        entity_percentage = 0
        if stats['total_original_entities'] > 0:
            entity_percentage = (stats['entity_overlap_count'] / stats['total_original_entities']) * 100
        
        # Calculate average subgraph sizes
        avg_original_relations = 0
        avg_retrieved_relations = 0
        avg_original_entities = 0
        avg_retrieved_entities = 0
        
        if stats['count'] > 0:
            avg_original_relations = stats['original_subgraph_relations'] / stats['count']
            avg_retrieved_relations = stats['retrieved_subgraph_relations'] / stats['count']
            avg_original_entities = stats['original_subgraph_entities'] / stats['count']
            avg_retrieved_entities = stats['retrieved_subgraph_entities'] / stats['count']
        
        # Calculate size ratios
        relation_size_ratio = 0
        entity_size_ratio = 0
        
        if stats['original_subgraph_relations'] > 0:
            relation_size_ratio = stats['retrieved_subgraph_relations'] / stats['original_subgraph_relations']
        
        if stats['original_subgraph_entities'] > 0:
            entity_size_ratio = stats['retrieved_subgraph_entities'] / stats['original_subgraph_entities']
        
        level_summaries[level] = {
            'count': stats['count'],
            'overlap': {
                'relation_percentage': relation_percentage,
                'entity_percentage': entity_percentage
            },
            'subgraph_size': {
                'avg_original_relations': avg_original_relations,
                'avg_retrieved_relations': avg_retrieved_relations,
                'avg_original_entities': avg_original_entities,
                'avg_retrieved_entities': avg_retrieved_entities,
                'relation_size_ratio': relation_size_ratio,
                'entity_size_ratio': entity_size_ratio
            }
        }
    
    # Calculate overall percentages and averages
    overall_relation_percentage = 0
    if total_original_relations > 0:
        overall_relation_percentage = (total_relation_overlap / total_original_relations) * 100
    
    overall_entity_percentage = 0
    if total_original_entities > 0:
        overall_entity_percentage = (total_entity_overlap / total_original_entities) * 100
    
    avg_original_relations = 0
    avg_retrieved_relations = 0
    avg_original_entities = 0
    avg_retrieved_entities = 0
    
    if len(results) > 0:
        avg_original_relations = total_original_subgraph_relations / len(results)
        avg_retrieved_relations = total_retrieved_subgraph_relations / len(results)
        avg_original_entities = total_original_subgraph_entities / len(results)
        avg_retrieved_entities = total_retrieved_subgraph_entities / len(results)
    
    relation_size_ratio = 0
    entity_size_ratio = 0
    
    if total_original_subgraph_relations > 0:
        relation_size_ratio = total_retrieved_subgraph_relations / total_original_subgraph_relations
    
    if total_original_subgraph_entities > 0:
        entity_size_ratio = total_retrieved_subgraph_entities / total_original_subgraph_entities
    
    summary = {
        'total_questions': len(results),
        'overlap': {
            'relation': {
                'total_overlap_count': total_relation_overlap,
                'total_original_count': total_original_relations,
                'overall_percentage': overall_relation_percentage
            },
            'entity': {
                'total_overlap_count': total_entity_overlap,
                'total_original_count': total_original_entities,
                'overall_percentage': overall_entity_percentage
            }
        },
        'subgraph_size': {
            'original': {
                'total_relations': total_original_subgraph_relations,
                'total_entities': total_original_subgraph_entities,
                'avg_relations': avg_original_relations,
                'avg_entities': avg_original_entities
            },
            'retrieved': {
                'total_relations': total_retrieved_subgraph_relations,
                'total_entities': total_retrieved_subgraph_entities,
                'avg_relations': avg_retrieved_relations,
                'avg_entities': avg_retrieved_entities
            },
            'ratio': {
                'relations': relation_size_ratio,
                'entities': entity_size_ratio
            }
        },
        'level_breakdown': level_summaries
    }
    
    return results, summary

def main():
    # Run the analysis
    results, summary = calculate_overlap_and_size()
    
    # Print summary
    print("\nOverall Summary:")
    print(f"Total questions analyzed: {summary['total_questions']}")
    
    print("\nRelation Overlap:")
    print(f"Total original relations: {summary['overlap']['relation']['total_original_count']}")
    print(f"Total relations found in retrieved data: {summary['overlap']['relation']['total_overlap_count']}")
    print(f"Overall relation overlap percentage: {summary['overlap']['relation']['overall_percentage']:.2f}%")
    
    print("\nEntity Overlap:")
    print(f"Total original entities: {summary['overlap']['entity']['total_original_count']}")
    print(f"Total entities found in retrieved data: {summary['overlap']['entity']['total_overlap_count']}")
    print(f"Overall entity overlap percentage: {summary['overlap']['entity']['overall_percentage']:.2f}%")
    
    print("\nSubgraph Size Comparison:")
    print("Original Subgraphs:")
    print(f"  Average relations per subgraph: {summary['subgraph_size']['original']['avg_relations']:.2f}")
    print(f"  Average entities per subgraph: {summary['subgraph_size']['original']['avg_entities']:.2f}")
    print("Retrieved Subgraphs:")
    print(f"  Average relations per subgraph: {summary['subgraph_size']['retrieved']['avg_relations']:.2f}")
    print(f"  Average entities per subgraph: {summary['subgraph_size']['retrieved']['avg_entities']:.2f}")
    print("Size Ratios (Retrieved/Original):")
    print(f"  Relations ratio: {summary['subgraph_size']['ratio']['relations']:.2f}x")
    print(f"  Entities ratio: {summary['subgraph_size']['ratio']['entities']:.2f}x")
    
    print("\nBreakdown by Question Level:")
    for level, stats in summary['level_breakdown'].items():
        print(f"\nLevel: {level}")
        print(f"Number of questions: {stats['count']}")
        
        print("Overlap Percentages:")
        print(f"  Relation overlap: {stats['overlap']['relation_percentage']:.2f}%")
        print(f"  Entity overlap: {stats['overlap']['entity_percentage']:.2f}%")
        
        print("Subgraph Sizes:")
        print(f"  Original - Avg relations: {stats['subgraph_size']['avg_original_relations']:.2f}, Avg entities: {stats['subgraph_size']['avg_original_entities']:.2f}")
        print(f"  Retrieved - Avg relations: {stats['subgraph_size']['avg_retrieved_relations']:.2f}, Avg entities: {stats['subgraph_size']['avg_retrieved_entities']:.2f}")
        print(f"  Size Ratios - Relations: {stats['subgraph_size']['relation_size_ratio']:.2f}x, Entities: {stats['subgraph_size']['entity_size_ratio']:.2f}x")
    
    # Save the results to a file
    output = {
        'results': results,
        'summary': summary
    }
    
    with open('/home/dhruvgu2/SubgraphRetrievalKBQA/src/eval/enhanced_partial_type_top20_min0_subgraph_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to comprehensive_analysis_grailqa_dev.json")

if __name__ == "__main__":
    main()