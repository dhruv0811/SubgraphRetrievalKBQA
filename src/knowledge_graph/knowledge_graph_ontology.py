from typing import List, Tuple, Dict, Set
from pathlib import Path
import json
from collections import defaultdict

class KnowledgeBaseOntology:
    def __init__(self, ontology_file: str = "/home/dhruvgu2/SubgraphRetrievalKBQA/data/GrailQA/ontology/fb_roles_with_reverse.txt", max_hop: int = 2) -> None:
        # Maximum hop distance to consider in path searches
        self.max_hop = max_hop
        
        # Initialize efficient data structures for querying
        self.subject_to_rel_obj = defaultdict(list)  # subject -> [(relation, object), ...]
        self.object_to_subj_rel = defaultdict(list)  # object -> [(subject, relation), ...]
        self.relation_to_subj_obj = defaultdict(list)  # relation -> [(subject, object), ...]
        
        # Additional indexes for faster access
        self.subj_rel_to_obj = {}  # (subject, relation) -> [objects]
        self.obj_rel_to_subj = {}  # (object, relation) -> [subjects]
        
        # Load ontology if provided
        if ontology_file:
            self.load_ontology(ontology_file)
    
    def load_ontology(self, ontology_file: str) -> None:
        """Load ontology from a file."""
        with open(ontology_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    subject = parts[0]
                    relation = parts[1]
                    object_ = parts[2]
                    
                    # Update subject_to_rel_obj
                    self.subject_to_rel_obj[subject].append((relation, object_))
                    
                    # Update object_to_subj_rel
                    self.object_to_subj_rel[object_].append((subject, relation))
                    
                    # Update relation_to_subj_obj
                    self.relation_to_subj_obj[relation].append((subject, object_))
                    
                    # Update additional indexes
                    key = (subject, relation)
                    if key not in self.subj_rel_to_obj:
                        self.subj_rel_to_obj[key] = []
                    self.subj_rel_to_obj[key].append(object_)
                    
                    key = (object_, relation)
                    if key not in self.obj_rel_to_subj:
                        self.obj_rel_to_subj[key] = []
                    self.obj_rel_to_subj[key].append(subject)
    
    def set_max_hop(self, max_hop: int) -> None:
        """Set the maximum hop distance for path searches."""
        self.max_hop = max_hop
    
    def get_relation(self, entity: str, limit: int = 100) -> List[str]:
        """Get all relations for a given entity."""
        result = set()
        
        # Get relations where entity is the subject
        for relation, _ in self.subject_to_rel_obj[entity]:
            result.add(relation)
            if len(result) >= limit:
                break
        
        return list(result)
    
    def get_tail(self, src: str, relation: str) -> List[str]:
        """Get all tail entities for a given head entity and relation."""
        result = []
        
        key = (src, relation)
        if key in self.subj_rel_to_obj:
            result = self.subj_rel_to_obj[key]
        else:
            # Fallback to slower lookup if the optimized index doesn't have it
            for rel, obj in self.subject_to_rel_obj[src]:
                if rel == relation:
                    result.append(obj)
        
        return result
    
    def get_single_tail_relation_triplet(self, src: str) -> List[str]:
        """Get relations where a given entity has only one tail."""
        relation_count = {}
        
        for relation, _ in self.subject_to_rel_obj[src]:
            if relation not in relation_count:
                relation_count[relation] = 0
            relation_count[relation] += 1
            
        return [k for k, v in relation_count.items() if v == 1]
    
    def get_all_path(self, src_: str, tgt_: str) -> List[List[Tuple[str, str, str]]]:
        """Get all paths between two entities up to max_hop."""
        src = src_
        tgt = tgt_
        one_hop = []
        two_hop = []
        third_hop = []
        
        # Check for direct relations (1-hop)
        for relation, obj in self.subject_to_rel_obj[src]:
            if obj == tgt:
                one_hop.append([(src, relation, tgt)])
        
        if self.max_hop >= 2:
            # Check for two-hop paths
            for relation1, intermediate in self.subject_to_rel_obj[src]:
                for relation2, obj in self.subject_to_rel_obj[intermediate]:
                    if obj == tgt:
                        two_hop.append([(src, relation1, intermediate), (intermediate, relation2, tgt)])
        
        if self.max_hop >= 3:
            # Process for third-hop paths
            for relation1, intermediate1 in self.subject_to_rel_obj[src]:
                for relation2, intermediate2 in self.subject_to_rel_obj[intermediate1]:
                    for relation3, obj in self.subject_to_rel_obj[intermediate2]:
                        if obj == tgt:
                            third_hop.append([
                                (src, relation1, intermediate1), 
                                (intermediate1, relation2, intermediate2),
                                (intermediate2, relation3, tgt)
                            ])
        
        return one_hop + two_hop + third_hop
    
    def get_shortest_path_limit(self, src_: str, tgt_: str) -> List[List[Tuple[str, str, str]]]:
        """Get the shortest path between two entities up to max_hop."""
        src = src_
        tgt = tgt_
        
        # Check for direct relations (1-hop)
        one_hop = []
        for relation, obj in self.subject_to_rel_obj[src]:
            if obj == tgt and relation != "type.object.type":
                one_hop.append([(src, relation, tgt)])
        
        if one_hop:
            return one_hop
        
        if self.max_hop >= 2:
            # If no direct relations, check for two-hop paths
            single_hop_relations = self.get_single_tail_relation_triplet(src)
            two_hop = []
            
            for r0 in single_hop_relations:
                # Get intermediate entities
                for intermediate in self.get_tail(src, r0):
                    # Check if intermediate connects to target
                    for relation2, obj in self.subject_to_rel_obj[intermediate]:
                        if obj == tgt and relation2 != "type.object.type":
                            two_hop.append([(src, r0, intermediate), (intermediate, relation2, tgt)])
            
            if two_hop:
                return two_hop
        
        if self.max_hop >= 3:
            # If no 2-hop paths, check for 3-hop paths
            three_hop = []
            
            # Get all 1-hop neighbors from source
            for relation1, intermediate1 in self.subject_to_rel_obj[src]:
                if relation1 == "type.object.type":
                    continue
                
                # Get all 1-hop neighbors from intermediate1
                for relation2, intermediate2 in self.subject_to_rel_obj[intermediate1]:
                    if relation2 == "type.object.type":
                        continue
                    
                    # Check if intermediate2 connects to target
                    for relation3, obj in self.subject_to_rel_obj[intermediate2]:
                        if obj == tgt and relation3 != "type.object.type":
                            three_hop.append([
                                (src, relation1, intermediate1),
                                (intermediate1, relation2, intermediate2),
                                (intermediate2, relation3, tgt)
                            ])
            
            return three_hop
        
        return []
    
    def search_one_hop_relaiotn(self, src: str, tgt: str) -> List[str]:
        """Get direct relations between two entities."""
        result = []
        
        for relation, obj in self.subject_to_rel_obj[src]:
            if obj == tgt:
                result.append(relation)
        
        return result
    
    def search_two_hop_relation(self, src: str, tgt: str) -> List[Tuple[str, str]]:
        """Get two-hop relations between two entities."""
        result = []
        
        for relation1, intermediate in self.subject_to_rel_obj[src]:
            for relation2, obj in self.subject_to_rel_obj[intermediate]:
                if obj == tgt:
                    result.append((relation1, relation2))
        
        return result
    
    def search_three_hop_relation(self, src: str, tgt: str) -> List[Tuple[str, str, str]]:
        """Get three-hop relations between two entities."""
        result = []
        
        for relation1, intermediate1 in self.subject_to_rel_obj[src]:
            for relation2, intermediate2 in self.subject_to_rel_obj[intermediate1]:
                for relation3, obj in self.subject_to_rel_obj[intermediate2]:
                    if obj == tgt:
                        result.append((relation1, relation2, relation3))
        
        return result
    
    def deduce_subgraph_by_path_one(self, src: str, rels: List[str]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Get subgraph by following a 1-hop path."""
        nodes = [src]
        triples = []
        
        for relation, obj in self.subject_to_rel_obj[src]:
            if relation == rels[0]:
                nodes.append(obj)
                triples.append((src, relation, obj))
        
        return nodes, triples
    
    def deduce_subgraph_by_path_two(self, src: str, rels: List[str]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Get subgraph by following a 2-hop path."""
        nodes = [src]
        triples = []
        
        for relation1, intermediate in self.subject_to_rel_obj[src]:
            if relation1 == rels[0]:
                nodes.append(intermediate)
                triples.append((src, relation1, intermediate))
                
                for relation2, obj in self.subject_to_rel_obj[intermediate]:
                    if relation2 == rels[1]:
                        nodes.append(obj)
                        triples.append((intermediate, relation2, obj))
        
        return list(set(nodes)), list(set(triples))
    
    def deduce_subgraph_by_path_three(self, src: str, rels: List[str]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Get subgraph by following a 3-hop path."""
        nodes = [src]
        triples = []
        
        for relation1, intermediate1 in self.subject_to_rel_obj[src]:
            if relation1 == rels[0]:
                nodes.append(intermediate1)
                triples.append((src, relation1, intermediate1))
                
                for relation2, intermediate2 in self.subject_to_rel_obj[intermediate1]:
                    if relation2 == rels[1]:
                        nodes.append(intermediate2)
                        triples.append((intermediate1, relation2, intermediate2))
                        
                        for relation3, obj in self.subject_to_rel_obj[intermediate2]:
                            if relation3 == rels[2]:
                                nodes.append(obj)
                                triples.append((intermediate2, relation3, obj))
        
        return list(set(nodes)), list(set(triples))
    
    def deduce_subgraph_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Get subgraph by following a path."""
        path = [r for r in path if r != no_hop_flag]
        
        if len(path) == 0:
            return [src], []
        elif len(path) == 1:
            return self.deduce_subgraph_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_subgraph_by_path_two(src, path)
        elif len(path) == 3 and self.max_hop >= 3:
            return self.deduce_subgraph_by_path_three(src, path)
        else:
            raise ValueError(f"Path length {len(path)} exceeds maximum hop distance {self.max_hop}")
    
    def deduce_leaves_by_path_one(self, src: str, rels: List[str]) -> List[str]:
        """Get leaf entities by following a 1-hop path."""
        key = (src, rels[0])
        if key in self.subj_rel_to_obj:
            return self.subj_rel_to_obj[key]
        
        result = []
        for relation, obj in self.subject_to_rel_obj[src]:
            if relation == rels[0]:
                result.append(obj)
        
        return result
    
    def deduce_leaves_by_path_two(self, src: str, rels: List[str]) -> List[str]:
        """Get leaf entities by following a 2-hop path."""
        result = []
        
        # Get intermediate entities from first hop
        intermediates = self.get_tail(src, rels[0])
        
        # Follow second hop from each intermediate
        for intermediate in intermediates:
            for obj in self.get_tail(intermediate, rels[1]):
                result.append(obj)
        
        return result
    
    def deduce_leaves_by_path_three(self, src: str, rels: List[str]) -> List[str]:
        """Get leaf entities by following a 3-hop path."""
        result = []
        
        # Get intermediate entities from first hop
        intermediates1 = self.get_tail(src, rels[0])
        
        # Follow second hop from each first intermediate
        for intermediate1 in intermediates1:
            intermediates2 = self.get_tail(intermediate1, rels[1])
            
            # Follow third hop from each second intermediate
            for intermediate2 in intermediates2:
                for obj in self.get_tail(intermediate2, rels[2]):
                    result.append(obj)
        
        return result
    
    def deduce_leaves_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> List[str]:
        """Get leaf entities by following a path."""
        path = [r for r in path if r != no_hop_flag]
        
        if len(path) == 0:
            return [src]
        elif len(path) == 1:
            return self.deduce_leaves_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_leaves_by_path_two(src, path)
        elif len(path) == 3 and self.max_hop >= 3:
            return self.deduce_leaves_by_path_three(src, path)
        else:
            raise ValueError(f"Path length {len(path)} exceeds maximum hop distance {self.max_hop}")
    
    def deduce_leaves_count_by_path_one(self, src: str, rels: List[str]) -> int:
        """Count leaf entities by following a 1-hop path."""
        key = (src, rels[0])
        if key in self.subj_rel_to_obj:
            return len(self.subj_rel_to_obj[key])
        
        count = 0
        for relation, _ in self.subject_to_rel_obj[src]:
            if relation == rels[0]:
                count += 1
        
        return count
    
    def deduce_leaves_count_by_path_two(self, src: str, rels: List[str]) -> int:
        """Count leaf entities by following a 2-hop path."""
        count = 0
        
        # Get intermediate entities from first hop
        intermediates = self.get_tail(src, rels[0])
        
        # Count second hop entities from each intermediate
        for intermediate in intermediates:
            key = (intermediate, rels[1])
            if key in self.subj_rel_to_obj:
                count += len(self.subj_rel_to_obj[key])
            else:
                for relation, _ in self.subject_to_rel_obj[intermediate]:
                    if relation == rels[1]:
                        count += 1
        
        return count
    
    def deduce_leaves_count_by_path_three(self, src: str, rels: List[str]) -> int:
        """Count leaf entities by following a 3-hop path."""
        count = 0
        
        # Get intermediate entities from first hop
        intermediates1 = self.get_tail(src, rels[0])
        
        # Get second hop intermediates
        for intermediate1 in intermediates1:
            intermediates2 = self.get_tail(intermediate1, rels[1])
            
            # Count third hop entities
            for intermediate2 in intermediates2:
                key = (intermediate2, rels[2])
                if key in self.subj_rel_to_obj:
                    count += len(self.subj_rel_to_obj[key])
                else:
                    for relation, _ in self.subject_to_rel_obj[intermediate2]:
                        if relation == rels[2]:
                            count += 1
        
        return count
    
    def deduce_leaves_count_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> int:
        """Count leaf entities by following a path."""
        path = [r for r in path if r != no_hop_flag]
        
        if len(path) == 0:
            return 1
        elif len(path) == 1:
            return self.deduce_leaves_count_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_leaves_count_by_path_two(src, path)
        elif len(path) == 3 and self.max_hop >= 3:
            return self.deduce_leaves_count_by_path_three(src, path)
        else:
            raise ValueError(f"Path length {len(path)} exceeds maximum hop distance {self.max_hop}")
    
    def get_hr2t_with_limit(self, src: str, relation: str, limit: int = 100) -> List[str]:
        """Get tail entities for a given head entity and relation with a limit."""
        key = (src, relation)
        if key in self.subj_rel_to_obj:
            return self.subj_rel_to_obj[key][:limit]
        
        result = []
        for rel, obj in self.subject_to_rel_obj[src]:
            if rel == relation:
                result.append(obj)
                if len(result) >= limit:
                    break
        
        return result
    
    def is_ent(self, tp_str: str) -> bool:
        """Check if a string is an entity."""
        if len(tp_str) < 3:
            return False
        if tp_str.startswith("m.") or tp_str.startswith("g."):
            return True
        return False
    
    def deduce_relations_from_src_list(self, src_list: List[str], limit: int = 100) -> List[str]:
        """Get relations from a list of source entities."""
        result = set()
        
        for src in src_list:
            if self.is_ent(src):
                for relation, _ in self.subject_to_rel_obj[src]:
                    result.add(relation)
                    if len(result) >= limit:
                        break
        
        return list(result)
    
    def deduce_leaves_from_src_list_and_relation(self, src_list: List[str], relation: str, limit: int = 100) -> List[str]:
        """Get leaf entities from a list of source entities and a relation."""
        result = []
        
        for src in src_list:
            if self.is_ent(src):
                key = (src, relation)
                if key in self.subj_rel_to_obj:
                    result.extend(self.subj_rel_to_obj[key])
                    if len(result) >= limit:
                        break
                else:
                    for rel, obj in self.subject_to_rel_obj[src]:
                        if rel == relation:
                            result.append(obj)
                            if len(result) >= limit:
                                break
        
        return result[:limit]
    
    def deduce_leaves_relation_by_path_one(self, src: str, rels: List[str], node_limit: int = 30, rel_limit: int = 50) -> List[str]:
        """Get relations of leaf entities by following a 1-hop path."""
        one_hop_leaves = self.get_hr2t_with_limit(src, rels[0], node_limit)
        leave_relations = self.deduce_relations_from_src_list(one_hop_leaves, rel_limit)
        return leave_relations
    
    def deduce_leaves_relation_by_path_two(self, src: str, rels: List[str], one_node_limit: int = 30, two_node_limit: int = 100, rel_limit: int = 50) -> List[str]:
        """Get relations of leaf entities by following a 2-hop path."""
        one_hop_leaves = self.get_hr2t_with_limit(src, rels[0], one_node_limit)
        two_hop_leaves = self.deduce_leaves_from_src_list_and_relation(one_hop_leaves, rels[1], two_node_limit)
        leave_relations = self.deduce_relations_from_src_list(two_hop_leaves, rel_limit)
        return leave_relations
    
    def deduce_leaves_relation_by_path_three(self, src: str, rels: List[str], one_node_limit: int = 20, two_node_limit: int = 50, three_node_limit: int = 100, rel_limit: int = 50) -> List[str]:
        """Get relations of leaf entities by following a 3-hop path."""
        # First hop
        one_hop_leaves = self.get_hr2t_with_limit(src, rels[0], one_node_limit)
        
        # Second hop
        two_hop_leaves = []
        for node in one_hop_leaves:
            hop_results = self.get_hr2t_with_limit(node, rels[1], two_node_limit // len(one_hop_leaves) + 1)
            two_hop_leaves.extend(hop_results)
            if len(two_hop_leaves) >= two_node_limit:
                two_hop_leaves = two_hop_leaves[:two_node_limit]
                break
        
        # Third hop
        three_hop_leaves = []
        for node in two_hop_leaves:
            hop_results = self.get_hr2t_with_limit(node, rels[2], three_node_limit // len(two_hop_leaves) + 1)
            three_hop_leaves.extend(hop_results)
            if len(three_hop_leaves) >= three_node_limit:
                three_hop_leaves = three_hop_leaves[:three_node_limit]
                break
        
        # Get relations from leaf nodes
        leave_relations = self.deduce_relations_from_src_list(three_hop_leaves, rel_limit)
        return leave_relations
    
    def deduce_leaves_relation_by_path(self, src: str, path: List[str]) -> List[str]:
        """Get relations of leaf entities by following a path."""
        path = [r for r in path if r != 'END OF HOP']
        
        if len(path) == 0:
            return self.get_relation(src)
        elif len(path) == 1:
            return self.deduce_leaves_relation_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_leaves_relation_by_path_two(src, path)
        elif len(path) == 3 and self.max_hop >= 3:
            return self.deduce_leaves_relation_by_path_three(src, path)
        else:
            raise ValueError(f"Path length {len(path)} exceeds maximum hop distance {self.max_hop}")