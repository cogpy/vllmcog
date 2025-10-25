"""AtomSpace implementation for knowledge representation.

The AtomSpace is a hypergraph knowledge representation system that stores
relationships between concepts and enables pattern matching and reasoning.
"""

import threading
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class AtomType(Enum):
    """Types of atoms in the AtomSpace."""
    NODE = "node"
    LINK = "link"


@dataclass
class Atom:
    """Base class for atoms in the AtomSpace."""
    
    name: str
    atom_type: AtomType
    truth_value: float = 1.0
    attention_value: float = 0.0
    incoming: Set['Atom'] = field(default_factory=set)
    outgoing: List['Atom'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.name, self.atom_type))
    
    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return self.name == other.name and self.atom_type == other.atom_type


class Node(Atom):
    """Node atom representing a concept or entity."""
    
    def __init__(self, name: str, truth_value: float = 1.0, **kwargs):
        super().__init__(
            name=name,
            atom_type=AtomType.NODE,
            truth_value=truth_value,
            **kwargs
        )


class Link(Atom):
    """Link atom representing a relationship between atoms."""
    
    def __init__(self, name: str, outgoing: List[Atom], 
                 truth_value: float = 1.0, **kwargs):
        super().__init__(
            name=name,
            atom_type=AtomType.LINK,
            outgoing=outgoing,
            truth_value=truth_value,
            **kwargs
        )
        # Update incoming sets for outgoing atoms
        for atom in outgoing:
            atom.incoming.add(self)


class AtomSpace:
    """Hypergraph knowledge representation system.
    
    The AtomSpace stores atoms (nodes and links) and provides operations
    for querying, pattern matching, and reasoning over the knowledge graph.
    """
    
    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._lock = threading.RLock()
        self._indices: Dict[str, Set[Atom]] = {
            'nodes': set(),
            'links': set(),
        }
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the AtomSpace.
        
        Args:
            atom: The atom to add
            
        Returns:
            The added atom (or existing atom if already present)
        """
        with self._lock:
            key = f"{atom.atom_type.value}:{atom.name}"
            
            if key in self._atoms:
                # Merge with existing atom
                existing = self._atoms[key]
                existing.truth_value = max(existing.truth_value, 
                                          atom.truth_value)
                existing.metadata.update(atom.metadata)
                return existing
            
            self._atoms[key] = atom
            
            if atom.atom_type == AtomType.NODE:
                self._indices['nodes'].add(atom)
            else:
                self._indices['links'].add(atom)
            
            return atom
    
    def add_node(self, name: str, **kwargs) -> Node:
        """Add a node to the AtomSpace.
        
        Args:
            name: Name of the node
            **kwargs: Additional node attributes
            
        Returns:
            The created node
        """
        node = Node(name=name, **kwargs)
        return self.add_atom(node)
    
    def add_link(self, name: str, outgoing: List[Atom], **kwargs) -> Link:
        """Add a link to the AtomSpace.
        
        Args:
            name: Name of the link
            outgoing: List of atoms this link connects
            **kwargs: Additional link attributes
            
        Returns:
            The created link
        """
        link = Link(name=name, outgoing=outgoing, **kwargs)
        return self.add_atom(link)
    
    def get_atom(self, name: str, atom_type: AtomType) -> Optional[Atom]:
        """Retrieve an atom by name and type.
        
        Args:
            name: Name of the atom
            atom_type: Type of the atom
            
        Returns:
            The atom if found, None otherwise
        """
        with self._lock:
            key = f"{atom_type.value}:{name}"
            return self._atoms.get(key)
    
    def get_all_nodes(self) -> Set[Node]:
        """Get all nodes in the AtomSpace."""
        with self._lock:
            return self._indices['nodes'].copy()
    
    def get_all_links(self) -> Set[Link]:
        """Get all links in the AtomSpace."""
        with self._lock:
            return self._indices['links'].copy()
    
    def get_incoming(self, atom: Atom) -> Set[Atom]:
        """Get all atoms pointing to the given atom.
        
        Args:
            atom: The target atom
            
        Returns:
            Set of atoms with outgoing connections to the target
        """
        return atom.incoming.copy()
    
    def get_outgoing(self, atom: Atom) -> List[Atom]:
        """Get all atoms the given atom points to.
        
        Args:
            atom: The source atom
            
        Returns:
            List of outgoing atoms
        """
        return atom.outgoing.copy()
    
    def pattern_match(self, pattern: Dict[str, Any]) -> List[Atom]:
        """Simple pattern matching in the AtomSpace.
        
        Args:
            pattern: Pattern specification with 'type', 'name' filters
            
        Returns:
            List of atoms matching the pattern
        """
        with self._lock:
            results = []
            
            atom_type = pattern.get('type')
            name_pattern = pattern.get('name')
            min_truth = pattern.get('min_truth_value', 0.0)
            
            atoms_to_search = []
            if atom_type == 'node':
                atoms_to_search = self._indices['nodes']
            elif atom_type == 'link':
                atoms_to_search = self._indices['links']
            else:
                atoms_to_search = list(self._atoms.values())
            
            for atom in atoms_to_search:
                if name_pattern and name_pattern not in atom.name:
                    continue
                if atom.truth_value < min_truth:
                    continue
                results.append(atom)
            
            return results
    
    def clear(self):
        """Clear all atoms from the AtomSpace."""
        with self._lock:
            self._atoms.clear()
            self._indices['nodes'].clear()
            self._indices['links'].clear()
    
    def size(self) -> int:
        """Get the total number of atoms in the AtomSpace."""
        with self._lock:
            return len(self._atoms)


__all__ = ["Atom", "Node", "Link", "AtomSpace", "AtomType"]
