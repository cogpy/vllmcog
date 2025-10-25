"""Tests for AtomSpace implementation."""

import pytest
from vllm.opencog.atomspace import AtomSpace, Node, Link, AtomType, Atom


class TestAtom:
    """Tests for Atom base class."""
    
    def test_atom_creation(self):
        """Test basic atom creation."""
        atom = Atom(name="test", atom_type=AtomType.NODE)
        assert atom.name == "test"
        assert atom.atom_type == AtomType.NODE
        assert atom.truth_value == 1.0
    
    def test_atom_equality(self):
        """Test atom equality comparison."""
        atom1 = Atom(name="test", atom_type=AtomType.NODE)
        atom2 = Atom(name="test", atom_type=AtomType.NODE)
        atom3 = Atom(name="other", atom_type=AtomType.NODE)
        
        assert atom1 == atom2
        assert atom1 != atom3
    
    def test_atom_hash(self):
        """Test atom hashing for sets/dicts."""
        atom1 = Atom(name="test", atom_type=AtomType.NODE)
        atom2 = Atom(name="test", atom_type=AtomType.NODE)
        
        atom_set = {atom1, atom2}
        assert len(atom_set) == 1


class TestNode:
    """Tests for Node class."""
    
    def test_node_creation(self):
        """Test node creation."""
        node = Node(name="concept", truth_value=0.9)
        assert node.name == "concept"
        assert node.atom_type == AtomType.NODE
        assert node.truth_value == 0.9
    
    def test_node_metadata(self):
        """Test node with metadata."""
        node = Node(
            name="agent",
            metadata={"type": "inference", "id": 1}
        )
        assert node.metadata["type"] == "inference"
        assert node.metadata["id"] == 1


class TestLink:
    """Tests for Link class."""
    
    def test_link_creation(self):
        """Test link creation."""
        node1 = Node(name="concept1")
        node2 = Node(name="concept2")
        
        link = Link(name="relationship", outgoing=[node1, node2])
        
        assert link.name == "relationship"
        assert link.atom_type == AtomType.LINK
        assert len(link.outgoing) == 2
    
    def test_link_incoming_update(self):
        """Test that links update incoming sets."""
        node1 = Node(name="source")
        node2 = Node(name="target")
        
        link = Link(name="connects", outgoing=[node1, node2])
        
        assert link in node1.incoming
        assert link in node2.incoming


class TestAtomSpace:
    """Tests for AtomSpace class."""
    
    def test_atomspace_creation(self):
        """Test atomspace initialization."""
        atomspace = AtomSpace()
        assert atomspace.size() == 0
    
    def test_add_node(self):
        """Test adding nodes to atomspace."""
        atomspace = AtomSpace()
        
        node = atomspace.add_node("concept1")
        assert isinstance(node, Node)
        assert atomspace.size() == 1
    
    def test_add_duplicate_node(self):
        """Test that duplicate nodes are merged."""
        atomspace = AtomSpace()
        
        node1 = atomspace.add_node("concept", truth_value=0.8)
        node2 = atomspace.add_node("concept", truth_value=0.9)
        
        # Should be the same node with updated truth value
        assert node1 == node2
        assert atomspace.size() == 1
        assert node1.truth_value == 0.9
    
    def test_add_link(self):
        """Test adding links to atomspace."""
        atomspace = AtomSpace()
        
        node1 = atomspace.add_node("node1")
        node2 = atomspace.add_node("node2")
        
        link = atomspace.add_link("relates", [node1, node2])
        
        assert isinstance(link, Link)
        assert atomspace.size() == 3  # 2 nodes + 1 link
    
    def test_get_atom(self):
        """Test retrieving atoms by name and type."""
        atomspace = AtomSpace()
        
        node = atomspace.add_node("test_node")
        
        retrieved = atomspace.get_atom("test_node", AtomType.NODE)
        assert retrieved == node
    
    def test_get_all_nodes(self):
        """Test getting all nodes."""
        atomspace = AtomSpace()
        
        atomspace.add_node("node1")
        atomspace.add_node("node2")
        atomspace.add_node("node3")
        
        nodes = atomspace.get_all_nodes()
        assert len(nodes) == 3
    
    def test_get_all_links(self):
        """Test getting all links."""
        atomspace = AtomSpace()
        
        node1 = atomspace.add_node("node1")
        node2 = atomspace.add_node("node2")
        
        atomspace.add_link("link1", [node1, node2])
        atomspace.add_link("link2", [node2, node1])
        
        links = atomspace.get_all_links()
        assert len(links) == 2
    
    def test_pattern_match_by_type(self):
        """Test pattern matching by atom type."""
        atomspace = AtomSpace()
        
        atomspace.add_node("concept1")
        atomspace.add_node("concept2")
        node1 = atomspace.add_node("node1")
        node2 = atomspace.add_node("node2")
        atomspace.add_link("link", [node1, node2])
        
        # Match all nodes
        results = atomspace.pattern_match({"type": "node"})
        assert len(results) == 4
        
        # Match all links
        results = atomspace.pattern_match({"type": "link"})
        assert len(results) == 1
    
    def test_pattern_match_by_name(self):
        """Test pattern matching by name."""
        atomspace = AtomSpace()
        
        atomspace.add_node("agent:1")
        atomspace.add_node("agent:2")
        atomspace.add_node("task:1")
        
        # Match agents
        results = atomspace.pattern_match({"name": "agent"})
        assert len(results) == 2
    
    def test_pattern_match_by_truth_value(self):
        """Test pattern matching by truth value."""
        atomspace = AtomSpace()
        
        atomspace.add_node("high_truth", truth_value=0.9)
        atomspace.add_node("low_truth", truth_value=0.3)
        
        # Match high truth values
        results = atomspace.pattern_match({"min_truth_value": 0.8})
        assert len(results) == 1
    
    def test_clear(self):
        """Test clearing the atomspace."""
        atomspace = AtomSpace()
        
        atomspace.add_node("node1")
        atomspace.add_node("node2")
        assert atomspace.size() == 2
        
        atomspace.clear()
        assert atomspace.size() == 0
    
    def test_thread_safety(self):
        """Test basic thread safety."""
        import threading
        
        atomspace = AtomSpace()
        
        def add_nodes(prefix):
            for i in range(10):
                atomspace.add_node(f"{prefix}_{i}")
        
        threads = [
            threading.Thread(target=add_nodes, args=(f"thread{i}",))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert atomspace.size() == 50
