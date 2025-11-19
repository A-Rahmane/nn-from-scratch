"""Computational graph visualization utilities.

This module provides functions to visualize computational graphs:
- visualize_computation_graph: Create visual representation using graphviz

Visualization helps with:
- Understanding data flow
- Debugging gradient issues
- Teaching/explaining autograd
- Documenting model architectures
"""

from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def visualize_computation_graph(
    tensor: "Tensor",
    format: str = "png",
    filename: str = "computation_graph",
    view: bool = True
) -> None:
    """
    Visualize the computational graph leading to a tensor.
    
    Creates a directed graph showing:
    - Tensor nodes (boxes)
    - Operation names
    - Tensor shapes
    - Gradient tracking status
    - Connections between tensors
    
    Requires:
        graphviz package: pip install graphviz
        graphviz system library: https://graphviz.org/download/
    
    Graph Layout:
        - Left to right: inputs → operations → outputs
        - Boxes: tensor nodes
        - Edges: data flow
        - Labels: operation names and shapes
    
    Args:
        tensor: Output tensor to visualize graph for
        format: Output format ('png', 'pdf', 'svg', 'dot')
        filename: Output filename (without extension)
        view: If True, open the graph after creating it
    
    Raises:
        ImportError: If graphviz package is not installed
    
    Notes:
        - Only visualizes the computational graph, not the actual data values
        - Large graphs may be difficult to read
        - Colors and styling can be customized by modifying the function
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("ERROR: graphviz package not installed.")
        print("Install with: pip install graphviz")
        print("Also ensure graphviz system library is installed:")
        print("  - Ubuntu/Debian: sudo apt-get install graphviz")
        print("  - macOS: brew install graphviz")
        print("  - Windows: https://graphviz.org/download/")
        return
    
    # Create directed graph
    dot = Digraph(format=format, comment='Computational Graph')
    dot.attr(rankdir='LR')  # Left to right layout
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Track visited nodes and edges
    nodes: Set[int] = set()
    edges: Set[tuple] = set()
    
    def add_node(t: "Tensor") -> None:
        """Add a tensor node to the graph.
        
        Args:
            t: Tensor to add as node
        """
        node_id = str(id(t))
        
        if id(t) not in nodes:
            nodes.add(id(t))
            
            # Create label with operation, shape, and gradient info
            label_parts = []
            
            # Operation name
            if t._op:
                label_parts.append(f"op: {t._op}")
            else:
                label_parts.append("input")
            
            # Shape
            label_parts.append(f"shape: {t.shape}")
            
            # Gradient tracking status
            if t.requires_grad:
                grad_status = "grad: ✓" if t.grad is not None else "grad: tracked"
                label_parts.append(grad_status)
            
            label = "\\n".join(label_parts)
            
            # Different colors for different node types
            if not t._op:
                # Input tensor (no operation)
                color = 'lightgreen'
            elif t.grad is not None:
                # Has computed gradient
                color = 'lightyellow'
            elif t.requires_grad:
                # Tracks gradients but hasn't computed yet
                color = 'lightblue'
            else:
                # No gradient tracking
                color = 'lightgray'
            
            dot.node(node_id, label, fillcolor=color)
    
    def build_graph(t: "Tensor") -> None:
        """Recursively build graph by traversing computational graph.
        
        Args:
            t: Current tensor to process
        """
        # Add current node
        add_node(t)
        
        # Add parent nodes and edges
        for parent in t._prev:
            # Add parent node
            add_node(parent)
            
            # Add edge from parent to current
            edge = (str(id(parent)), str(id(t)))
            if edge not in edges:
                edges.add(edge)
                
                # Label edge with operation if available
                edge_label = ""
                if t._op:
                    edge_label = t._op
                
                dot.edge(edge[0], edge[1], label=edge_label)
            
            # Recursively process parent
            build_graph(parent)
    
    # Build the graph starting from output tensor
    build_graph(tensor)
    
    # Render the graph
    try:
        output_path = dot.render(filename, cleanup=True, view=view)
        print(f"✓ Computational graph saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to render graph: {e}")
        print("Ensure graphviz system library is installed.")


# Test visualization when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Visualization ===\n")
    
    # Test simple graph
    print("Test 1: Simple computation graph")
    x = Tensor([2.0], requires_grad=True)
    y = x ** 2
    z = y + 3
    
    try:
        visualize_computation_graph(z, filename="test_simple", view=False)
        print("✓ Simple graph created\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Test complex graph
    print("Test 2: Multi-path computation graph")
    x = Tensor([3.0], requires_grad=True)
    y1 = x ** 2
    y2 = x * 3
    z = y1 + y2
    
    try:
        visualize_computation_graph(z, filename="test_multipath", view=False)
        print("✓ Multi-path graph created\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Test with gradients computed
    print("Test 3: Graph with computed gradients")
    x = Tensor([2.0], requires_grad=True)
    y = (x ** 2 + x * 3).sum()
    y.backward()
    
    try:
        visualize_computation_graph(y, filename="test_with_grads", view=False)
        print("✓ Graph with gradients created\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    print("✅ Visualization tests complete!")
    print("Note: Check the generated .png files in the current directory")