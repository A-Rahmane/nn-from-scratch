from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def visualize_computation_graph(
    tensor: "Tensor",
    format: str = "png",
    filename: str = "computation_graph",
    view: bool = True
) -> None:
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
    
    dot = Digraph(format=format, comment='Computational Graph')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    nodes: Set[int] = set()
    edges: Set[tuple] = set()
    
    def add_node(t: "Tensor") -> None:
        node_id = str(id(t))
        
        if id(t) not in nodes:
            nodes.add(id(t))
            
            label_parts = []
            
            if t._op:
                label_parts.append(f"op: {t._op}")
            else:
                label_parts.append("input")
            
            label_parts.append(f"shape: {t.shape}")
            
            if t.requires_grad:
                grad_status = "grad: ✓" if t.grad is not None else "grad: tracked"
                label_parts.append(grad_status)
            
            label = "\\n".join(label_parts)
            
            if not t._op:
                color = 'lightgreen'
            elif t.grad is not None:
                color = 'lightyellow'
            elif t.requires_grad:
                color = 'lightblue'
            else:
                color = 'lightgray'
            
            dot.node(node_id, label, fillcolor=color)
    
    def build_graph(t: "Tensor") -> None:
        add_node(t)
        
        for parent in t._prev:
            add_node(parent)
            
            edge = (str(id(parent)), str(id(t)))
            if edge not in edges:
                edges.add(edge)
                
                edge_label = ""
                if t._op:
                    edge_label = t._op
                
                dot.edge(edge[0], edge[1], label=edge_label)
            
            build_graph(parent)
    
    build_graph(tensor)
    
    try:
        output_path = dot.render(filename, cleanup=True, view=view)
        print(f"✓ Computational graph saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to render graph: {e}")
        print("Ensure graphviz system library is installed.")