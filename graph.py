import networkx as nx
from pyvis.network import Network
import json
import uuid

def extract_clinical_entities(chunks, query_text="Clinical Query", stage=3, reasoning_path=None):
    """
    Extracts clinical entities with dynamic support for 'Live Tracking' in stages.
    Stage 1: Documents & Query only.
    Stage 2: Expand all Medical Entities.
    Stage 3: Highlight the active Reasoning Path (dimming the rest).
    """
    G = nx.DiGraph()
    
    # STAGE 1: Query and Documents
    G.add_node("Query", group="Query", color="#38bdf8", size=25)
    for chunk in chunks:
        doc_id = chunk["id"]
        G.add_node(doc_id, group="Document", color="#94a3b8", size=15)
        G.add_edge(doc_id, "Query", label="RETRIEVED", color="#64748b")
        
    if stage == 1:
        return G
        
    # STAGE 2: Add Entities
    text_content = " ".join([c["content"].lower() for c in chunks])
    
    # Achilles Tendinopathy relationships
    if "achilles" in text_content:
        G.add_node("Achilles Tendinopathy", group="Condition", color="#22c55e", size=20)
        G.add_edge("guideline-achilles-001", "Achilles Tendinopathy", label="MENTIONS")
        G.add_edge("guideline-achilles-002", "Achilles Tendinopathy", label="MENTIONS")
        
        if "load management" in text_content:
            G.add_node("Load Management", group="Treatment", color="#3b82f6", size=20)
            G.add_edge("Load Management", "Achilles Tendinopathy", label="TREATS")
            
        if "corticosteroid" in text_content:
            G.add_node("Corticosteroid Injections", group="Intervention", color="#ef4444", size=20)
            G.add_node("Tendon Rupture", group="Risk", color="#ef4444", size=20)
            G.add_edge("Corticosteroid Injections", "Achilles Tendinopathy", label="CONTRAINDICATED FOR")
            G.add_edge("Corticosteroid Injections", "Tendon Rupture", label="INCREASES RISK OF")
            
    # Concussion relationships
    if "concussion" in text_content:
        G.add_node("Concussion", group="Condition", color="#22c55e", size=20)
        G.add_edge("guideline-concussion-001", "Concussion", label="MENTIONS")
        
        if "nsaid" in text_content:
            G.add_node("NSAIDs", group="Medication", color="#3b82f6", size=20)
            G.add_node("Intracranial Bleeding", group="Risk", color="#ef4444", size=20)
            G.add_edge("NSAIDs", "Concussion", label="CONTRAINDICATED IN < 48H")
            G.add_edge("NSAIDs", "Intracranial Bleeding", label="INCREASES RISK OF")
            
    if stage == 2 or not reasoning_path:
        return G

    # STAGE 3: Reasoning Path Highlighting (Active Traversal)
    # Dim all non-relevant nodes and edges
    for node in G.nodes:
        G.nodes[node]['color'] = 'rgba(148, 163, 184, 0.2)' # Dim grey
        G.nodes[node]['size'] = 10
        G.nodes[node]['font'] = {'color': 'rgba(226, 232, 240, 0.3)'}
        
    for u, v in G.edges:
        G.edges[u, v]['color'] = 'rgba(148, 163, 184, 0.1)'
        G.edges[u, v]['width'] = 1
        G.edges[u, v]['font'] = {'color': 'rgba(226, 232, 240, 0.2)'}

    # "Glowing" Path: Highlight nodes in reasoning path
    for node in reasoning_path:
        if node in G.nodes:
            # Determine color based on node semantics
            color = "#ef4444" if node in ["Tendon Rupture", "Corticosteroid Injections", "Intracranial Bleeding"] else "#38bdf8"
            G.nodes[node]['color'] = color
            G.nodes[node]['size'] = 35
            G.nodes[node]['font'] = {'color': '#ffffff', 'size': 18, 'weight': 'bold'}
            G.nodes[node]['shadow'] = {'enabled': True, 'color': color, 'size': 30, 'x': 0, 'y': 0}
            
    # Highlight sequential edges connecting the path
    for i in range(len(reasoning_path) - 1):
        u, v = reasoning_path[i], reasoning_path[i+1]
        
        # Check both directions since reasoning paths might reverse edge definitions
        if G.has_edge(u, v):
            edge = G.edges[u, v]
            edge_col = "#ef4444" if "CONTRAINDICATED" in edge.get('label', '') or "RISK" in edge.get('label', '') else "#06b6d4"
            edge['color'] = edge_col
            edge['width'] = 5
            edge['font'] = {'color': edge_col, 'size': 14}
        elif G.has_edge(v, u):
            edge = G.edges[v, u]
            edge_col = "#ef4444" if "CONTRAINDICATED" in edge.get('label', '') or "RISK" in edge.get('label', '') else "#06b6d4"
            edge['color'] = edge_col
            edge['width'] = 5
            edge['font'] = {'color': edge_col, 'size': 14}
            
    return G

def generate_interactive_graph(G, output_path="graph.html"):
    """
    Generates a PyVis HTML interactive graph from a NetworkX graph.
    """
    net = Network(height="100%", width="100%", bgcolor="#0b1115", font_color="#e2e8f0", directed=True)
    net.from_nx(G)
    
    # Configure physics for a professional fluid look
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "shadow": {
          "enabled": true,
          "color": "rgba(56, 189, 248, 0.4)",
          "size": 15,
          "x": 0,
          "y": 0
        }
      },
      "edges": {
        "color": {
          "color": "#94a3b8",
          "highlight": "#38bdf8"
        },
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        }
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -150,
          "centralGravity": 0.05,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "minVelocity": 0.75,
        "stabilization": {
           "enabled": true,
           "iterations": 50,
           "updateInterval": 10
        }
      }
    }
    """)
    
    # Add JS script to automatically focus/zoom camera on the active nodes in the graph
    # This fulfills the "Camera Focus" requirement
    net.save_graph(output_path)
    
    with open(output_path, "r", encoding="utf-8") as f:
        html_data = f.read()
        
    # Inject JS for auto-focusing on the large active nodes
    focus_js = """
    <script type="text/javascript">
      network.once("stabilizationIterationsDone", function() {
          var allNodes = nodes.get();
          var activeNodeIds = [];
          for (var i = 0; i < allNodes.length; i++) {
              if (allNodes[i].size > 20) {
                  activeNodeIds.push(allNodes[i].id);
              }
          }
          if (activeNodeIds.length > 0) {
              network.fit({
                  nodes: activeNodeIds,
                  animation: {
                      duration: 1500,
                      easingFunction: 'easeInOutQuad'
                  }
              });
          }
      });
    </script>
    </body>
    """
    html_data = html_data.replace("</body>", focus_js)
    
    return html_data
