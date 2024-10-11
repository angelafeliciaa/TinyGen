import ast
import os
import networkx as nx
from pyvis.network import Network
import html
import json

def is_binary_content(content: str) -> bool:
    return '\x00' in content or not is_valid_utf8(content)

def is_valid_utf8(content: str) -> bool:
    try:
        content.encode('utf-8').decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def extract_code_snippet(content, max_lines=10):
    lines = content.split('\n')
    return '\n'.join(lines[:max_lines]).strip()

class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = {}
        self.current_function = None

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.function_calls[node.name] = set()
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and self.current_function:
            self.function_calls[self.current_function].add(node.func.id)
        self.generic_visit(node)

def parse_file(content):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [], {}, {}

    imports = []
    definitions = {}
    visitor = FunctionCallVisitor()
    visitor.visit(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append((f"{module}.{alias.name}", alias.name))
        elif isinstance(node, ast.FunctionDef):
            definitions[node.name] = 'function'
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = 'class'

    return imports, definitions, visitor.function_calls

def build_import_graph(repo_content):
    graph = nx.DiGraph()

    for file_path, content in repo_content.items():
        if is_binary_content(content):
            continue

        module_name = file_path.replace('.py', '').replace('/', '.')
        snippet = extract_code_snippet(content)
        
        graph.add_node(module_name, snippet=snippet, type='file')

        imports, definitions, function_calls = parse_file(content)
        for full_name, symbol in imports:
            target_module = full_name.split('.')[0]
            graph.add_edge(module_name, target_module, symbol=symbol)

        for name, def_type in definitions.items():
            node_name = f"{module_name}.{name}"
            graph.add_node(node_name, type=def_type)
            graph.add_edge(module_name, node_name)

            if name in function_calls:
                for called_func in function_calls[name]:
                    called_node = f"{module_name}.{called_func}"
                    if graph.has_node(called_node):
                        graph.add_edge(node_name, called_node, type='function_call')

    return graph

def get_node_color(node_type):
    colors = {
        'file': '#ADD8E6',    # Light Blue
        'function': '#90EE90', # Light Green
        'class': '#FFB6C1',    # Light Pink
        'module': '#FFD700'    # Gold
    }
    return colors.get(node_type, '#FFFFFF')  # Default to white if type is unknown

def visualize_import_graph(repo_content, output_file):
    # Filter repo_content to only include files that should be processed
    filtered_repo_content = {file_path: content for file_path, content in repo_content.items() if should_process_file(file_path)}
    
    # Build the import graph using the filtered content
    graph = build_import_graph(filtered_repo_content)
    
    # Create Network
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    
    # Define options
    options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": True,
                "iterations": 1000,
                "updateInterval": 25
            }
        },
        "nodes": {
            "font": {
                "size": 12
            }
        },
        "edges": {
            "font": {
                "size": 10
            },
            "smooth": False
        }
    }
    
    # Set options
    net.set_options(json.dumps(options))
    
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        color = get_node_color(node_type)
        snippet = html.escape(data.get('snippet', 'No snippet available'))
        net.add_node(
            node,
            label=node.split('.')[-1],
            title=f"{node}\n\n{snippet}",
            color=color,
            group=node_type
        )
    
    for source, target, data in graph.edges(data=True):
        edge_type = data.get('type', '')
        if edge_type == 'function_call':
            net.add_edge(source, target, color='#FF0000', title='Function Call')
        else:
            net.add_edge(source, target, title=data.get('symbol', ''))

    # Add custom HTML for search and legend
    custom_html = """
    <input type="text" id="searchBox" placeholder="Search nodes..." style="position: absolute; top: 10px; left: 10px; z-index: 1000;">
    <div id="legend" style="position: absolute; top: 10px; right: 10px; z-index: 1000; background: white; padding: 10px; border: 1px solid black;">
        <div><span style="color: #ADD8E6;">■</span> File</div>
        <div><span style="color: #90EE90;">■</span> Function</div>
        <div><span style="color: #FFB6C1;">■</span> Class</div>
        <div><span style="color: #FFD700;">■</span> Module</div>
        <div><span style="color: #FF0000;">―</span> Function Call</div>
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', (event) => {
        const searchBox = document.getElementById('searchBox');
        searchBox.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const network = document.getElementById('mynetwork').getElementsByTagName('iframe')[0].contentWindow.network;
            if (network && network.body && network.body.data && network.body.data.nodes) {
                const allNodes = network.body.data.nodes.get();
                const matchingNodes = allNodes.filter(node => 
                    node.label.toLowerCase().includes(searchTerm) || 
                    (node.title && node.title.toLowerCase().includes(searchTerm))
                );
                const matchingIds = matchingNodes.map(node => node.id);
                
                network.body.data.nodes.update(allNodes.map(node => ({
                    ...node,
                    color: matchingIds.includes(node.id) ? 
                        {background: '#FFFF00', border: node.color.background} : 
                        node.color
                })));
            }
        });
    });
    </script>
    """

    # Save graph with custom HTML
    net.save_graph(output_file)
    
    # Read the saved file
    with open(output_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Insert custom HTML before the closing body tag
    modified_html = html_content.replace('</body>', f'{custom_html}</body>')
    
    # Write the modified content back to the file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(modified_html)

    return output_file

def should_process_file(file_path):
    # List of directories and file extensions to ignore
    ignore_dirs = ['.idea', '.git', '__pycache__', 'venv', 'env']
    ignore_extensions = ['.pyc', '.pyo', '.pyd', '.db', '.lock', '.toml', '.md', '.txt', '.xml', '.json']
    ignore_filenames = ['requirements.txt', 'Pipfile', 'poetry.lock', 'README.md', 'README.rst']

    # Check if the file is in an ignored directory
    if any(ignored_dir in file_path.split(os.sep) for ignored_dir in ignore_dirs):
        return False

    # Check if the file has an ignored extension
    if any(file_path.endswith(ext) for ext in ignore_extensions):
        return False

    # Check if the file is in the list of ignored filenames
    if os.path.basename(file_path) in ignore_filenames:
        return False

    return True