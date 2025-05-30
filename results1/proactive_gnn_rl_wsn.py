import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from datetime import datetime
import os
import heapq
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Create directory for saving results
os.makedirs("results", exist_ok=True)

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## WSN Parameters ##############################

# Sensing Field Dimensions (meters)
FIELD_X = 100
FIELD_Y = 100

# Network parameters
NUM_NODES = 100
SINK_X = 100
SINK_Y = 100
TRANSMISSION_RANGE = 20
CH_PERCENTAGE = 0.1  # 10% of nodes as Cluster Heads

# GNN parameters
GNN_HIDDEN_CHANNELS = 64
GNN_NUM_LAYERS = 3
PREDICTION_HORIZON = 10  # Number of rounds to predict future states

# Energy parameters (all in Joules)
INITIAL_ENERGY_MIN = 2.0
INITIAL_ENERGY_MAX = 5.0
E_ELEC = 50e-9        # Energy for running transceiver circuitry (J/bit)
E_AMP = 100e-12       # Energy for transmitter amplifier (J/bit/m²)
E_DA = 5e-9           # Energy for data aggregation (J/bit)
PACKET_SIZE = 4000    # Size of data packet (bits)

# Simulation parameters
MAX_ROUNDS = 1000     # Maximum number of rounds to run the simulation
DEAD_NODE_THRESHOLD = 0.05  # Node is dead when energy falls below this threshold

############################## DRL Parameters ##############################

# DQN Hyperparameters
MEMORY_SIZE = 5000     # Reduced memory size
BATCH_SIZE = 32        # Smaller batch size for faster training
GAMMA = 0.95           # Slightly reduced discount factor
EPSILON_START = 0.9    # Start with less exploration
EPSILON_END = 0.05     # Higher minimum exploration
EPSILON_DECAY = 0.99   # Faster decay
TARGET_UPDATE = 5      # More frequent updates
LEARNING_RATE = 0.003  # Higher learning rate for faster convergence
WARMUP_STEPS = 200     # Reduced warmup steps
GNN_TRAINING_INTERVAL = 3  # More frequent GNN training
DRL_TRAINING_INTERVAL = 1 # Define DRL_TRAINING_INTERVAL

# GNN Parameters
NODE_FEATURE_SIZE = 7  # [energy_ratio, x, y, dist_to_sink, hop_count, network_energy, congestion]
EDGE_FEATURE_SIZE = 3  # Edge features: distance, energy ratio, signal strength
GNN_OUTPUT_SIZE = 64   # Output embedding size from GNN (should match hidden channels)

# Proactive planning parameters
FUTURE_DISCOUNT = 0.8  # Discount factor for future energy predictions
PLANNING_HORIZON = 5   # How many rounds ahead to plan routes
TRAFFIC_PREDICTION_WINDOW = 10  # Window size for traffic pattern prediction

############################## Graph Neural Network ##############################

class WSN_GNN(nn.Module):
    """Graph Neural Network for WSN topology modeling and prediction"""
    
    def __init__(self, node_features, edge_features, hidden_channels, output_size):
        super(WSN_GNN, self).__init__()
        self.hidden_channels = hidden_channels
        
        # GNN layers - all using same hidden_channels size
        self.conv1 = GATConv(node_features, hidden_channels, edge_dim=edge_features)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=edge_features)
        self.conv3 = GATConv(hidden_channels, hidden_channels, edge_dim=edge_features)
        
        # Energy prediction layers
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_channels + node_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        # Route scoring layers
        self.route_scorer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Graph convolution layers with residual connections
        x1 = F.elu(self.conv1(x, edge_index, edge_attr))
        x2 = F.elu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.elu(self.conv3(x2, edge_index, edge_attr))
        
        # Add residual connections (now all tensors have same size)
        node_embeddings = x1 + x2 + x3
        
        # Global pooling if batch information is provided
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
        
        return node_embeddings, graph_embedding
    
    def predict_energy(self, node_embeddings, node_features):
        # Concatenate node embeddings with original features for energy prediction
        combined = torch.cat([node_embeddings, node_features], dim=1)
        energy_pred = self.energy_predictor(combined)
        return energy_pred
    
    def score_route(self, path_embeddings):
        # Score the entire path based on node embeddings
        path_embedding = path_embeddings.mean(dim=0, keepdim=True)
        return self.route_scorer(path_embedding)

############################## Q-Network for DRL ##############################

class QNetwork(nn.Module):
    """Deep Q-Network for WSN routing decisions enhanced with GNN embeddings"""
    
    def __init__(self, state_size, gnn_embedding_size, hidden_size):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.gnn_embedding_size = gnn_embedding_size
        
        # Adjust network architecture with correct dimensions
        self.state_encoder = nn.Linear(state_size, hidden_size)
        self.gnn_encoder = nn.Linear(gnn_embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Concatenated hidden states
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, state, gnn_embedding):
        # Ensure input dimensions are correct
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(gnn_embedding.shape) == 1:
            gnn_embedding = gnn_embedding.unsqueeze(0)
        
        # Process state and GNN embedding separately
        state_encoded = F.relu(self.state_encoder(state))
        gnn_encoded = F.relu(self.gnn_encoder(gnn_embedding))
        
        # Combine encoded features
        combined = torch.cat([state_encoded, gnn_encoded], dim=1)
        
        # Pass through remaining layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

############################## Experience Replay ##############################

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', 
                        ['state', 'gnn_embedding', 'action', 'reward', 
                         'next_state', 'next_gnn_embedding', 'done'])

class ReplayMemory:
    """Experience replay buffer to store and sample transitions"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, gnn_embedding, action, reward, next_state, next_gnn_embedding, done):
        """Add a new experience to memory"""
        self.memory.append(Experience(state, gnn_embedding, action, reward, 
                                     next_state, next_gnn_embedding, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

############################## Enhanced DRL Agent ##############################

class ProactiveDRLAgent:
    """Enhanced DRL Agent with GNN for WSN routing optimization"""
    
    def __init__(self, state_size, gnn_embedding_size, hidden_size):
        self.state_size = state_size
        self.gnn_embedding_size = gnn_embedding_size
        
        # Initialize Q networks (online and target)
        self.q_network = QNetwork(state_size, gnn_embedding_size, hidden_size).to(device)
        self.target_network = QNetwork(state_size, gnn_embedding_size, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Initialize replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Initialize exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_end = EPSILON_END
        
        # Initialize step counter
        self.t_step = 0
    
    def get_state(self, current_node, network, sink_pos):
        """Extract state features from node and network"""
        # Normalize values to [0,1] range for better learning
        normalized_energy = current_node['E'] / INITIAL_ENERGY_MAX
        normalized_x = current_node['x'] / FIELD_X
        normalized_y = current_node['y'] / FIELD_Y
        normalized_dist_to_sink = np.sqrt((current_node['x'] - sink_pos[0])**2 + 
                                        (current_node['y'] - sink_pos[1])**2) / np.sqrt(FIELD_X**2 + FIELD_Y**2)
        normalized_hop_count = min(1.0, current_node.get('hop', 0) / (NUM_NODES/2))
        
        # Calculate percentage of remaining network energy
        total_energy = sum(node['E'] for node in network if isinstance(node, dict) and node.get('cond', 1) == 1)
        max_possible_energy = NUM_NODES * INITIAL_ENERGY_MAX
        network_energy_percentage = total_energy / max_possible_energy
        
        # Calculate congestion (based on number of active nodes in proximity)
        proximity_nodes = sum(1 for node in network 
                             if isinstance(node, dict) and 
                             node.get('cond', 1) == 1 and
                             np.sqrt((node.get('x', 0) - current_node['x'])**2 + 
                                    (node.get('y', 0) - current_node['y'])**2) <= TRANSMISSION_RANGE)
        normalized_congestion = proximity_nodes / NUM_NODES
        
        # Return state as tensor
        state = torch.tensor([
            normalized_energy,
            normalized_x,
            normalized_y,
            normalized_dist_to_sink,
            normalized_hop_count,
            network_energy_percentage,
            normalized_congestion
        ], dtype=torch.float32).to(device)
        
        return state.unsqueeze(0)  # Add batch dimension
    
    def get_action(self, state, gnn_embedding, available_nodes, network_graph=None, source_node=None):
        """Select action (next hop node) using epsilon-greedy policy with proactive considerations"""
        if not available_nodes:
            return None
            
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select random next hop
            return random.choice(available_nodes)
        else:
            # Exploit: select node with highest Q-value
            with torch.no_grad():
                q_values = []
                # Create network graph to get node mappings
                G = create_network_graph(available_nodes + [source_node], (SINK_X, SINK_Y), TRANSMISSION_RANGE)
                pyg_data, node_map = graph_to_pyg_data(G, available_nodes + [source_node])
                
                for node in available_nodes:
                    node_state = self.get_state(node, available_nodes, (SINK_X, SINK_Y))
                    # Get node's GNN embedding using the correct mapping
                    node_mapped_id = node_map.get(node['id'], 0)
                    if node_mapped_id < gnn_embedding.size(0):
                        node_gnn_embedding = gnn_embedding[node_mapped_id].unsqueeze(0)
                        q_value = self.q_network(node_state, node_gnn_embedding)
                        
                        # Factor in proactive considerations if network graph is provided
                        proactive_bonus = 0
                        if network_graph and source_node:
                            # Calculate how critical this node is to network connectivity
                            try:
                                # Check if this node is a cut vertex
                                G = network_graph.copy()
                                G.remove_node(node['id'])
                                if source_node['id'] in G.nodes and NUM_NODES in G.nodes:
                                    paths = nx.has_path(G, source_node['id'], NUM_NODES)
                                    if not paths:
                                        proactive_bonus -= 0.5  # Penalize choosing critical cut vertices
                                
                                # Add bonus for nodes with higher predicted future energy
                                if 'predicted_energy' in node:
                                    energy_ratio = node['predicted_energy'] / node['E'] if node['E'] > 0 else 0
                                    proactive_bonus += energy_ratio * 0.2
                            except:
                                pass
                        
                        q_values.append((node, q_value.item() + proactive_bonus))
                
                # Select node with highest adjusted Q-value
                if q_values:
                    return max(q_values, key=lambda x: x[1])[0]
                return random.choice(available_nodes)  # Fallback to random choice if no valid Q-values
    
    def update_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        """Update model weights based on batch of experiences"""
        # Check if enough samples in memory
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample random batch from memory
        experiences = self.memory.sample(BATCH_SIZE)
        
        # Convert batch to tensors
        states = torch.cat([e.state for e in experiences])
        gnn_embeddings = torch.cat([e.gnn_embedding for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).unsqueeze(-1).to(device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).unsqueeze(-1).to(device)
        next_states = torch.cat([e.next_state for e in experiences])
        next_gnn_embeddings = torch.cat([e.next_gnn_embedding for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).unsqueeze(-1).to(device)
        
        # Get expected Q values
        q_expected = self.q_network(states, gnn_embeddings)
        
        # Get next Q values from target network
        with torch.no_grad():
            q_targets_next = self.target_network(next_states, next_gnn_embeddings)
            q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Clip gradients
        self.optimizer.step()
        
        # Update target network
        self.t_step += 1
        if self.t_step % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()

############################## Network Graph Builder ##############################

def create_network_graph(network, sink_pos, transmission_range):
    """Construct a networkx graph from the WSN topology"""
    G = nx.Graph()
    
    # Add all alive nodes first
    for node in network:
        if isinstance(node, dict) and node.get('cond', 1) == 1:
            G.add_node(node['id'], 
                      pos=(node['x'], node['y']),
                      energy=node['E'],
                      role=node.get('role', 0),
                      energy_ratio=node['E']/node['Eo'])
    
    # Add sink node
    G.add_node(NUM_NODES, pos=sink_pos, energy=float('inf'), role=2, energy_ratio=1.0)
    
    # Add edges between nodes within transmission range
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        node1_id = nodes[i]
        node1_pos = G.nodes[node1_id]['pos']
        
        for j in range(i + 1, len(nodes)):
            node2_id = nodes[j]
            node2_pos = G.nodes[node2_id]['pos']
            
            distance = np.sqrt((node1_pos[0] - node2_pos[0])**2 + 
                             (node1_pos[1] - node2_pos[1])**2)
            
            if distance <= transmission_range:
                # Energy-weighted distance as edge weight
                energy_factor = min(G.nodes[node1_id]['energy'], 
                                  G.nodes[node2_id]['energy']) / INITIAL_ENERGY_MAX
                weight = distance * (1 + (1 - energy_factor))
                G.add_edge(node1_id, node2_id,
                          weight=weight,
                          distance=distance,
                          energy_ratio=energy_factor)
    
    return G

# Update the global sink position
SINK_POS = (SINK_X, SINK_Y)

def graph_to_pyg_data(G, network):
    """Convert networkx graph to PyTorch Geometric Data format"""
    # Create edge index
    edge_index = []
    edge_attr = []
    
    # First create node mapping to ensure consecutive indices
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    
    for edge in G.edges(data=True):
        source, target, data = edge
        # Use mapped indices
        edge_index.append([node_map[source], node_map[target]])
        edge_index.append([node_map[target], node_map[source]])  # Add reverse edge for undirected graph
        
        # Edge features: [distance, energy_ratio, signal_strength]
        distance = data.get('distance', 0)
        energy_ratio = data.get('energy_ratio', 1.0)
        # Simple signal strength model based on distance
        signal_strength = max(0, 1 - (distance / TRANSMISSION_RANGE))
        
        edge_features = [distance / TRANSMISSION_RANGE, energy_ratio, signal_strength]
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)  # Same features for reverse edge
    
    # If there are no edges, handle it specially
    if not edge_index:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_SIZE), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create node features
    x = []
    node_map = {}  # Map from original node ids to new consecutive ids
    
    # Process network nodes
    for i, node_id in enumerate(G.nodes()):
        node_map[node_id] = i
        
        if node_id == NUM_NODES:  # Sink node (previously -1)
            # Features for sink: [1.0 energy, position, role=2, etc.]
            x.append([
                1.0,  # Energy ratio
                SINK_POS[0] / FIELD_X,  # Normalized x
                SINK_POS[1] / FIELD_Y,  # Normalized y
                0.0,  # Distance to sink = 0
                0.0,  # Hop count = 0
                1.0,  # Network energy (irrelevant for sink)
                0.0   # Congestion (irrelevant for sink)
            ])
        else:
            # Find the node in network
            node = next((n for n in network if n['id'] == node_id), None)
            if node:
                normalized_energy = node['E'] / INITIAL_ENERGY_MAX
                normalized_x = node['x'] / FIELD_X
                normalized_y = node['y'] / FIELD_Y
                normalized_dist_to_sink = node['dts'] / np.sqrt(FIELD_X**2 + FIELD_Y**2)
                normalized_hop_count = node['hop'] / (FIELD_X/TRANSMISSION_RANGE)
                
                # Network energy percentage
                total_energy = sum(n['E'] for n in network if n['cond'] == 1)
                max_possible_energy = NUM_NODES * INITIAL_ENERGY_MAX
                network_energy_percentage = total_energy / max_possible_energy
                
                # Congestion
                proximity_nodes = sum(1 for n in network if n['cond'] == 1 and
                                     np.sqrt((n['x'] - node['x'])**2 + 
                                            (n['y'] - node['y'])**2) <= TRANSMISSION_RANGE)
                normalized_congestion = proximity_nodes / NUM_NODES
                
                x.append([
                    normalized_energy,
                    normalized_x,
                    normalized_y,
                    normalized_dist_to_sink,
                    normalized_hop_count,
                    network_energy_percentage,
                    normalized_congestion
                ])
    
    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_map

############################## WSN Functions ##############################

def initialize_network(num_nodes, field_x, field_y, sink_x, sink_y, range_c):
    """Initialize WSN with randomly placed nodes"""
    network = []
    
    # Create sensor nodes
    for i in range(num_nodes):
        node = {}
        node['id'] = i
        node['x'] = random.uniform(0, field_x)
        node['y'] = random.uniform(0, field_y)
        node['E'] = random.uniform(INITIAL_ENERGY_MIN, INITIAL_ENERGY_MAX)  # Random initial energy
        node['Eo'] = node['E']  # Store original energy
        node['cond'] = 1  # 1=alive, 0=dead
        # Calculate distance to sink
        node['dts'] = np.sqrt((sink_x - node['x'])**2 + (sink_y - node['y'])**2)
        # Estimate hop count to sink
        node['hop'] = np.ceil(node['dts'] / range_c)
        node['role'] = 0  # 0=regular node, 1=cluster head
        node['closest'] = 0
        node['cluster'] = None
        node['prev'] = 0
        node['traffic'] = 0  # Track traffic through this node
        # Initialize energy prediction
        node['predicted_energy'] = node['E']
        # Energy consumption history
        node['energy_history'] = []
        network.append(node)
    
    return network

def select_cluster_heads(network, ch_percentage, energy_predictions=None):
    """Select cluster heads based on energy, position, and predicted future energy"""
    num_ch = int(len([n for n in network if n['cond'] == 1]) * ch_percentage)
    
    # Score nodes based on current energy, position, and predicted future energy
    for node in network:
        if node['cond'] == 1:  # Only consider alive nodes
            energy_factor = node['E'] / node['Eo']  # Normalized remaining energy
            position_factor = 1 - (node['dts'] / np.sqrt(FIELD_X**2 + FIELD_Y**2))  # Relative position
            
            # Consider predicted future energy if available
            future_energy_factor = 0
            if energy_predictions is not None and node['id'] in energy_predictions:
                # Use the average of predicted energy levels
                future_energies = energy_predictions[node['id']]
                if len(future_energies) > 0:
                    avg_future_energy = sum(future_energies) / len(future_energies)
                    future_energy_factor = avg_future_energy / node['Eo']
            
            # Traffic factor - prefer nodes with lower historical traffic
            traffic_factor = 0
            if hasattr(node, 'traffic') and node['traffic'] > 0:
                traffic_factor = 1 - min(1, node['traffic'] / 100)  # Normalize traffic
            
            # Combined score
            node['score'] = (0.4 * energy_factor + 
                            0.2 * position_factor + 
                            0.3 * future_energy_factor +
                            0.1 * traffic_factor)
    
    # Sort by score and select top nodes as CHs
    alive_nodes = [node for node in network if node['cond'] == 1]
    sorted_nodes = sorted(alive_nodes, key=lambda x: x['score'], reverse=True)
    
    # Reset all roles first
    for node in network:
        if node['cond'] == 1:
            node['role'] = 0
    
    # Assign CH roles to top nodes
    ch_count = 0
    cluster_heads = []
    
    # Make sure we have at least 1 cluster head if there are any alive nodes
    num_ch = max(1, num_ch) if alive_nodes else 0
    
    for node in sorted_nodes:
        if ch_count < num_ch:
            node['role'] = 1  # Set as cluster head
            cluster_heads.append(node)
            ch_count += 1
            
    # Ensure proper coverage of the network
    if cluster_heads:
        # Create a graph to check connectivity
        G = nx.Graph()
        for ch in cluster_heads:
            G.add_node(ch['id'], pos=(ch['x'], ch['y']))
        
        # Add sink
        G.add_node(NUM_NODES, pos=(SINK_X, SINK_Y))
        
        # Connect CHs based on transmission range
        for ch1 in cluster_heads:
            # Connect to sink if possible
            dist_to_sink = np.sqrt((ch1['x'] - SINK_X)**2 + (ch1['y'] - SINK_Y)**2)
            if dist_to_sink <= TRANSMISSION_RANGE:
                G.add_edge(ch1['id'], NUM_NODES)
            
            # Connect to other CHs
            for ch2 in cluster_heads:
                if ch1['id'] != ch2['id']:
                    dist = np.sqrt((ch1['x'] - ch2['x'])**2 + (ch1['y'] - ch2['y'])**2)
                    if dist <= TRANSMISSION_RANGE:
                        G.add_edge(ch1['id'], ch2['id'])
        
        # Check if we need additional CHs to ensure sink connectivity
        # Find components not connected to sink
        if NUM_NODES in G.nodes:
            sink_component = nx.node_connected_component(G, NUM_NODES)
            disconnected_chs = [ch for ch in cluster_heads if ch['id'] not in sink_component]
            
            if disconnected_chs:
                # Add more CHs to connect disconnected components
                for extra_node in sorted_nodes:
                    if extra_node['role'] == 0:  # Not already a CH
                        # Check if this node can connect a disconnected CH to the sink
                        can_connect = False
                        for ch in disconnected_chs:
                            ch_dist = np.sqrt((extra_node['x'] - ch['x'])**2 + (extra_node['y'] - ch['y'])**2)
                            sink_dist = np.sqrt((extra_node['x'] - SINK_X)**2 + (extra_node['y'] - SINK_Y)**2)
                            
                            if ch_dist <= TRANSMISSION_RANGE and sink_dist <= TRANSMISSION_RANGE:
                                can_connect = True
                                break
                        
                        if can_connect:
                            extra_node['role'] = 1  # Make this node a CH
                            cluster_heads.append(extra_node)
                            ch_count += 1
                            # Update graph and check connectivity again
                            G.add_node(extra_node['id'], pos=(extra_node['x'], extra_node['y']))
                            
                            # Connect to other CHs and sink
                            if sink_dist <= TRANSMISSION_RANGE:
                                G.add_edge(extra_node['id'], NUM_NODES)
                            
                            for other_ch in cluster_heads:
                                if other_ch['id'] != extra_node['id']:
                                    dist = np.sqrt((extra_node['x'] - other_ch['x'])**2 + 
                                                  (extra_node['y'] - other_ch['y'])**2)
                                    if dist <= TRANSMISSION_RANGE:
                                        G.add_edge(extra_node['id'], other_ch['id'])
                            
                            # Break if we've added too many extra CHs
                            if ch_count >= 1.5 * num_ch:
                                break
    
    return cluster_heads

def form_clusters(network, cluster_heads):
    """Assign nodes to their nearest cluster head with proactive considerations"""
    # Reset cluster assignments
    for node in network:
        if node['cond'] == 1 and node['role'] == 0:
            node['cluster'] = None
            
            # Track candidates and their scores
            candidates = []
            
            # Find all reachable cluster heads
            for ch in cluster_heads:
                dist = np.sqrt((node['x'] - ch['x'])**2 + (node['y'] - ch['y'])**2)
                if dist <= TRANSMISSION_RANGE:
                    # Score based on distance, energy, and load balancing
                    distance_score = 1 - (dist / TRANSMISSION_RANGE)
                    energy_score = ch['E'] / ch['Eo']
                    
                    # Load balancing - prefer CHs with fewer members
                    current_members = sum(1 for n in network 
                                        if n['cond'] == 1 and n['cluster'] == ch['id'])
                    load_score = 1 - (current_members / NUM_NODES)
                    
                    # Traffic consideration
                    traffic_score = 1 - min(1, ch.get('traffic', 0) / 100)
                    
                    # Predicted energy consideration
                    future_energy_score = ch.get('predicted_energy', ch['E']) / ch['Eo']
                    
                    # Combined score
                    total_score = (0.3 * distance_score + 
                                  0.25 * energy_score + 
                                  0.2 * load_score +
                                  0.15 * traffic_score +
                                  0.1 * future_energy_score)
                    
                    candidates.append((ch, dist, total_score))
            
            # Assign to the best cluster head
            if candidates:
                best_ch = max(candidates, key=lambda x: x[2])[0]
                node['cluster'] = best_ch['id']
                node['closest'] = best_ch['id']


def predict_traffic_patterns(network, traffic_history, window_size=TRAFFIC_PREDICTION_WINDOW):
    """Predict future traffic patterns based on historical data"""
    predictions = {}
    
    for node in network:
        if node['cond'] == 1 and node['id'] in traffic_history:
            node_history = traffic_history[node['id']]
            
            if len(node_history) >= window_size:
                # Simple moving average prediction
                recent_traffic = node_history[-window_size:]
                predicted_traffic = sum(recent_traffic) / len(recent_traffic)
                predictions[node['id']] = predicted_traffic
            else:
                # Use current traffic if not enough history
                predictions[node['id']] = node.get('traffic', 0)
    
    return predictions

def calculate_energy_consumption(from_node, to_node, packet_size):
    """Calculate energy consumption for transmitting packets between nodes"""
    distance = np.sqrt((from_node['x'] - to_node['x'])**2 + 
                      (from_node['y'] - to_node['y'])**2)
    
    # Energy for transmission (ETX) and reception (ERX)
    etx = E_ELEC * packet_size + E_AMP * packet_size * (distance ** 2)
    erx = E_ELEC * packet_size
    
    return etx, erx

def find_optimal_path_drl(source, sink_pos, network, agent, gnn_model, network_graph=None):
    """Find optimal path using DRL agent with GNN embeddings"""
    path = [source]
    current_node = source
    visited = set([source['id']])
    total_energy_consumed = 0
    
    # Create initial network graph if not provided
    if network_graph is None:
        network_graph = create_network_graph(network, sink_pos, TRANSMISSION_RANGE)
    
    # Get initial GNN embeddings
    pyg_data, node_map = graph_to_pyg_data(network_graph, network)
    
    with torch.no_grad():
        gnn_model.eval()
        node_embeddings, _ = gnn_model(pyg_data.x.to(device), 
                                     pyg_data.edge_index.to(device),
                                     pyg_data.edge_attr.to(device))
    
    while True:
        # Check if sink is reachable
        distance_to_sink = np.sqrt((current_node['x'] - sink_pos[0])**2 + 
                                  (current_node['y'] - sink_pos[1])**2)
        
        if distance_to_sink <= TRANSMISSION_RANGE:
            # Can reach sink directly
            break
            
        # Find available next hop candidates
        available_nodes = []
        for node in network:
            if (node['cond'] == 1 and 
                node['id'] not in visited and 
                node['id'] != current_node['id']):
                
                distance = np.sqrt((current_node['x'] - node['x'])**2 + 
                                 (current_node['y'] - node['y'])**2)
                if distance <= TRANSMISSION_RANGE:
                    available_nodes.append(node)
        
        if not available_nodes:
            # No available next hop, routing failed
            return None, float('inf')
        
        # Get current network state as graph
        G = create_network_graph(network, sink_pos, TRANSMISSION_RANGE)
        pyg_data, node_map = graph_to_pyg_data(G, network)
        
        # Get GNN embeddings
        with torch.no_grad():
            gnn_model.eval()
            node_embeddings, _ = gnn_model(pyg_data.x.to(device), 
                                         pyg_data.edge_index.to(device),
                                         pyg_data.edge_attr.to(device))
        
        # Get current state and GNN embedding
        current_state = agent.get_state(current_node, network, sink_pos)
        current_node_mapped_id = node_map.get(current_node['id'], 0)
        current_gnn_embedding = node_embeddings[current_node_mapped_id].unsqueeze(0)
        
        # Get next hop from DRL agent
        next_node = agent.get_action(current_state, current_gnn_embedding, 
                                   available_nodes, network_graph, current_node)
        
        if next_node is None:
            # No valid action, routing failed
            return None, float('inf')
        
        # Calculate energy consumption
        etx, erx = calculate_energy_consumption(current_node, next_node, PACKET_SIZE)
        total_energy_consumed += etx + erx
        
        # Update traffic counters
        current_node['traffic'] = current_node.get('traffic', 0) + 1
        next_node['traffic'] = next_node.get('traffic', 0) + 1
        
        # Move to next node
        path.append(next_node)
        current_node = next_node
        visited.add(current_node['id'])
        
        # Prevent infinite loops
        if len(path) > NUM_NODES:
            return None, float('inf')
    
    return path, total_energy_consumed

def update_energy_after_transmission(network, path, energy_consumed):
    """Update node energies after packet transmission"""
    if not path:
        return
    
    # Distribute energy consumption among path nodes
    energy_per_hop = energy_consumed / len(path)
    
    for node in path:
        # Find node in network and update energy
        for net_node in network:
            if net_node['id'] == node['id']:
                net_node['E'] = max(0, net_node['E'] - energy_per_hop)
                # Update energy history for prediction
                net_node['energy_history'].append(net_node['E'])
                # Keep only recent history
                if len(net_node['energy_history']) > 20:
                    net_node['energy_history'] = net_node['energy_history'][-20:]
                break

def train_gnn_model(gnn_model, gnn_optimizer, network_history, energy_targets):
    """Train the GNN model on historical network data"""
    if len(network_history) < 2:
        return None
    
    gnn_model.train()
    total_loss = 0
    num_batches = 0
    
    # Create training batches from historical data
    for i in range(len(network_history) - 1):
        current_network = network_history[i]
        next_network = network_history[i + 1]
        
        # Create graph for current state
        G = create_network_graph(current_network, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
        if len(G.nodes) == 0:
            continue
            
        pyg_data, node_map = graph_to_pyg_data(G, current_network)
        
        # Get node embeddings and predictions
        node_embeddings, node_predictions = gnn_model(pyg_data.x.to(device), 
                                                     pyg_data.edge_index.to(device),
                                                     pyg_data.edge_attr.to(device))
        
        # Create target energies for next time step
        target_energies = []
        for node_id in node_map.keys():
            if node_id == NUM_NODES:  # Skip sink
                continue
            next_node = next((n for n in next_network if n['id'] == node_id), None)
            if next_node:
                target_energies.append(next_node['E'] / INITIAL_ENERGY_MAX)
        
        if len(target_energies) == 0:
            continue
            
        target_energies = torch.tensor(target_energies, dtype=torch.float32).to(device)
        
        # Predict future energy levels
        node_features = pyg_data.x[:len(target_energies)]  # Exclude sink
        energy_predictions = gnn_model.predict_energy(node_embeddings[:len(target_energies)], 
                                                     node_features)
        
        # Calculate loss (predict next round energy)
        if energy_predictions.shape[0] == target_energies.shape[0]:
            loss = F.mse_loss(energy_predictions[:, 0], target_energies)
            
            # Backpropagation
            gnn_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            gnn_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else None

def predict_future_energies(gnn_model, network, prediction_horizon=PREDICTION_HORIZON):
    """Predict future energy levels for all nodes"""
    gnn_model.eval()
    predictions = {}
    # Create current network graph
    G = create_network_graph(network, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
    if len(G.nodes) == 0:
        return predictions
    pyg_data, node_map = graph_to_pyg_data(G, network)
    with torch.no_grad():
        # Get embeddings
        node_embeddings, _ = gnn_model(pyg_data.x.to(device), 
                                     pyg_data.edge_index.to(device),
                                     pyg_data.edge_attr.to(device))
        # Predict future energies
        for node_id, mapped_id in node_map.items():
            if node_id == NUM_NODES:  # Skip sink
                continue
            node = next((n for n in network if n['id'] == node_id), None)
            if not node:
                continue
            node_features = pyg_data.x[mapped_id].unsqueeze(0)
            node_embedding = node_embeddings[mapped_id].unsqueeze(0)
            energy_pred = gnn_model.predict_energy(node_embedding, node_features)
            # Use the predicted value for all future steps (since model is single-step)
            pred_energy = energy_pred[0, 0].item() * INITIAL_ENERGY_MAX
            predicted_energies = [max(0, pred_energy)] * prediction_horizon
            predictions[node_id] = predicted_energies
            
            # Update node's predicted energy (average of future predictions)
            node['predicted_energy'] = sum(predicted_energies) / len(predicted_energies)
    
    return predictions

def calculate_reward(path, energy_consumed, network, sink_pos):
    """Calculate reward for DRL agent based on path quality and proactive metrics"""
    if not path:
        return -100  # Large penalty for failed routing
    
    # Base reward components
    energy_efficiency = 1 / (energy_consumed + 1e-6)  # Higher reward for lower energy consumption
    path_length_penalty = len(path) / NUM_NODES  # Penalty for longer paths
    
    # Network connectivity preservation
    connectivity_bonus = 0
    total_energy = sum(node['E'] for node in network if node['cond'] == 1)
    network_energy_ratio = total_energy / (NUM_NODES * INITIAL_ENERGY_MAX)
    connectivity_bonus = network_energy_ratio * 10
    
    # Load balancing reward
    load_balance_reward = 0
    if len(path) > 1:
        # Check if path uses diverse routes (less congested nodes)
        avg_traffic = sum(node.get('traffic', 0) for node in path) / len(path)
        max_traffic = max(node.get('traffic', 0) for node in network if node['cond'] == 1)
        if max_traffic > 0:
            load_balance_reward = (1 - avg_traffic / max_traffic) * 5
    
    # Future viability reward (based on predicted energies)
    future_viability_reward = 0
    for node in path:
        predicted_energy = node.get('predicted_energy', node['E'])
        current_energy = node['E']
        if current_energy > 0:
            future_ratio = predicted_energy / current_energy
            future_viability_reward += future_ratio
    future_viability_reward = (future_viability_reward / len(path)) * 5
    
    # Distance to sink factor
    final_node = path[-1]
    distance_to_sink = np.sqrt((final_node['x'] - sink_pos[0])**2 + 
                              (final_node['y'] - sink_pos[1])**2)
    distance_reward = (TRANSMISSION_RANGE - distance_to_sink) / TRANSMISSION_RANGE * 2
    
    # Combine all reward components
    total_reward = (10 * energy_efficiency +
                   connectivity_bonus +
                   load_balance_reward +
                   future_viability_reward +
                   distance_reward -
                   5 * path_length_penalty)
    
    return total_reward

def check_node_failures(network):
    """Check for node failures based on energy levels"""
    dead_nodes = 0
    for node in network:
        if node['cond'] == 1 and node['E'] <= DEAD_NODE_THRESHOLD:
            node['cond'] = 0  # Mark as dead
            dead_nodes += 1
    return dead_nodes

def calculate_network_metrics(network, round_num, agent_epsilon=None):
    """Calculate various network performance metrics"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    cluster_heads = [node for node in alive_nodes if node['role'] == 1]
    
    metrics = {
        'round': round_num,
        'alive_nodes': len(alive_nodes),
        'alive_percentage': len(alive_nodes) / NUM_NODES * 100 if NUM_NODES > 0 else 0,
        'total_energy': sum(node['E'] for node in alive_nodes),
        'avg_energy': sum(node['E'] for node in alive_nodes) / len(alive_nodes) if alive_nodes else 0,
        'min_energy': min(node['E'] for node in alive_nodes) if alive_nodes else 0,
        'max_energy': max(node['E'] for node in alive_nodes) if alive_nodes else 0,
        'energy_variance': np.var([node['E'] for node in alive_nodes]) if alive_nodes else 0,
        'total_traffic': sum(node.get('traffic', 0) for node in network),
        'avg_traffic': sum(node.get('traffic', 0) for node in network) / NUM_NODES if NUM_NODES > 0 else 0,
        'num_cluster_heads': len(cluster_heads),
        'drl_epsilon': agent_epsilon # Add DRL epsilon to metrics
    }
    
    # Calculate network connectivity
    G = create_network_graph(network, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
    if len(G.nodes) > 1:
        metrics['connectivity'] = nx.is_connected(G)
        if NUM_NODES in G.nodes:
            # Calculate how many nodes can reach the sink
            reachable_to_sink = 0
            for node in alive_nodes:
                if nx.has_path(G, node['id'], NUM_NODES):
                    reachable_to_sink += 1
            metrics['sink_connectivity'] = reachable_to_sink / len(alive_nodes) * 100
    else:
        metrics['connectivity'] = False
        metrics['sink_connectivity'] = 0
    
    return metrics

def visualize_network(network, round_num, sink_pos, save_plot=True):
    """Visualize the current network state"""
    plt.figure(figsize=(12, 10))
    
    # Plot alive nodes
    alive_nodes = [node for node in network if node['cond'] == 1]
    dead_nodes = [node for node in network if node['cond'] == 0]
    
    # Color nodes based on energy level
    if alive_nodes:
        alive_x = [node['x'] for node in alive_nodes]
        alive_y = [node['y'] for node in alive_nodes]
        alive_energies = [node['E'] / node['Eo'] for node in alive_nodes]
        
        scatter = plt.scatter(alive_x, alive_y, c=alive_energies, 
                            cmap='RdYlGn', s=60, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(scatter, label='Energy Ratio')
        
        # Mark cluster heads
        ch_nodes = [node for node in alive_nodes if node['role'] == 1]
        if ch_nodes:
            ch_x = [node['x'] for node in ch_nodes]
            ch_y = [node['y'] for node in ch_nodes]
            plt.scatter(ch_x, ch_y, c='blue', marker='s', s=100, 
                       alpha=0.8, label='Cluster Heads')
    
    # Plot dead nodes
    if dead_nodes:
        dead_x = [node['x'] for node in dead_nodes]
        dead_y = [node['y'] for node in dead_nodes]
        plt.scatter(dead_x, dead_y, c='black', marker='x', s=60, 
                   alpha=0.8, label='Dead Nodes')
    
    # Plot sink
    plt.scatter(sink_pos[0], sink_pos[1], c='red', marker='*', s=200, 
               label='Sink', edgecolors='black', linewidth=2)
    
    # Add network statistics as text
    alive_count = len(alive_nodes)
    total_energy = sum(node['E'] for node in alive_nodes)
    avg_energy = total_energy / alive_count if alive_count > 0 else 0
    
    stats_text = f'Round: {round_num}\n'
    stats_text += f'Alive Nodes: {alive_count}/{NUM_NODES}\n'
    stats_text += f'Avg Energy: {avg_energy:.3f}J\n'
    stats_text += f'Total Energy: {total_energy:.3f}J'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'WSN Topology - Round {round_num}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, FIELD_X)
    plt.ylim(0, FIELD_Y)
    
    if save_plot:
        plt.savefig(f'results/network_round_{round_num:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

############################## Main Simulation ##############################

def run_proactive_gnn_wsn_simulation():
    """Run the main WSN simulation with proactive GNN-DRL routing"""
    print("="*70)
    print("PROACTIVE GNN-DRL WSN ROUTING SIMULATION")
    print("="*70)
    
    # Initialize network
    print("Initializing WSN...")
    network = initialize_network(NUM_NODES, FIELD_X, FIELD_Y, SINK_X, SINK_Y, TRANSMISSION_RANGE)
    sink_pos = (SINK_X, SINK_Y)
    
    # Initialize GNN model
    print("Initializing GNN model...")
    gnn_model = WSN_GNN(NODE_FEATURE_SIZE, EDGE_FEATURE_SIZE, 
                       GNN_HIDDEN_CHANNELS, GNN_OUTPUT_SIZE).to(device)
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=LEARNING_RATE)
    
    # Initialize DRL agent with correct embedding size
    print("Initializing DRL agent...")
    state_size = 7  # Updated to match the actual state features
    gnn_embedding_size = GNN_HIDDEN_CHANNELS  # Use the hidden channel size as embedding size
    agent = ProactiveDRLAgent(state_size, gnn_embedding_size, GNN_HIDDEN_CHANNELS)
    
    # Initialize tracking variables
    network_history = []
    traffic_history = {node['id']: [] for node in network}
    metrics_history = []
    gnn_losses = []
    drl_losses = []
    energy_predictions_history = []
    
    # Initialize round counter
    round_num = 0
    
    print(f"Starting simulation with {NUM_NODES} nodes...")
    print(f"Initial network energy: {sum(node['E'] for node in network):.3f}J")
    
    # Simulation main loop
    with tqdm(total=MAX_ROUNDS, desc="Simulation Progress") as pbar:
        while round_num < MAX_ROUNDS:
            round_num += 1
            pbar.update(1)
            
            # Check for node failures
            dead_nodes = check_node_failures(network)
            alive_nodes = [node for node in network if node['cond'] == 1]
            
            if len(alive_nodes) == 0:
                print(f"\nAll nodes dead at round {round_num}")
                break
            
            # Store network state for GNN training
            network_history.append([node.copy() for node in network])
            if len(network_history) > 50:  # Keep only recent history
                network_history = network_history[-50:]
            
            # Predict future energy levels using GNN (every few rounds)
            energy_predictions = {}
            if round_num % 5 == 0 and len(network_history) >= 2:
                energy_predictions = predict_future_energies(gnn_model, network)
                energy_predictions_history.append({
                    'round': round_num,
                    'predictions': energy_predictions.copy()
                })
            
            # Predict traffic patterns
            traffic_predictions = predict_traffic_patterns(network, traffic_history)
            
            # Select cluster heads with proactive considerations
            cluster_heads = select_cluster_heads(network, CH_PERCENTAGE, energy_predictions)
            
            # Form clusters
            form_clusters(network, cluster_heads)
            
            # Simulate data transmission from random nodes
            num_transmissions = min(10, len(alive_nodes))  # Simulate 10 random transmissions
            transmission_count = 0
            
            for _ in range(num_transmissions):
                if transmission_count >= num_transmissions:
                    break
                    
                # Select random source node (non-CH alive node)
                non_ch_nodes = [node for node in alive_nodes if node['role'] == 0]
                if not non_ch_nodes:
                    continue
                    
                source_node = random.choice(non_ch_nodes)
                
                # Create network graph for routing
                network_graph = create_network_graph(network, sink_pos, TRANSMISSION_RANGE)
                
                # Find optimal path using DRL agent
                path, energy_consumed = find_optimal_path_drl(source_node, sink_pos, 
                                                            network, agent, gnn_model, 
                                                            network_graph)
                
                if path is not None and energy_consumed < float('inf'):
                    # Calculate reward for DRL training
                    reward = calculate_reward(path, energy_consumed, network, sink_pos)
                    
                    # Store experience in replay memory (simplified for this example)
                    if len(network_history) >= 2:
                        # Get current state and next state after energy update
                        current_state = agent.get_state(source_node, network, sink_pos)
                        
                        # Get GNN embeddings for current state
                        G = create_network_graph(network, sink_pos, TRANSMISSION_RANGE)
                        pyg_data, node_map = graph_to_pyg_data(G, network)
                        
                        with torch.no_grad():
                            gnn_model.eval()
                            node_embeddings, _ = gnn_model(pyg_data.x.to(device), 
                                                         pyg_data.edge_index.to(device),
                                                         pyg_data.edge_attr.to(device))
                        
                        source_mapped_id = node_map.get(source_node['id'], 0)
                        current_gnn_embedding = node_embeddings[source_mapped_id].unsqueeze(0)
                        
                        # Update network energy
                        update_energy_after_transmission(network, path, energy_consumed)
                        
                        # Get next state after energy update
                        next_state = agent.get_state(source_node, network, sink_pos)
                        
                        # Store experience
                        action_idx = 0  # Simplified action representation
                        done = source_node['E'] <= DEAD_NODE_THRESHOLD
                        
                        agent.memory.push(current_state, current_gnn_embedding, action_idx, 
                                        reward, next_state, current_gnn_embedding, done)
                        
                        # Train DRL agent
                        if len(agent.memory) > BATCH_SIZE and round_num % DRL_TRAINING_INTERVAL == 0:
                            # print(f"\nTraining DRL agent at round {round_num}...")
                            drl_loss = agent.learn()
                            if drl_loss is not None:
                                drl_losses.append(drl_loss)
            
            # Update DRL agent's epsilon
            agent.update_epsilon()
            
            # Train GNN model periodically
            if round_num % GNN_TRAINING_INTERVAL == 0 and len(network_history) >= 5:
                print(f"\nTraining GNN at round {round_num}...")
                gnn_loss = train_gnn_model(gnn_model, gnn_optimizer, network_history, 
                                         energy_predictions)
                if gnn_loss is not None:
                    gnn_losses.append(gnn_loss)
                    print(f"GNN Loss: {gnn_loss:.6f}")
            
            # Calculate and store metrics
            metrics = calculate_network_metrics(network, round_num, agent.epsilon) # Pass agent.epsilon
            metrics_history.append(metrics)
            
            # Print progress every 100 rounds
            if round_num % 100 == 0:
                print(f"\nRound {round_num}:")
                print(f"  Alive nodes: {metrics['alive_nodes']}/{NUM_NODES} ({metrics['alive_percentage']:.1f}%)")
                print(f"  Average energy: {metrics['avg_energy']:.3f}J")
                print(f"  Network connectivity: {metrics['sink_connectivity']:.1f}%")
                print(f"  DRL Epsilon: {agent.epsilon:.3f}")
            
            # Visualize network periodically
            if round_num % 200 == 0:
                visualize_network(network, round_num, sink_pos)
            
            # Early termination if network is severely degraded
            if metrics['alive_percentage'] < 10:
                print(f"\nNetwork severely degraded at round {round_num}. Terminating simulation.")
                break
    
    # Final visualization and results
    print("\n" + "="*70)
    print("SIMULATION COMPLETED")
    print("="*70)
    
    # Final network state
    final_metrics = calculate_network_metrics(network, round_num)
    print(f"Final round: {round_num}")
    print(f"Final alive nodes: {final_metrics['alive_nodes']}/{NUM_NODES} ({final_metrics['alive_percentage']:.1f}%)")
    print(f"Final network energy: {final_metrics['total_energy']:.3f}J")
    print(f"Total data transmissions: {sum(len(history) for history in traffic_history.values())}")
    
    # Save detailed results
    print("\nSaving simulation results...")
    
    # Save metrics to file
    import json
    results_data = {
        'simulation_params': {
            'num_nodes': NUM_NODES,
            'field_size': (FIELD_X, FIELD_Y),
            'transmission_range': TRANSMISSION_RANGE,
            'initial_energy_range': (INITIAL_ENERGY_MIN, INITIAL_ENERGY_MAX),
            'max_rounds': MAX_ROUNDS
        },
        'final_metrics': final_metrics,
        'metrics_history': metrics_history,
        'gnn_losses': gnn_losses,
        'drl_losses': drl_losses,
        'energy_predictions_history': energy_predictions_history
    }
    
    with open('results/simulation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Plot performance metrics
    plot_simulation_results(metrics_history, gnn_losses, drl_losses)
    
    # Final network visualization
    visualize_network(network, round_num, sink_pos, save_plot=True)
    
    print("Results saved in 'results/' directory")
    return metrics_history, network, final_metrics

def plot_simulation_results(metrics_history, gnn_losses, drl_losses):
    """Plot comprehensive simulation results"""
    rounds = [m['round'] for m in metrics_history]
    
    # Create subplots with additional metrics
    fig, axes = plt.subplots(6, 2, figsize=(18, 24)) # Increased rows for more plots
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]
    ax7, ax8 = axes[3]
    ax9, ax10 = axes[4] # New row for plots
    ax11, ax12 = axes[5] # New row for plots
    
    # Plot 1: Network lifetime (alive nodes over time)
    alive_percentages = [m['alive_percentage'] for m in metrics_history]
    ax1.plot(rounds, alive_percentages, 'b-', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Alive Nodes (%)')
    ax1.set_title('Network Lifetime')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy consumption
    avg_energies = [m['avg_energy'] for m in metrics_history]
    total_energies = [m['total_energy'] for m in metrics_history]
    ax2.plot(rounds, avg_energies, 'g-', label='Average Energy', linewidth=2)
    ax2.plot(rounds, total_energies, 'r--', label='Total Energy', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Network Energy Consumption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Network connectivity
    connectivity = [m['sink_connectivity'] for m in metrics_history]
    ax3.plot(rounds, connectivity, 'm-', linewidth=2)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Sink Connectivity (%)')
    ax3.set_title('Network Connectivity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Traffic distribution
    avg_traffic = [m['avg_traffic'] for m in metrics_history]
    ax4.plot(rounds, avg_traffic, 'c-', linewidth=2)
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Average Traffic')
    ax4.set_title('Traffic Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: GNN training loss
    if gnn_losses:
        ax5.plot(range(len(gnn_losses)), gnn_losses, 'y-', linewidth=2)
        ax5.set_xlabel('Training Iterations')
        ax5.set_ylabel('Loss')
        ax5.set_title('GNN Training Loss')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: DRL training loss
    if drl_losses:
        ax6.plot(range(len(drl_losses)), drl_losses, 'k-', linewidth=2)
        ax6.set_xlabel('Training Iterations')
        ax6.set_ylabel('Loss')
        ax6.set_title('DRL Training Loss')
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: First Node Death Time
    first_death = None
    for m in metrics_history:
        if m['alive_nodes'] < NUM_NODES:
            first_death = m['round']
            break
    if first_death:
        ax7.axvline(x=first_death, color='r', linestyle='--', label=f'First Death at Round {first_death}')
    ax7.plot(rounds, [m['alive_nodes'] for m in metrics_history], 'b-', linewidth=2)
    ax7.set_xlabel('Round')
    ax7.set_ylabel('Number of Alive Nodes')
    ax7.set_title('First Node Death Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Network Throughput
    # Calculate throughput as successful packet transmissions per round
    throughput = []
    for m in metrics_history:
        # Calculate total bits transmitted in this round
        bits_per_round = m['total_traffic'] * PACKET_SIZE  # bits per round
        throughput.append(bits_per_round)  # bits per round
    
    ax8.plot(rounds, throughput, 'g-', linewidth=2)
    ax8.set_xlabel('Round')
    ax8.set_ylabel('Throughput (bits/round)')
    ax8.set_title('Network Throughput')
    ax8.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use scientific notation for y-axis
    ax8.grid(True, alpha=0.3)

    # Plot 9: Energy Variance
    energy_variance = [m.get('energy_variance', 0) for m in metrics_history] # Use .get for safety
    ax9.plot(rounds, energy_variance, 'p-', linewidth=2, label='Energy Variance')
    ax9.set_xlabel('Round')
    ax9.set_ylabel('Energy Variance (J^2)')
    ax9.set_title('Energy Variance Over Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot 10: Number of Active Cluster Heads
    # Assuming 'num_cluster_heads' is or will be added to metrics
    num_cluster_heads = [m.get('num_cluster_heads', np.nan) for m in metrics_history] # Use np.nan for missing data
    ax10.plot(rounds, num_cluster_heads, 's-', linewidth=2, label='Active Cluster Heads')
    ax10.set_xlabel('Round')
    ax10.set_ylabel('Number of Cluster Heads')
    ax10.set_title('Active Cluster Heads Over Time')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Plot 11: DRL Epsilon Decay (Placeholder if not directly in metrics_history)
    # This might need to be passed separately or calculated if agent's history is available
    # For now, let's assume it might be added to metrics or we plot a placeholder
    drl_epsilon = [m.get('drl_epsilon', np.nan) for m in metrics_history] # Placeholder
    ax11.plot(rounds, drl_epsilon, 'd-', linewidth=2, label='DRL Epsilon')
    ax11.set_xlabel('Round')
    ax11.set_ylabel('Epsilon Value')
    ax11.set_title('DRL Agent Epsilon Decay')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Plot 12: Node Role Distribution (Placeholder - requires more data from metrics)
    # This would typically be a stacked bar or area chart.
    # For simplicity, if 'roles_distribution' (e.g., {'ch': count, 'member': count}) is in metrics:
    # ch_counts = [m.get('roles_distribution', {}).get('ch', 0) for m in metrics_history]
    # member_counts = [m.get('roles_distribution', {}).get('member', 0) for m in metrics_history]
    # ax12.stackplot(rounds, ch_counts, member_counts, labels=['Cluster Heads', 'Member Nodes'], alpha=0.7)
    ax12.text(0.5, 0.5, 'Node Role Distribution (Placeholder)', horizontalalignment='center', verticalalignment='center', transform=ax12.transAxes)
    ax12.set_xlabel('Round')
    ax12.set_ylabel('Number of Nodes')
    ax12.set_title('Node Role Distribution Over Time')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    global MAX_ROUNDS
    """Main function to run the simulation"""
    print("\nProactive GNN-DRL WSN Routing Simulation")
    print("=" * 50)
    print(f"Configuration:")
    print(f"Number of nodes: {NUM_NODES}")
    print(f"Field size: {FIELD_X}m x {FIELD_Y}m")
    print(f"Transmission range: {TRANSMISSION_RANGE}m")
    print(f"Initial energy range: {INITIAL_ENERGY_MIN}J - {INITIAL_ENERGY_MAX}J")
    print(f"Using device: {device}")
    print("=" * 50)

    # Prompt user for number of rounds
    try:
        user_input = input(f"Enter the number of rounds to execute (default {MAX_ROUNDS}): ")
        num_rounds = int(user_input) if user_input.strip() else MAX_ROUNDS
        if num_rounds <= 0:
            print(f"Invalid input. Using default value: {MAX_ROUNDS}")
            num_rounds = MAX_ROUNDS
    except Exception:
        print(f"Invalid input. Using default value: {MAX_ROUNDS}")
        num_rounds = MAX_ROUNDS
    MAX_ROUNDS = num_rounds

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    try:
        # Run simulation
        metrics_history, final_network, final_metrics = run_proactive_gnn_wsn_simulation()
        
        # Print final summary
        print("\nSimulation Summary:")
        print("=" * 50)
        print(f"Network lifetime: {len(metrics_history)} rounds")
        print(f"Final alive nodes: {final_metrics['alive_nodes']}/{NUM_NODES} ({final_metrics['alive_percentage']:.1f}%)")
        print(f"Final network energy: {final_metrics['total_energy']:.3f}J")
        print(f"Final network connectivity: {final_metrics['sink_connectivity']:.1f}%")
        print("=" * 50)
        
        # Save simulation timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open('results/simulation_timestamp.txt', 'w') as f:
            f.write(f"Simulation completed at: {timestamp}\n")
            f.write(f"Network lifetime: {len(metrics_history)} rounds\n")
        
        print("\nSimulation completed successfully!")
        print("Results saved in 'results' directory")
        
        # Print node information table
        print_node_table(final_network)
        
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        raise

def print_node_table(network):
    """Print a formatted table with node information"""
    # Header
    print("\nNode Information Table")
    print("=" * 100)
    print("{:^6} | {:^10} | {:^12} | {:^12} | {:^10} | {:^8} | {:^12} | {:^15}".format(
        "ID", "Status", "Energy (J)", "Energy (%)", "Role", "Traffic", "Position", "Dist to Sink"
    ))
    print("-" * 100)
    
    # Sort nodes by ID for better readability
    sorted_nodes = sorted(network, key=lambda x: x['id'])
    
    for node in sorted_nodes:
        # Calculate energy percentage
        energy_percent = (node['E'] / node['Eo']) * 100
        # Get node status
        status = "Alive" if node['cond'] == 1 else "Dead"
        # Get node role
        role = "CH" if node['role'] == 1 else "Node"
        # Calculate distance to sink
        dist_to_sink = np.sqrt((node['x'] - SINK_X)**2 + (node['y'] - SINK_Y)**2)
        # Format position
        position = f"({node['x']:.1f}, {node['y']:.1f})"
        
        print("{:^6} | {:^10} | {:^12.3f} | {:^12.2f} | {:^10} | {:^8} | {:^12} | {:^15.2f}".format(
            node['id'],
            status,
            node['E'],
            energy_percent,
            role,
            node.get('traffic', 0),
            position,
            dist_to_sink
        ))
    
    print("=" * 100)
    
    # Print summary statistics
    alive_nodes = [n for n in network if n['cond'] == 1]
    print("\nSummary Statistics:")
    print(f"Total Alive Nodes: {len(alive_nodes)}/{len(network)}")
    print(f"Average Remaining Energy: {sum(n['E'] for n in network)/len(network):.3f}J")
    print(f"Total Network Traffic: {sum(n.get('traffic', 0) for n in network)} packets")
    cluster_heads = [n for n in network if n['role'] == 1 and n['cond'] == 1]
    print(f"Active Cluster Heads: {len(cluster_heads)}")

if __name__ == "__main__":
    main()