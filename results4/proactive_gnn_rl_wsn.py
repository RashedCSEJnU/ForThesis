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
import json
import csv
import time
import psutil
from functools import lru_cache
import gc
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

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

# Adding debug statements to trace execution
print("Debug: Device initialized")

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

# Sleep Scheduling Parameters
ENABLE_SLEEP_SCHEDULING = True  # Enable/disable sleep scheduling
DUTY_CYCLE = 0.3                # Percentage of time a node is active (30%)
SLEEP_ROUND_DURATION = 3        # Duration of each sleep cycle in rounds
WAKE_ROUND_DURATION = 2         # Duration of wake period in rounds
LISTEN_ROUND_DURATION = 1       # Duration of listen period before sleep
MIN_ACTIVE_NEIGHBORS = 2        # Minimum active neighbors required for coverage
COORDINATOR_DUTY_CYCLE = 0.8    # Cluster heads have higher duty cycle
ADAPTIVE_DUTY_CYCLE = True      # Enable adaptive duty cycling based on traffic
SLEEP_COORDINATION_ENERGY = 10e-9  # Energy for sleep coordination messages (per message)
PER_ROUND_VISUALIZATION = False # Toggle for per-round visualization
EXPORT_METRICS_CSV = True       # Toggle for exporting per-round metrics to CSV

# Duty cycle limits for adaptive scheduling
MIN_DUTY_CYCLE = 0.05
MAX_DUTY_CYCLE = 0.95

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
NODE_FEATURE_SIZE = 9  # [energy_ratio, x, y, dist_to_sink, hop_count, network_energy, congestion, sleep_state, duty_cycle]
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
        network_energy_percentage = total_energy / max_possible_energy if max_possible_energy > 0 else 0
        
        # Calculate congestion (based on number of active nodes in proximity)
        proximity_nodes = sum(1 for node in network 
                             if isinstance(node, dict) and 
                             node.get('cond', 1) == 1 and
                             np.sqrt((node.get('x', 0) - current_node['x'])**2 + 
                                    (node.get('y', 0) - current_node['y'])**2) <= TRANSMISSION_RANGE)
        normalized_congestion = proximity_nodes / NUM_NODES if NUM_NODES > 0 else 0

        # Sleep scheduling features
        sleep_state_encoded = 0.0 # Default to 'sleep'
        if current_node.get('sleep_state') == 'awake':
            sleep_state_encoded = 1.0
        elif current_node.get('sleep_state') == 'listen':
            sleep_state_encoded = 0.5
        
        normalized_duty_cycle = current_node.get('duty_cycle', DUTY_CYCLE) # Already normalized or should be
        
        # Return state as tensor
        state = torch.tensor([
            normalized_energy,
            normalized_x,
            normalized_y,
            normalized_dist_to_sink,
            normalized_hop_count,
            network_energy_percentage,
            normalized_congestion,
            sleep_state_encoded,
            normalized_duty_cycle
        ], dtype=torch.float32).to(device)
        
        return state.unsqueeze(0)  # Add batch dimension
    
    def get_action(self, state, gnn_embedding, available_nodes, network_graph=None, source_node=None):
        """Select action (next hop node) using epsilon-greedy policy with proactive considerations"""
        if not available_nodes:
            return None
            
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select random next hop with some preference for better nodes
            node_weights = []
            for node in available_nodes:
                # Give higher weights to nodes with more energy and better sleep states
                energy_weight = node['E'] / node['Eo'] if node['Eo'] > 0 else 0.1
                sleep_weight = 1.0 if node.get('sleep_state') == 'awake' else 0.5
                weight = energy_weight * sleep_weight
                node_weights.append(weight)
            
            # Weighted random selection during exploration
            if sum(node_weights) > 0:
                probabilities = [w / sum(node_weights) for w in node_weights]
                return np.random.choice(available_nodes, p=probabilities)
            else:
                return random.choice(available_nodes)
        else:
            # Exploit: select node with highest Q-value plus proactive bonuses
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
                        
                        # Enhanced proactive considerations
                        proactive_bonus = 0
                        
                        # 1. Network connectivity preservation
                        if network_graph and source_node:
                            try:
                                # Check if this node is a cut vertex
                                G = network_graph.copy()
                                G.remove_node(node['id'])
                                if source_node['id'] in G.nodes and NUM_NODES in G.nodes:
                                    paths = nx.has_path(G, source_node['id'], NUM_NODES)
                                    if not paths:
                                        proactive_bonus -= 0.5  # Penalize choosing critical cut vertices
                                
                                # Bonus for nodes that improve overall connectivity
                                original_components = len(list(nx.connected_components(network_graph)))
                                if original_components > 1:
                                    # This path could help bridge components
                                    proactive_bonus += 0.3
                            except:
                                pass
                        
                        # 2. Future energy viability
                        if 'predicted_energy' in node:
                            energy_ratio = node['predicted_energy'] / node['E'] if node['E'] > 0 else 0
                            if energy_ratio > 0.8:  # Predicted to maintain good energy
                                proactive_bonus += 0.3
                            elif energy_ratio < 0.5:  # Predicted to lose significant energy
                                proactive_bonus -= 0.2
                        
                        # 3. Load balancing
                        traffic_load = node.get('traffic', 0)
                        if traffic_load < 5:  # Low traffic node
                            proactive_bonus += 0.2
                        elif traffic_load > 15:  # High traffic node
                            proactive_bonus -= 0.3
                        
                        # 4. Sleep state consideration
                        if node.get('sleep_state') == 'awake':
                            proactive_bonus += 0.1
                        elif node.get('sleep_state') == 'sleep':
                            proactive_bonus -= 0.4
                        
                        # 5. Distance to sink consideration
                        sink_distance = np.sqrt((node['x'] - SINK_X)**2 + (node['y'] - SINK_Y)**2)
                        max_distance = np.sqrt(FIELD_X**2 + FIELD_Y**2)
                        distance_factor = 1 - (sink_distance / max_distance)
                        proactive_bonus += distance_factor * 0.2
                        
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
    node_map = {node_id: idx for idx, node_id in enumerate(G.nodes())}
    
    for edge in G.edges(data=True):
        source, target, data = edge
        # Use mapped indices
        edge_index.append([node_map[source], node_map[target]])
        edge_index.append([node_map[target], node_map[source]])  # Add reverse edge for undirected graph
        
        # Edge features: [distance, energy_ratio, signal_strength]
        distance = data.get('distance', 0)
        energy_ratio = data.get('energy_ratio', 1.0)
        # Simple signal strength model based on distance
        signal_strength = max(0, 1 - (distance / TRANSMISSION_RANGE)) if TRANSMISSION_RANGE > 0 else 0
        
        edge_features = [distance / TRANSMISSION_RANGE if TRANSMISSION_RANGE > 0 else 0, energy_ratio, signal_strength]
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
    x_features = [] # Renamed from x to avoid confusion
    
    # Process network nodes
    for i, node_id_orig in enumerate(G.nodes()): # Use node_id_orig to avoid conflict with node variable
        
        if node_id_orig == NUM_NODES:  # Sink node
            x_features.append([
                1.0,  # Energy ratio
                SINK_POS[0] / FIELD_X if FIELD_X > 0 else 0,  # Normalized x
                SINK_POS[1] / FIELD_Y if FIELD_Y > 0 else 0,  # Normalized y
                0.0,  # Distance to sink = 0
                0.0,  # Hop count = 0
                1.0,  # Network energy (irrelevant for sink)
                0.0,  # Congestion (irrelevant for sink)
                1.0,  # Sleep state (always awake)
                1.0   # Duty cycle (always active)
            ])
        else:
            # Find the node in network
            node = next((n for n in network if n['id'] == node_id_orig), None)
            if node:
                normalized_energy = node['E'] / INITIAL_ENERGY_MAX if INITIAL_ENERGY_MAX > 0 else 0
                normalized_x = node['x'] / FIELD_X if FIELD_X > 0 else 0
                normalized_y = node['y'] / FIELD_Y if FIELD_Y > 0 else 0
                max_dist = np.sqrt(FIELD_X**2 + FIELD_Y**2)
                normalized_dist_to_sink = node['dts'] / max_dist if max_dist > 0 else 0
                max_hops = (FIELD_X / TRANSMISSION_RANGE) if TRANSMISSION_RANGE > 0 else 1
                normalized_hop_count = node['hop'] / max_hops if max_hops > 0 else 0
                
                # Network energy percentage
                total_network_energy = sum(n['E'] for n in network if n['cond'] == 1)
                max_possible_total_energy = NUM_NODES * INITIAL_ENERGY_MAX
                network_energy_percentage = total_network_energy / max_possible_total_energy if max_possible_total_energy > 0 else 0
                
                # Congestion
                proximity_nodes = sum(1 for n_neighbor in network if n_neighbor['cond'] == 1 and
                                     np.sqrt((n_neighbor['x'] - node['x'])**2 + 
                                            (n_neighbor['y'] - node['y'])**2) <= TRANSMISSION_RANGE)
                normalized_congestion = proximity_nodes / NUM_NODES if NUM_NODES > 0 else 0

                # Sleep features
                sleep_state_encoded = 0.0 # Default to 'sleep'
                if node.get('sleep_state') == 'awake':
                    sleep_state_encoded = 1.0
                elif node.get('sleep_state') == 'listen':
                    sleep_state_encoded = 0.5
                
                normalized_duty_cycle = node.get('duty_cycle', DUTY_CYCLE)

                x_features.append([
                    normalized_energy,
                    normalized_x,
                    normalized_y,
                    normalized_dist_to_sink,
                    normalized_hop_count,
                    network_energy_percentage,
                    normalized_congestion,
                    sleep_state_encoded,
                    normalized_duty_cycle
                ])
            else: # Should not happen if G is built from network
                x_features.append([0.0] * NODE_FEATURE_SIZE)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_features, dtype=torch.float) # Renamed from x
    
    # Create PyG data object
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_map

############################## WSN Functions ##############################

def initialize_network(num_nodes, field_x, field_y, sink_x, sink_y, range_c):
    """Initialize WSN with randomly placed nodes"""
    print("Debug: Entered initialize_network")
    network = []

    # Create sensor nodes
    for i in range(num_nodes):
        print(f"Debug: Initializing node {i}")
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

        # Sleep scheduling attributes
        node['sleep_state'] = 'awake'  # 'awake', 'listen', 'sleep'
        node['duty_cycle'] = DUTY_CYCLE if node['role'] == 0 else COORDINATOR_DUTY_CYCLE
        node['sleep_timer'] = WAKE_ROUND_DURATION # Start awake
        node['consecutive_sleep_rounds'] = 0
        node['failed_transmissions'] = 0
        node['neighbors_in_range'] = 0 # For adaptive duty cycle

        network.append(node)
        print(f"Debug: Node {i} initialized")

    print("Debug: All nodes initialized")
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
                    avg_future_energy = np.mean(future_energies)
                    current_energy = node['E']
                    if current_energy > 0:
                        future_energy_factor = min(2.0, avg_future_energy / current_energy)
                    else:
                        future_energy_factor = 0
            
            # Traffic factor - prefer nodes with lower historical traffic
            traffic_factor = 1.0
            if hasattr(node, 'traffic') and node['traffic'] > 0:
                traffic_factor = 1 - min(1, node['traffic'] / 100)  # Normalize traffic
            
            # Sleep scheduling factor - prefer nodes that are awake
            sleep_factor = 1.0
            if ENABLE_SLEEP_SCHEDULING:
                if node.get('sleep_state') == 'awake':
                    sleep_factor = 1.0
                elif node.get('sleep_state') == 'listen':
                    sleep_factor = 0.8
                else:  # sleep
                    sleep_factor = 0.3
            
            # Combined score
            node['score'] = (0.35 * energy_factor + 
                            0.15 * position_factor + 
                            0.25 * future_energy_factor +
                            0.1 * traffic_factor +
                            0.15 * sleep_factor)
    
    # Sort by score and select top nodes as CHs
    alive_nodes = [node for node in network if node['cond'] == 1]
    sorted_nodes = sorted(alive_nodes, key=lambda x: x.get('score', 0), reverse=True)
    
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
            # Update duty cycle for cluster heads
            if ENABLE_SLEEP_SCHEDULING:
                node['duty_cycle'] = COORDINATOR_DUTY_CYCLE
                node['sleep_state'] = 'awake'  # CHs should be awake
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
                # Try to add intermediate CHs to connect disconnected ones
                remaining_nodes = [n for n in sorted_nodes if n['role'] == 0 and n['cond'] == 1]
                for remaining_node in remaining_nodes[:min(5, len(remaining_nodes))]:
                    remaining_node['role'] = 1
                    if ENABLE_SLEEP_SCHEDULING:
                        remaining_node['duty_cycle'] = COORDINATOR_DUTY_CYCLE
                        remaining_node['sleep_state'] = 'awake'
                    cluster_heads.append(remaining_node)
                    if len(cluster_heads) >= num_ch * 1.5:  # Don't exceed 150% of target
                        break
    
    return cluster_heads

def form_clusters(network, cluster_heads):
    """Assign nodes to their nearest cluster head with proactive considerations"""
    # Reset cluster assignments
    for node in network:
        if node['cond'] == 1 and node['role'] == 0:
            node['cluster'] = None
            node['closest'] = 0
            
            # Find nearest cluster head
            min_distance = float('inf')
            nearest_ch = None
            
            for ch in cluster_heads:
                if ch['cond'] == 1:
                    distance = np.sqrt((node['x'] - ch['x'])**2 + (node['y'] - ch['y'])**2)
                    
                    # Consider energy and predicted energy in clustering decision
                    energy_weight = ch['E'] / ch['Eo'] if ch['Eo'] > 0 else 0
                    
                    # Sleep scheduling consideration
                    sleep_weight = 1.0
                    if ENABLE_SLEEP_SCHEDULING:
                        if ch.get('sleep_state') == 'awake':
                            sleep_weight = 1.0
                        elif ch.get('sleep_state') == 'listen':
                            sleep_weight = 0.8
                        else:  # sleep
                            sleep_weight = 0.5
                    
                    # Weighted distance considering energy and sleep state
                    weighted_distance = distance * (2 - energy_weight) * (2 - sleep_weight)
                    
                    if weighted_distance < min_distance and distance <= TRANSMISSION_RANGE:
                        min_distance = weighted_distance
                        nearest_ch = ch
            
            if nearest_ch:
                node['cluster'] = nearest_ch['id']
                node['closest'] = nearest_ch['id']
            else:
                # If no CH is within range, find the closest one anyway
                min_actual_distance = float('inf')
                for ch in cluster_heads:
                    if ch['cond'] == 1:
                        distance = np.sqrt((node['x'] - ch['x'])**2 + (node['y'] - ch['y'])**2)
                        if distance < min_actual_distance:
                            min_actual_distance = distance
                            nearest_ch = ch
                
                if nearest_ch:
                    node['cluster'] = nearest_ch['id']
                    node['closest'] = nearest_ch['id']


def predict_traffic_patterns(network, traffic_history, window_size=TRAFFIC_PREDICTION_WINDOW):
    """Predict future traffic patterns based on historical data"""
    predictions = {}
    
    for node in network:
        node_id = node['id']
        
        # Initialize if not in history
        if node_id not in traffic_history:
            traffic_history[node_id] = []
        
        # Add current traffic to history
        current_traffic = node.get('traffic', 0)
        traffic_history[node_id].append(current_traffic)
        
        # Keep only recent history
        if len(traffic_history[node_id]) > window_size:
            traffic_history[node_id] = traffic_history[node_id][-window_size:]
        
        # Predict future traffic (simple moving average with trend)
        if len(traffic_history[node_id]) >= 2:
            recent_traffic = traffic_history[node_id][-min(5, len(traffic_history[node_id])):]
            avg_traffic = np.mean(recent_traffic)
            
            # Calculate trend
            if len(recent_traffic) >= 3:
                trend = (recent_traffic[-1] - recent_traffic[0]) / len(recent_traffic)
            else:
                trend = 0
            
            # Predict next round traffic
            predicted_traffic = max(0, avg_traffic + trend)
            predictions[node_id] = predicted_traffic
        else:
            predictions[node_id] = current_traffic
    
    return predictions

def calculate_energy_consumption(from_node, to_node, packet_size):
    """Calculate energy consumption for transmitting packets between nodes"""
    distance = np.sqrt((from_node['x'] - to_node['x'])**2 + 
                      (from_node['y'] - to_node['y'])**2)
    
    # Energy for transmission (ETX) and reception (ERX)
    etx = E_ELEC * packet_size + E_AMP * packet_size * (distance ** 2)
    erx = E_ELEC * packet_size
    
    return etx, erx

def calculate_transmission_energy(distance, packet_size):
    """Calculate transmission energy based on distance"""
    if distance <= 0:
        return 0
    return E_ELEC * packet_size + E_AMP * packet_size * (distance ** 2)

############################## Sleep Scheduling Functions ##############################

def update_sleep_states(network, round_num):
    """Update sleep states of all nodes based on duty cycle and scheduling"""
    if not ENABLE_SLEEP_SCHEDULING:
        return
    
    for node in network:
        if node['cond'] == 0:  # Dead nodes stay dead
            continue
            
        # Cluster heads have different sleep patterns
        if node['role'] == 1:  # Cluster head
            # CHs should stay awake most of the time
            node['sleep_state'] = 'awake'
            node['duty_cycle'] = COORDINATOR_DUTY_CYCLE
            continue
        
        # Regular nodes follow duty cycle
        node['sleep_timer'] -= 1
        
        # Adaptive duty cycle based on traffic and energy
        if ADAPTIVE_DUTY_CYCLE:
            base_duty = DUTY_CYCLE
            
            # Adjust based on energy level
            energy_ratio = node['E'] / node['Eo'] if node['Eo'] > 0 else 0
            if energy_ratio < 0.3:
                # Low energy, reduce duty cycle to conserve power
                energy_adjustment = -0.2
            elif energy_ratio > 0.7:
                # High energy, can afford higher duty cycle for better performance
                energy_adjustment = 0.1
            else:
                # Medium energy, maintain current duty cycle
                energy_adjustment = 0
            
            # Adjust based on traffic
            traffic_level = node.get('traffic', 0)
            if traffic_level > 10:
                traffic_adjustment = 0.15  # High traffic, stay awake more
            elif traffic_level > 5:
                traffic_adjustment = 0.05
            else:
                traffic_adjustment = 0
            
            # Adjust based on number of neighbors
            neighbors_count = node.get('neighbors_in_range', 0)
            if neighbors_count < MIN_ACTIVE_NEIGHBORS:
                neighbor_adjustment = 0.2  # Need to stay awake for coverage
            else:
                neighbor_adjustment = -0.1  # Can afford to sleep more
            
            # Apply adjustments
            new_duty_cycle = base_duty + energy_adjustment + traffic_adjustment + neighbor_adjustment
            node['duty_cycle'] = max(MIN_DUTY_CYCLE, min(MAX_DUTY_CYCLE, new_duty_cycle))
        
        # State machine for sleep scheduling
        if node['sleep_state'] == 'awake':
            if node['sleep_timer'] <= 0:
                # Check if node should go to listen state
                if random.random() > node['duty_cycle']:
                    node['sleep_state'] = 'listen'
                    node['sleep_timer'] = LISTEN_ROUND_DURATION
                else:
                    node['sleep_timer'] = WAKE_ROUND_DURATION
        
        elif node['sleep_state'] == 'listen':
            if node['sleep_timer'] <= 0:
                # Decide whether to sleep or wake up
                if random.random() > node['duty_cycle']:
                    node['sleep_state'] = 'sleep'
                    node['sleep_timer'] = SLEEP_ROUND_DURATION
                    node['consecutive_sleep_rounds'] = 0
                else:
                    node['sleep_state'] = 'awake'
                    node['sleep_timer'] = WAKE_ROUND_DURATION
        
        elif node['sleep_state'] == 'sleep':
            node['consecutive_sleep_rounds'] += 1
            if node['sleep_timer'] <= 0:
                # Wake up after sleep period
                node['sleep_state'] = 'awake'
                node['sleep_timer'] = WAKE_ROUND_DURATION
                node['consecutive_sleep_rounds'] = 0
            
            # Emergency wake up for critical situations
            if node['consecutive_sleep_rounds'] > SLEEP_ROUND_DURATION * 2:
                node['sleep_state'] = 'awake'
                node['sleep_timer'] = WAKE_ROUND_DURATION
                node['consecutive_sleep_rounds'] = 0

def can_node_transmit(node):
    """Check if a node can transmit based on its sleep state"""
    if not ENABLE_SLEEP_SCHEDULING:
        return node['cond'] == 1
    
    return (node['cond'] == 1 and 
            node['sleep_state'] in ['awake', 'listen'])

def calculate_sleep_energy_savings(network):
    """Calculate energy savings from sleep scheduling"""
    if not ENABLE_SLEEP_SCHEDULING:
        return 0
    
    total_savings = 0
    for node in network:
        if node['cond'] == 1 and node['sleep_state'] == 'sleep':
            # Assume sleeping nodes consume 10% of active energy
            active_energy = E_ELEC * PACKET_SIZE * 0.1  # Base listening energy
            sleep_energy = active_energy * 0.1
            total_savings += (active_energy - sleep_energy)
    
    return total_savings

def update_neighbors_count(network):
    """Update the count of active neighbors for each node"""
    for node in network:
        if node['cond'] == 0:
            continue
            
        neighbors_count = 0
        for other_node in network:
            if (other_node['cond'] == 1 and 
                other_node['id'] != node['id'] and
                can_node_transmit(other_node)):
                
                distance = np.sqrt((node['x'] - other_node['x'])**2 + 
                                 (node['y'] - other_node['y'])**2)
                if distance <= TRANSMISSION_RANGE:
                    neighbors_count += 1
        
        node['neighbors_in_range'] = neighbors_count

############################## Training and Inference ##############################

def train_gnn_model(gnn_model, gnn_optimizer, network_history, energy_targets):
    """Train the GNN model on historical network data with improved loss calculation"""
    if len(network_history) < 2:
        return None
    
    gnn_model.train()
    total_loss = 0
    num_batches = 0
    total_energy_loss = 0
    total_topology_loss = 0
    
    # Create training batches from historical data
    for i in range(len(network_history) - 1):
        current_network = network_history[i]
        next_network = network_history[i + 1]
        
        # Filter out dead nodes for cleaner training
        current_alive = [n for n in current_network if n['cond'] == 1]
        next_alive = [n for n in next_network if n['cond'] == 1]
        
        if len(current_alive) < 2:  # Need at least 2 nodes for meaningful training
            continue
        
        try:
            # Create graph for current state
            G = create_network_graph(current_alive, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
            if len(G.nodes) == 0:
                continue
                
            pyg_data, node_map = graph_to_pyg_data(G, current_alive)
            
            if pyg_data.x.size(0) == 0:  # Empty graph
                continue
            
            # Get node embeddings and predictions
            node_embeddings, graph_embedding = gnn_model(pyg_data.x.to(device), 
                                                         pyg_data.edge_index.to(device),
                                                         pyg_data.edge_attr.to(device))
            
            # Create target energies for next time step
            target_energies = []
            valid_node_indices = []
            
            for idx, (node_id, mapped_idx) in enumerate(node_map.items()):
                if node_id == NUM_NODES:  # Skip sink
                    continue
                    
                # Find corresponding node in next network state
                next_node = next((n for n in next_alive if n['id'] == node_id), None)
                if next_node and mapped_idx < len(current_alive):
                    # Normalize energy to [0,1] range
                    normalized_energy = next_node['E'] / INITIAL_ENERGY_MAX
                    target_energies.append(normalized_energy)
                    valid_node_indices.append(mapped_idx)
            
            if len(target_energies) == 0 or len(valid_node_indices) == 0:
                continue
                
            target_energies = torch.tensor(target_energies, dtype=torch.float32).to(device)
            
            # Get energy predictions for valid nodes
            valid_embeddings = node_embeddings[valid_node_indices]
            valid_features = pyg_data.x[valid_node_indices]
            
            energy_predictions = gnn_model.predict_energy(valid_embeddings, valid_features)
            
            # Energy prediction loss
            if energy_predictions.shape[0] == target_energies.shape[0]:
                energy_loss = F.mse_loss(energy_predictions[:, 0], target_energies)
                
                # Add L2 regularization for better generalization
                l2_reg = 0.0
                for param in gnn_model.parameters():
                    l2_reg += torch.norm(param)
                
                # Topology preservation loss (encourage similar embeddings for similar nodes)
                topology_loss = 0.0
                if node_embeddings.size(0) > 1:
                    # Compute pairwise distances in embedding space
                    embedding_dists = torch.cdist(node_embeddings, node_embeddings)
                    
                    # Compute pairwise distances in feature space (energy, position)
                    feature_dists = torch.cdist(pyg_data.x[:, :3], pyg_data.x[:, :3])  # Use first 3 features
                    
                    # Encourage embeddings to preserve feature space relationships
                    topology_loss = F.mse_loss(embedding_dists, feature_dists) * 0.1
                
                # Combined loss with regularization
                total_batch_loss = energy_loss + topology_loss + l2_reg * 1e-4
                
                # Backpropagation
                gnn_optimizer.zero_grad()
                total_batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
                gnn_optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_energy_loss += energy_loss.item()
                total_topology_loss += topology_loss.item() if isinstance(topology_loss, torch.Tensor) else topology_loss
                num_batches += 1
                
        except Exception as e:
            print(f"Error in GNN training batch {i}: {e}")
            continue
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_energy_loss = total_energy_loss / num_batches
        avg_topology_loss = total_topology_loss / num_batches
        
        # Print detailed loss information
        if num_batches % 5 == 0:  # Print every 5th training iteration
            print(f"GNN Training - Total Loss: {avg_loss:.6f}, Energy Loss: {avg_energy_loss:.6f}, Topology Loss: {avg_topology_loss:.6f}")
        
        return avg_loss
    
    return None

def predict_future_energies(gnn_model, network, prediction_horizon=PREDICTION_HORIZON):
    """Predict future energy levels for all nodes with multi-step prediction"""
    gnn_model.eval()
    predictions = {}
    
    # Create current network graph
    alive_nodes = [n for n in network if n['cond'] == 1]
    if len(alive_nodes) == 0:
        return predictions
        
    G = create_network_graph(alive_nodes, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
    if len(G.nodes) == 0:
        return predictions
        
    try:
        pyg_data, node_map = graph_to_pyg_data(G, alive_nodes)
        
        with torch.no_grad():
            # Get embeddings
            node_embeddings, graph_embedding = gnn_model(pyg_data.x.to(device), 
                                         pyg_data.edge_index.to(device),
                                         pyg_data.edge_attr.to(device))
            
            # Multi-step energy prediction
            for node_id, mapped_id in node_map.items():
                if node_id == NUM_NODES:  # Skip sink
                    continue
                    
                node = next((n for n in alive_nodes if n['id'] == node_id), None)
                if not node:
                    continue
                    
                predicted_energies = []
                current_energy = node['E']
                
                # Predict energy for each step in the horizon
                for step in range(prediction_horizon):
                    if mapped_id < node_embeddings.size(0):
                        # Create modified node features for this prediction step
                        node_features = pyg_data.x[mapped_id].clone().unsqueeze(0)
                        
                        # Update energy feature with current predicted energy
                        normalized_energy = current_energy / INITIAL_ENERGY_MAX
                        node_features[0, 0] = normalized_energy  # First feature is energy
                        
                        node_embedding = node_embeddings[mapped_id].unsqueeze(0)
                        energy_pred = gnn_model.predict_energy(node_embedding, node_features)
                        
                        # Convert back to actual energy value
                        pred_energy = energy_pred[0, 0].item() * INITIAL_ENERGY_MAX
                        
                        # Apply energy decay based on typical consumption patterns
                        decay_factor = 0.95 ** step  # Gradual decay over time
                        traffic_factor = 1 - (node.get('traffic', 0) / 100) * 0.1  # Traffic impact
                        sleep_factor = 1.0 if node.get('sleep_state') == 'sleep' else 0.9  # Sleep savings
                        
                        pred_energy = max(0, pred_energy * decay_factor * traffic_factor * sleep_factor)
                        predicted_energies.append(pred_energy)
                        current_energy = pred_energy  # Use predicted energy for next step
                    else:
                        # Fallback prediction based on current consumption rate
                        consumption_rate = (node['Eo'] - node['E']) / max(1, node.get('round', 1))
                        pred_energy = max(0, current_energy - consumption_rate * (step + 1))
                        predicted_energies.append(pred_energy)
                        current_energy = pred_energy
                
                predictions[node_id] = predicted_energies
                
                # Update node's predicted energy (weighted average of future predictions)
                if predicted_energies:
                    weights = [0.5 ** i for i in range(len(predicted_energies))]  # Exponential decay weights
                    weighted_avg = sum(p * w for p, w in zip(predicted_energies, weights)) / sum(weights)
                    node['predicted_energy'] = weighted_avg
                    
                    # Add confidence measure based on variance
                    if len(predicted_energies) > 1:
                        variance = np.var(predicted_energies)
                        node['prediction_confidence'] = max(0, 1 - variance / (INITIAL_ENERGY_MAX ** 2))
                    else:
                        node['prediction_confidence'] = 0.5
                        
    except Exception as e:
        print(f"Error in energy prediction: {e}")
        # Fallback to simple energy prediction
        for node in alive_nodes:
            if node['id'] != NUM_NODES:
                current_energy = node['E']
                fallback_predictions = []
                for step in range(prediction_horizon):
                    # Simple linear decay prediction
                    consumption_rate = (node['Eo'] - node['E']) / max(1, node.get('round', 1))
                    pred_energy = max(0, current_energy - consumption_rate * (step + 1))
                    fallback_predictions.append(pred_energy)
                predictions[node['id']] = fallback_predictions
                node['predicted_energy'] = sum(fallback_predictions) / len(fallback_predictions)
                node['prediction_confidence'] = 0.3  # Lower confidence for fallback
    
    return predictions

def calculate_reward(path, energy_consumed, network, sink_pos):
    """Calculate sophisticated reward for DRL agent based on multiple factors"""
    if not path or len(path) == 0:
        return -100  # Large penalty for failed routing
    
    # Base metrics
    path_length = len(path)
    alive_nodes = [node for node in network if node['cond'] == 1]
    total_nodes = len(network)
    
    if len(alive_nodes) == 0:
        return -100
    
    # 1. Energy Efficiency Reward (40% weight)
    energy_efficiency = 1 / (energy_consumed + 1e-6)
    normalized_energy_efficiency = min(10, energy_efficiency)  # Cap the reward
    energy_reward = normalized_energy_efficiency * 4.0
    
    # 2. Path Length Penalty (15% weight) 
    optimal_path_length = max(1, np.ceil(path[-1]['dts'] / TRANSMISSION_RANGE))
    path_efficiency = optimal_path_length / max(1, path_length)
    path_reward = path_efficiency * 1.5
    
    # 3. Network Longevity Reward (25% weight)
    total_energy = sum(node['E'] for node in alive_nodes)
    max_possible_energy = len(alive_nodes) * INITIAL_ENERGY_MAX
    network_energy_ratio = total_energy / max_possible_energy if max_possible_energy > 0 else 0
    
    # Bonus for preserving high-energy nodes
    min_energy_in_path = min(node['E'] for node in path)
    avg_energy_in_path = sum(node['E'] for node in path) / len(path)
    energy_preservation = (min_energy_in_path / INITIAL_ENERGY_MAX) * 0.5 + \
                         (avg_energy_in_path / INITIAL_ENERGY_MAX) * 0.5
    
    longevity_reward = (network_energy_ratio + energy_preservation) * 1.25
    
    # 4. Load Balancing Reward (10% weight)
    if len(path) > 1:
        # Reward paths that use nodes with lower traffic
        path_traffic = [node.get('traffic', 0) for node in path]
        avg_path_traffic = sum(path_traffic) / len(path_traffic)
        max_network_traffic = max(node.get('traffic', 0) for node in alive_nodes)
        
        if max_network_traffic > 0:
            traffic_balance = 1 - (avg_path_traffic / max_network_traffic)
        else:
            traffic_balance = 1.0
        
        load_balance_reward = traffic_balance * 1.0
    else:
        load_balance_reward = 0.5
    
    # 5. Future Viability Reward (15% weight)
    future_viability = 0
    prediction_bonus = 0
    
    for node in path:
        # Current energy ratio
        current_ratio = node['E'] / node['Eo'] if node['Eo'] > 0 else 0
        
        # Predicted future energy ratio
        predicted_energy = node.get('predicted_energy', node['E'])
        if node['E'] > 0:
            future_ratio = predicted_energy / node['E']
            prediction_confidence = node.get('prediction_confidence', 0.5)
            
            # Weighted future viability
            future_viability += (current_ratio * 0.6 + future_ratio * 0.4) * prediction_confidence
        else:
            future_viability += current_ratio
        
        # Bonus for high-confidence predictions of stable energy
        if node.get('prediction_confidence', 0) > 0.7 and predicted_energy > node['E'] * 0.8:
            prediction_bonus += 0.1
    
    future_viability = (future_viability / len(path)) if len(path) > 0 else 0
    future_reward = (future_viability + prediction_bonus) * 1.5
    
    # 6. Connectivity Preservation Bonus (10% weight)
    connectivity_bonus = 0
    
    # Check if path maintains good connectivity to sink
    final_node = path[-1]
    distance_to_sink = np.sqrt((final_node['x'] - sink_pos[0])**2 + 
                              (final_node['y'] - sink_pos[1])**2)
    
    if distance_to_sink <= TRANSMISSION_RANGE:
        connectivity_bonus += 1.0  # Direct connection to sink
    elif distance_to_sink <= TRANSMISSION_RANGE * 2:
        connectivity_bonus += 0.5  # Close to sink
    
    # Bonus for nodes that improve overall connectivity
    original_components = len(list(nx.connected_components(create_network_graph(network, sink_pos, TRANSMISSION_RANGE))))
    if original_components > 1:
        # This path could help bridge components
        connectivity_bonus += 0.3
    
    connectivity_reward = connectivity_bonus * 1.0
    
    # 7. Sleep Scheduling Efficiency Bonus (5% weight)
    sleep_bonus = 0
    awake_nodes_in_path = sum(1 for node in path if node.get('sleep_state') == 'awake')
    sleep_efficiency = awake_nodes_in_path / len(path) if len(path) > 0 else 0
    sleep_bonus = sleep_efficiency * 0.5
    
    # 8. Network Health Penalty/Bonus
    network_health = len(alive_nodes) / total_nodes
    if network_health < 0.3:
        health_penalty = -2.0  # Severe penalty for degraded network
    elif network_health < 0.5:
        health_penalty = -1.0  # Moderate penalty
    elif network_health > 0.8:
        health_penalty = 1.0   # Bonus for healthy network
    else:
        health_penalty = 0
    
    # Combine all rewards with weights
    total_reward = (
        energy_reward * 0.40 +           # Energy efficiency (40%)
        path_reward * 0.15 +             # Path efficiency (15%)
        longevity_reward * 0.25 +        # Network longevity (25%)
        load_balance_reward * 0.10 +     # Load balancing (10%)
        future_reward * 0.15 +           # Future viability (15%)
        connectivity_reward * 0.10 +     # Connectivity (10%)
        sleep_bonus * 0.05 +             # Sleep efficiency (5%)
        health_penalty                   # Network health adjustment
    )
    
    # Apply additional penalties for edge cases
    if energy_consumed > INITIAL_ENERGY_MAX:
        total_reward -= 5.0  # Penalty for excessive energy consumption
    
    if path_length > NUM_NODES / 2:
        total_reward -= 2.0  # Penalty for very long paths
    
    # Final clamping to reasonable range
    final_reward = max(-50, min(50, total_reward))
    
    return final_reward

def update_dead_nodes(network):
    """Update dead nodes status"""
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

def visualize_network(network, round_num, sink_pos, save_plot=True, recent_paths=None):
    """Enhanced visualization of the current network state with improved aesthetics and information"""
    # Create larger figure with subplots for multiple views
    fig = plt.figure(figsize=(20, 12))
    
    # Main network topology plot
    ax1 = plt.subplot(2, 3, (1, 4))  # Large main plot
    
    # Separate nodes by status and role
    alive_nodes = [node for node in network if node['cond'] == 1]
    dead_nodes = [node for node in network if node['cond'] == 0]
    ch_nodes = [node for node in alive_nodes if node['role'] == 1]
    regular_nodes = [node for node in alive_nodes if node['role'] == 0]
    
    # Plot transmission range circles for cluster heads (optional)
    for ch in ch_nodes:
        circle = plt.Circle((ch['x'], ch['y']), TRANSMISSION_RANGE, 
                          fill=False, color='lightblue', alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
    
    # Plot connectivity lines between nodes and their cluster heads
    for node in regular_nodes:
        if node.get('cluster'):
            ch = next((ch for ch in ch_nodes if ch['id'] == node['cluster']), None)
            if ch:
                distance = np.sqrt((node['x'] - ch['x'])**2 + (node['y'] - ch['y'])**2)
                if distance <= TRANSMISSION_RANGE:
                    ax1.plot([node['x'], ch['x']], [node['y'], ch['y']], 
                           'gray', alpha=0.3, linewidth=0.5)
    
    # Plot recent transmission paths if available
    if recent_paths:
        for i, path in enumerate(recent_paths[-5:]):  # Show last 5 paths
            if path and len(path) > 1:
                path_x = [node['x'] for node in path]
                path_y = [node['y'] for node in path]
                alpha = 0.8 - (i * 0.15)  # Fade older paths
                ax1.plot(path_x, path_y, color='orange', alpha=alpha, 
                        linewidth=2, marker='o', markersize=3)
    
    # Plot regular alive nodes with energy-based coloring
    if regular_nodes:
        regular_x = [node['x'] for node in regular_nodes]
        regular_y = [node['y'] for node in regular_nodes]
        regular_energies = [node['E'] / node['Eo'] for node in regular_nodes]
        
        scatter1 = ax1.scatter(regular_x, regular_y, c=regular_energies, 
                             cmap='RdYlGn', s=80, alpha=0.8, vmin=0, vmax=1,
                             edgecolors='black', linewidth=0.5)
    
    # Plot cluster heads with distinct styling
    if ch_nodes:
        ch_x = [node['x'] for node in ch_nodes]
        ch_y = [node['y'] for node in ch_nodes]
        ch_energies = [node['E'] / node['Eo'] for node in ch_nodes]
        
        ax1.scatter(ch_x, ch_y, c=ch_energies, cmap='RdYlGn', 
                   marker='s', s=150, alpha=0.9, vmin=0, vmax=1,
                   edgecolors='blue', linewidth=2, label='Cluster Heads')
        
        # Add CH ID labels
        for ch in ch_nodes:
            ax1.annotate(f'CH{ch["id"]}', (ch['x'], ch['y']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
    
    # Plot dead nodes
    if dead_nodes:
        dead_x = [node['x'] for node in dead_nodes]
        dead_y = [node['y'] for node in dead_nodes]
        ax1.scatter(dead_x, dead_y, c='black', marker='x', s=100, 
                   alpha=0.7, label='Dead Nodes', linewidth=2)
    
    # Plot sink with enhanced styling
    ax1.scatter(sink_pos[0], sink_pos[1], c='red', marker='*', s=400, 
               label='Sink', edgecolors='darkred', linewidth=3, zorder=10)
    ax1.annotate('SINK', sink_pos, xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color='red')
    
    # Add colorbar for energy levels
    if alive_nodes:
        cbar = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar.set_label('Energy Ratio', fontsize=12)
    
    # Enhanced network statistics
    alive_count = len(alive_nodes)
    total_energy = sum(node['E'] for node in alive_nodes)
    avg_energy = total_energy / alive_count if alive_count > 0 else 0
    min_energy = min(node['E'] for node in alive_nodes) if alive_nodes else 0
    max_energy = max(node['E'] for node in alive_nodes) if alive_nodes else 0
    
    stats_text = f'Round: {round_num}\n'
    stats_text += f'Alive Nodes: {alive_count}/{NUM_NODES} ({alive_count/NUM_NODES*100:.1f}%)\n'
    stats_text += f'Cluster Heads: {len(ch_nodes)}\n'
    stats_text += f'Avg Energy: {avg_energy:.3f}J\n'
    stats_text += f'Min Energy: {min_energy:.3f}J\n'
    stats_text += f'Max Energy: {max_energy:.3f}J\n'
    stats_text += f'Total Energy: {total_energy:.3f}J'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
             verticalalignment='top', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(f'WSN Topology - Round {round_num}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, FIELD_X + 5)
    ax1.set_ylim(-5, FIELD_Y + 5)
    
    # Energy distribution histogram
    ax2 = plt.subplot(2, 3, 2)
    if alive_nodes:
        energies = [node['E'] for node in alive_nodes]
        ax2.hist(energies, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(avg_energy, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_energy:.2f}J')
        ax2.set_xlabel('Energy (J)')
        ax2.set_ylabel('Number of Nodes')
        ax2.set_title('Energy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Node status pie chart
    ax3 = plt.subplot(2, 3, 3)
    labels = ['Alive Regular', 'Cluster Heads', 'Dead']
    sizes = [len(regular_nodes), len(ch_nodes), len(dead_nodes)]
    colors = ['lightgreen', 'blue', 'red']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Node Status Distribution')
    
    # Traffic load visualization
    ax4 = plt.subplot(2, 3, 5)
    if alive_nodes:
        traffic_loads = [node.get('traffic', 0) for node in alive_nodes]
        node_ids = [node['id'] for node in alive_nodes]
        bars = ax4.bar(range(len(alive_nodes)), traffic_loads, 
                      color=['red' if node['role'] == 1 else 'blue' for node in alive_nodes])
        ax4.set_xlabel('Node Index')
        ax4.set_ylabel('Traffic Load')
        ax4.set_title('Traffic Distribution Among Nodes')
        ax4.grid(True, alpha=0.3)
    
    # Energy over time trend (if we have historical data)
    ax5 = plt.subplot(2, 3, 6)
    # This would need historical data - for now show current energy levels by node ID
    if alive_nodes:
        sorted_nodes = sorted(alive_nodes, key=lambda x: x['id'])
        node_ids = [node['id'] for node in sorted_nodes]
        energies = [node['E'] for node in sorted_nodes]
        colors = ['red' if node['role'] == 1 else 'green' for node in sorted_nodes]
        ax5.scatter(node_ids, energies, c=colors, alpha=0.7)
        ax5.set_xlabel('Node ID')
        ax5.set_ylabel('Current Energy (J)')
        ax5.set_title('Energy by Node ID')
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'results/network_round_{round_num:04d}.png', dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

############################## Advanced Network Analysis ##############################

def analyze_network_topology(network, round_num):
    """Perform advanced topology analysis for network optimization"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    if len(alive_nodes) < 2:
        return {}
    
    # Create network graph for analysis
    G = create_network_graph(alive_nodes, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
    
    analysis = {
        'round': round_num,
        'total_nodes': len(alive_nodes),
        'total_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
        'network_density': nx.density(G) if len(G.nodes()) > 1 else 0,
        'clustering_coefficient': nx.average_clustering(G) if len(G.nodes()) > 2 else 0,
        'connected_components': nx.number_connected_components(G),
        'largest_component_size': len(max(nx.connected_components(G), key=len)) if len(G.nodes()) > 0 else 0
    }
    
    # Analyze connectivity to sink
    if NUM_NODES in G.nodes:
        nodes_connected_to_sink = 0
        shortest_paths_to_sink = []
        
        for node_id in G.nodes():
            if node_id != NUM_NODES and nx.has_path(G, node_id, NUM_NODES):
                nodes_connected_to_sink += 1
                try:
                    path_length = nx.shortest_path_length(G, node_id, NUM_NODES)
                    shortest_paths_to_sink.append(path_length)
                except:
                    pass
        
        analysis['sink_connected_nodes'] = nodes_connected_to_sink
        analysis['sink_connectivity_ratio'] = nodes_connected_to_sink / len(alive_nodes) if alive_nodes else 0
        analysis['avg_path_to_sink'] = np.mean(shortest_paths_to_sink) if shortest_paths_to_sink else float('inf')
        analysis['max_path_to_sink'] = max(shortest_paths_to_sink) if shortest_paths_to_sink else float('inf')
    
    # Find critical nodes (cut vertices)
    cut_vertices = list(nx.articulation_points(G))
    analysis['critical_nodes'] = len(cut_vertices)
    analysis['critical_node_ratio'] = len(cut_vertices) / len(alive_nodes) if alive_nodes else 0
    
    # Energy distribution analysis
    energies = [node['E'] for node in alive_nodes]
    analysis['energy_std'] = np.std(energies)
    analysis['energy_cv'] = np.std(energies) / np.mean(energies) if np.mean(energies) > 0 else 0
    analysis['energy_gini'] = calculate_gini_coefficient(energies)
    
    # Traffic load analysis
    traffic_loads = [node.get('traffic', 0) for node in alive_nodes]
    analysis['traffic_std'] = np.std(traffic_loads)
    analysis['max_traffic_load'] = max(traffic_loads) if traffic_loads else 0
    analysis['traffic_imbalance'] = calculate_gini_coefficient(traffic_loads)
    
    # Sleep scheduling efficiency
    if ENABLE_SLEEP_SCHEDULING:
        awake_nodes = sum(1 for node in alive_nodes if node.get('sleep_state') == 'awake')
        sleep_nodes = sum(1 for node in alive_nodes if node.get('sleep_state') == 'sleep')
        listen_nodes = sum(1 for node in alive_nodes if node.get('sleep_state') == 'listen')
        
        analysis['sleep_efficiency'] = sleep_nodes / len(alive_nodes) if alive_nodes else 0
        analysis['awake_ratio'] = awake_nodes / len(alive_nodes) if alive_nodes else 0
        analysis['listen_ratio'] = listen_nodes / len(alive_nodes) if alive_nodes else 0
        
        # Calculate energy savings from sleep scheduling
        estimated_savings = calculate_sleep_energy_savings(network)
        analysis['sleep_energy_savings'] = estimated_savings
    
    return analysis

def calculate_gini_coefficient(values):
    """Calculate Gini coefficient for measuring inequality"""
    if len(values) == 0:
        return 0
    
    # Sort values
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    return gini if not np.isnan(gini) else 0

def detect_network_bottlenecks(network, traffic_threshold=10):
    """Detect potential network bottlenecks based on traffic and topology"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    G = create_network_graph(alive_nodes, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
    
    bottlenecks = []
    
    for node in alive_nodes:
        bottleneck_score = 0
        
        # High traffic load
        traffic = node.get('traffic', 0)
        if traffic > traffic_threshold:
            bottleneck_score += traffic / traffic_threshold
        
        # Critical node (articulation point)
        if node['id'] in nx.articulation_points(G):
            bottleneck_score += 2.0
        
        # Low energy with high degree
        degree = G.degree(node['id']) if node['id'] in G.nodes else 0
        energy_ratio = node['E'] / node['Eo'] if node['Eo'] > 0 else 0
        if degree > 3 and energy_ratio < 0.3:
            bottleneck_score += 1.5
        
        # Cluster head with high load
        if node.get('role') == 1 and traffic > 5:
            bottleneck_score += 1.0
        
        if bottleneck_score > 2.0:
            bottlenecks.append({
                'node_id': node['id'],
                'score': bottleneck_score,
                'traffic': traffic,
                'energy_ratio': energy_ratio,
                'degree': degree,
                'role': node.get('role', 0)
            })
    
    return sorted(bottlenecks, key=lambda x: x['score'], reverse=True)

def predict_network_lifetime(network, current_round, consumption_history):
    """Predict remaining network lifetime based on current trends"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    if len(alive_nodes) == 0:
        return 0
    
    # Calculate average energy consumption rate
    if len(consumption_history) >= 5:
        recent_consumption = consumption_history[-5:]
        avg_consumption_rate = np.mean(recent_consumption)
    else:
        # Estimate based on current energy levels
        total_consumed = sum(node['Eo'] - node['E'] for node in alive_nodes)
        avg_consumption_rate = total_consumed / max(1, current_round)
    
    # Predict when first node will die
    min_energy = min(node['E'] for node in alive_nodes)
    if avg_consumption_rate > 0:
        rounds_until_first_death = min_energy / avg_consumption_rate
    else:
        rounds_until_first_death = float('inf')
    
    # Predict when network becomes disconnected
    node_energies = [(node['E'], node['id']) for node in alive_nodes]
    node_energies.sort()  # Sort by energy level
    
    predicted_deaths = []
    for energy, node_id in node_energies:
        if avg_consumption_rate > 0:
            death_round = current_round + (energy / avg_consumption_rate)
            predicted_deaths.append((death_round, node_id))
    
    # Simulate network connectivity as nodes die
    connectivity_loss_round = float('inf')
    G = create_network_graph(alive_nodes, (SINK_X, SINK_Y), TRANSMISSION_RANGE)
    
    for death_round, node_id in predicted_deaths:
        if node_id in G.nodes:
            G.remove_node(node_id)
            if NUM_NODES in G.nodes:
                # Check if sink is still reachable by majority of nodes
                connected_to_sink = sum(1 for n in G.nodes() 
                                      if n != NUM_NODES and nx.has_path(G, n, NUM_NODES))
                total_remaining = len([n for n in G.nodes() if n != NUM_NODES])
                
                if total_remaining > 0 and connected_to_sink / total_remaining < 0.5:
                    connectivity_loss_round = death_round
                    break
    
    return {
        'first_node_death': rounds_until_first_death,
        'connectivity_loss': connectivity_loss_round,
        'avg_consumption_rate': avg_consumption_rate,
        'predicted_deaths': predicted_deaths[:10]  # Top 10 nodes likely to die first
    }

def optimize_cluster_head_placement(network, energy_predictions=None):
    """Optimize cluster head placement based on energy predictions and topology"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    if len(alive_nodes) < 2:
        return []
    
    # Calculate optimal number of CHs based on network size and connectivity
    optimal_ch_count = max(1, int(np.sqrt(len(alive_nodes)) * 0.8))
    
    # Score each node for CH suitability
    ch_candidates = []
    
    for node in alive_nodes:
        score = 0
        
        # Energy factor (40% weight)
        energy_ratio = node['E'] / node['Eo'] if node['Eo'] > 0 else 0
        score += energy_ratio * 0.4
        
        # Position factor (20% weight) - prefer central locations
        center_x, center_y = FIELD_X / 2, FIELD_Y / 2
        distance_to_center = np.sqrt((node['x'] - center_x)**2 + (node['y'] - center_y)**2)
        max_distance = np.sqrt((FIELD_X/2)**2 + (FIELD_Y/2)**2)
        centrality_score = 1 - (distance_to_center / max_distance)
        score += centrality_score * 0.2
        
        # Connectivity factor (25% weight) - prefer well-connected nodes
        neighbors = sum(1 for other in alive_nodes 
                       if other['id'] != node['id'] and
                       np.sqrt((node['x'] - other['x'])**2 + (node['y'] - other['y'])**2) <= TRANSMISSION_RANGE)
        connectivity_score = min(1.0, neighbors / 5)  # Normalize to max 5 neighbors
        score += connectivity_score * 0.25
        
        # Future energy factor (10% weight)
        if energy_predictions and node['id'] in energy_predictions:
            future_energies = energy_predictions[node['id']]
            if future_energies:
                avg_future_energy = np.mean(future_energies)
                future_ratio = min(2.0, avg_future_energy / node['E']) if node['E'] > 0 else 0
                score += (future_ratio / 2.0) * 0.1
        
        # Traffic factor (5% weight) - prefer nodes with lower current traffic
        traffic_penalty = min(1.0, node.get('traffic', 0) / 20)
        score += (1 - traffic_penalty) * 0.05
        
        ch_candidates.append((node, score))
    
    # Select top candidates
    ch_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_chs = [candidate[0] for candidate in ch_candidates[:optimal_ch_count]]
    
    # Ensure geographical distribution
    final_chs = []
    min_distance_between_chs = TRANSMISSION_RANGE * 1.5
    
    for ch in selected_chs:
        too_close = False
        for existing_ch in final_chs:
            distance = np.sqrt((ch['x'] - existing_ch['x'])**2 + (ch['y'] - existing_ch['y'])**2)
            if distance < min_distance_between_chs:
                too_close = True
                break
        
        if not too_close:
            final_chs.append(ch)
        
        if len(final_chs) >= optimal_ch_count:
            break
    
    # If we don't have enough CHs due to distance constraints, add more from candidates
    if len(final_chs) < max(1, optimal_ch_count // 2):
        remaining_candidates = [ch for ch in selected_chs if ch not in final_chs]
        final_chs.extend(remaining_candidates[:optimal_ch_count - len(final_chs)])
    
    return final_chs

def find_optimal_path_drl(source_node, sink_pos, network, agent, gnn_model, network_graph=None):
    """Find optimal path using DRL agent with GNN embeddings"""
    try:
        if source_node['cond'] == 0:  # Dead node
            return None, float('inf')
        
        path = [source_node]
        current_node = source_node
        visited = {source_node['id']}
        total_energy_consumed = 0
        max_hops = min(20, NUM_NODES)  # Prevent infinite loops
        
        # Create initial network graph if not provided
        if network_graph is None:
            network_graph = create_network_graph(network, sink_pos, TRANSMISSION_RANGE)
        
        for hop in range(max_hops):
            # Check if we can reach sink directly
            sink_distance = np.sqrt((current_node['x'] - sink_pos[0])**2 + 
                                  (current_node['y'] - sink_pos[1])**2)
            
            if sink_distance <= TRANSMISSION_RANGE:
                # Calculate energy for final transmission to sink
                final_energy = calculate_transmission_energy(sink_distance, PACKET_SIZE)
                total_energy_consumed += final_energy
                
                # Update current node energy
                current_node['E'] = max(0, current_node['E'] - final_energy)
                return path, total_energy_consumed
            
            # Find available next hop nodes
            available_nodes = []
            for node in network:
                if (node['cond'] == 1 and 
                    node['id'] not in visited and 
                    node['id'] != current_node['id']):
                    
                    distance = np.sqrt((current_node['x'] - node['x'])**2 + 
                                     (current_node['y'] - node['y'])**2)
                    if distance <= TRANSMISSION_RANGE and can_node_transmit(node):
                        available_nodes.append(node)
            
            if not available_nodes:
                break  # No available neighbors
            
            # Get current network state as graph for GNN
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
            if current_node_mapped_id < node_embeddings.size(0):
                current_gnn_embedding = node_embeddings[current_node_mapped_id].unsqueeze(0)
            else:
                current_gnn_embedding = torch.zeros(1, node_embeddings.size(1)).to(device)
            
            # Use DRL agent to select next hop
            next_node = agent.get_action(current_state, current_gnn_embedding, 
                                       available_nodes, network_graph, current_node)
            
            if next_node is None:
                break
            
            # Calculate energy consumption for this hop
            etx, erx = calculate_energy_consumption(current_node, next_node, PACKET_SIZE)
            hop_energy = etx + erx
            total_energy_consumed += hop_energy
            
            # Update energies
            current_node['E'] = max(0, current_node['E'] - etx)
            next_node['E'] = max(0, next_node['E'] - erx)
            
            # Update traffic counters
            current_node['traffic'] = current_node.get('traffic', 0) + 1
            next_node['traffic'] = next_node.get('traffic', 0) + 1
            
            path.append(next_node)
            visited.add(next_node['id'])
            current_node = next_node
        
        # Check if we got close enough to sink for final transmission
        final_distance = np.sqrt((current_node['x'] - sink_pos[0])**2 + 
                               (current_node['y'] - sink_pos[1])**2)
        
        if final_distance <= TRANSMISSION_RANGE * 1.5:  # Allow some tolerance
            final_energy = calculate_transmission_energy(final_distance, PACKET_SIZE)
            total_energy_consumed += final_energy
            current_node['E'] = max(0, current_node['E'] - final_energy)
            return path, total_energy_consumed
        
        return None, float('inf')  # Failed to reach sink
        
    except Exception as e:
        print(f"Error in find_optimal_path_drl: {e}")
        return None, float('inf')

def update_energy_after_transmission(network, path, energy_consumed):
    """Update node energies after packet transmission"""
    if not path or len(path) < 2:
        return
    
    # Energy has already been updated in find_optimal_path_drl
    # This function can be used for additional energy bookkeeping if needed
    
    # Update energy statistics for path nodes
    for node in path:
        if node['id'] in [n['id'] for n in network]:
            # Find the actual network node and update its energy
            network_node = next((n for n in network if n['id'] == node['id']), None)
            if network_node:
                network_node['E'] = node['E']  # Sync energies
                
                # Mark node as dead if energy is below threshold
                if network_node['E'] <= DEAD_NODE_THRESHOLD:
                    network_node['cond'] = 0

############################## Plotting Functions ##############################

def plot_simulation_results(metrics_history, gnn_losses, drl_losses):
    """Plot comprehensive simulation results"""
    rounds = [m['round'] for m in metrics_history]
    
    # Create subplots with additional metrics
    fig, axes = plt.subplots(6, 2, figsize=(18, 24))  # Increased rows for more plots
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]
    ax7, ax8 = axes[3]
    ax9, ax10 = axes[4]  # New row for plots
    ax11, ax12 = axes[5]  # New row for plots
    
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
    energy_variance = [m.get('energy_variance', 0) for m in metrics_history]  # Use .get for safety
    ax9.plot(rounds, energy_variance, 'p-', linewidth=2, label='Energy Variance')
    ax9.set_xlabel('Round')
    ax9.set_ylabel('Energy Variance (J^2)')
    ax9.set_title('Energy Variance Over Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot 10: Number of Active Cluster Heads
    # Assuming 'num_cluster_heads' is or will be added to metrics
    num_cluster_heads = [m.get('num_cluster_heads', np.nan) for m in metrics_history]  # Use np.nan for missing data
    ax10.plot(rounds, num_cluster_heads, 's-', linewidth=2, label='Active Cluster Heads')
    ax10.set_xlabel('Round')
    ax10.set_ylabel('Number of Cluster Heads')
    ax10.set_title('Active Cluster Heads Over Time')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Plot 11: DRL Epsilon Decay (Placeholder if not directly in metrics_history)
    # This might need to be passed separately or calculated if agent's history is available
    # For now, let's assume it might be added to metrics or we plot a placeholder
    drl_epsilon = [m.get('drl_epsilon', np.nan) for m in metrics_history]  # Placeholder
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


def plot_advanced_analysis(topology_analysis_history, bottleneck_history, consumption_history):
    """Plot advanced network analysis results"""
    if not topology_analysis_history:
        print("No topology analysis data available for plotting")
        return
    
    # Create figure with subplots for advanced analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]
    
    rounds = [analysis['round'] for analysis in topology_analysis_history]
    
    # Plot 1: Network Topology Metrics
    avg_degree = [analysis.get('avg_degree', 0) for analysis in topology_analysis_history]
    clustering_coeff = [analysis.get('clustering_coefficient', 0) for analysis in topology_analysis_history]
    network_density = [analysis.get('network_density', 0) for analysis in topology_analysis_history]
    
    ax1.plot(rounds, avg_degree, 'b-', label='Average Degree', linewidth=2)
    ax1.plot(rounds, clustering_coeff, 'r--', label='Clustering Coefficient', linewidth=2)
    ax1.plot(rounds, network_density, 'g:', label='Network Density', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Network Topology Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Connectivity Analysis
    sink_connectivity = [analysis.get('sink_connectivity_ratio', 0) for analysis in topology_analysis_history]
    avg_path_to_sink = [analysis.get('avg_path_to_sink', 0) for analysis in topology_analysis_history]
    # Normalize avg_path_to_sink for plotting
    max_path = max(avg_path_to_sink) if avg_path_to_sink else 1
    normalized_path = [p/max_path if max_path > 0 else 0 for p in avg_path_to_sink]
    
    ax2.plot(rounds, sink_connectivity, 'purple', label='Sink Connectivity Ratio', linewidth=2)
    ax2.plot(rounds, normalized_path, 'orange', label='Normalized Avg Path to Sink', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Ratio/Normalized Value')
    ax2.set_title('Network Connectivity Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy Distribution Analysis
    energy_gini = [analysis.get('energy_gini', 0) for analysis in topology_analysis_history]
    energy_cv = [analysis.get('energy_cv', 0) for analysis in topology_analysis_history]
    
    ax3.plot(rounds, energy_gini, 'darkred', label='Energy Gini Coefficient', linewidth=2)
    ax3.plot(rounds, energy_cv, 'darkgreen', label='Energy Coefficient of Variation', linewidth=2)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Inequality Measure')
    ax3.set_title('Energy Distribution Inequality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Critical Nodes Analysis
    critical_nodes = [analysis.get('critical_nodes', 0) for analysis in topology_analysis_history]
    critical_ratio = [analysis.get('critical_node_ratio', 0) for analysis in topology_analysis_history]
    
    ax4.plot(rounds, critical_nodes, 'darkblue', label='Number of Critical Nodes', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(rounds, critical_ratio, 'red', label='Critical Node Ratio', linewidth=2, linestyle='--')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Number of Critical Nodes', color='darkblue')
    ax4_twin.set_ylabel('Critical Node Ratio', color='red')
    ax4.set_title('Critical Nodes Analysis')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Bottleneck History
    if bottleneck_history:
        bottleneck_counts = [len(bottlenecks) for bottlenecks in bottleneck_history]
        if bottleneck_counts:
            ax5.plot(range(len(bottleneck_counts)), bottleneck_counts, 'red', linewidth=2, marker='o')
            ax5.set_xlabel('Analysis Interval')
            ax5.set_ylabel('Number of Bottlenecks')
            ax5.set_title('Network Bottlenecks Over Time')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Bottleneck Data', horizontalalignment='center', 
                    verticalalignment='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'No Bottleneck Data', horizontalalignment='center', 
                verticalalignment='center', transform=ax5.transAxes)
    
    # Plot 6: Energy Consumption Trends
    if consumption_history:
        ax6.plot(range(len(consumption_history)), consumption_history, 'green', linewidth=2)
        # Add moving average
        if len(consumption_history) > 10:
            window_size = min(10, len(consumption_history))
            moving_avg = []
            for i in range(len(consumption_history)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(consumption_history[start_idx:i+1]))
            ax6.plot(range(len(moving_avg)), moving_avg, 'darkgreen', linewidth=2, linestyle='--', label='Moving Average')
            ax6.legend()
        
        ax6.set_xlabel('Round')
        ax6.set_ylabel('Energy Consumption (J)')
        ax6.set_title('Energy Consumption Trends')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Consumption Data', horizontalalignment='center', 
                verticalalignment='center', transform=ax6.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/advanced_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Advanced analysis plots saved to 'results/advanced_analysis.png'")

def run_proactive_gnn_wsn_simulation():
    """Main simulation function that runs the proactive GNN-DRL WSN routing simulation"""
    print("Debug: Entered run_proactive_gnn_wsn_simulation")
    print("Initializing WSN...")
    try:
        network = initialize_network(NUM_NODES, FIELD_X, FIELD_Y, SINK_X, SINK_Y, TRANSMISSION_RANGE)
        print("Debug: WSN initialized with nodes:", len(network))
    except Exception as e:
        print(f"Error during WSN initialization: {e}")

    sink_pos = (SINK_X, SINK_Y)  # Define sink position
    print("Debug: Sink position set to:", sink_pos)

    print("Initializing GNN model...")
    try:
        gnn_model = WSN_GNN(NODE_FEATURE_SIZE, EDGE_FEATURE_SIZE, 
                           GNN_HIDDEN_CHANNELS, GNN_OUTPUT_SIZE).to(device)
        print("Debug: GNN model initialized")
    except Exception as e:
        print(f"Error during GNN model initialization: {e}")

    try:
        gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=LEARNING_RATE)  # Initialize optimizer
        print("Debug: GNN optimizer initialized")
    except Exception as e:
        print(f"Error during GNN optimizer initialization: {e}")

    print("Initializing DRL agent...")
    try:
        state_size = 9  # Updated to match actual state vector: [energy, x, y, dist_to_sink, hop_count, network_energy, congestion, sleep_state, duty_cycle]
        gnn_embedding_size = GNN_HIDDEN_CHANNELS
        agent = ProactiveDRLAgent(state_size, gnn_embedding_size, GNN_HIDDEN_CHANNELS)
        print("Debug: DRL agent initialized")
    except Exception as e:
        print(f"Error during DRL agent initialization: {e}")

    print("Debug: Starting simulation loop")
    # Initialize tracking variables
    network_history = []
    traffic_history = {node['id']: [] for node in network}
    metrics_history = []
    gnn_losses = []
    drl_losses = []
    energy_predictions_history = []
    print("Debug: Tracking variables initialized")

    # Initialize round counter
    round_num = 0
    print("Debug: Round counter initialized")

    print(f"Starting simulation with {NUM_NODES} nodes...")
    print(f"Initial network energy: {sum(node['E'] for node in network):.3f}J")

    print("Debug: Entering main simulation loop")
    # Simulation main loop
    print("Creating progress bar...")
    with tqdm(total=MAX_ROUNDS, desc="Simulation Progress", disable=False) as pbar:
        print("Progress bar created, starting loop...")
        while round_num < MAX_ROUNDS:
            round_num += 1
            pbar.update(1)
            
            # Check for node failures
            dead_nodes = update_dead_nodes(network)
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
                    
                    transmission_count += 1
            
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
            
            # Update sleep states and neighbor counts
            update_sleep_states(network, round_num)
            update_neighbors_count(network)
            
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

def main():
    """Main function to run the simulation"""
    global MAX_ROUNDS
    
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