import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from functools import lru_cache
import time
from typing import Dict, List, Tuple, Optional
import gc  # For garbage collection

# Performance optimization settings
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic operations for speed

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

# --- Begin migrated code from missing_functions.py ---
def check_node_failures(network):
    """Check for node failures based on energy levels"""
    dead_nodes = []
    for node in network:
        if node['cond'] == 1 and node['E'] <= DEAD_NODE_THRESHOLD:
            node['cond'] = 0  # Mark as dead
            dead_nodes.append(node)
    return dead_nodes

def process_transmissions_batch(source_nodes, sink_pos, network, agent, gnn_model, network_graph):
    """
    Optimized batch processing for multiple simultaneous transmissions
    Reduces redundant computations by sharing GNN embeddings and network graph
    """
    # Pre-compute GNN embeddings once for all transmissions
    pyg_data, node_map = graph_to_pyg_data(network_graph, network)
    
    with torch.no_grad():
        gnn_model.eval()
        if pyg_data.x.size(0) > 0:
            all_node_embeddings, _ = gnn_model(pyg_data.x.to(device), 
                                             pyg_data.edge_index.to(device),
                                             pyg_data.edge_attr.to(device) if pyg_data.edge_attr.size(0) > 0 else None)
        else:
            all_node_embeddings = torch.zeros((0, GNN_HIDDEN_CHANNELS)).to(device)
    
    batch_results = []
    
    for source_node in source_nodes:
        # Find optimal path using shared embeddings
        path, energy_consumed = find_optimal_path_drl(source_node, sink_pos, 
                                                    network, agent, gnn_model, 
                                                    network_graph)
        
        if path and energy_consumed > 0:
            # Calculate reward for this transmission
            reward = calculate_reward(path, energy_consumed, network, sink_pos)
            
            # Prepare state information using pre-computed embeddings
            current_state = agent.get_state(source_node, network, sink_pos)
            source_mapped_id = node_map.get(source_node['id'], 0)
            
            if source_mapped_id < all_node_embeddings.size(0):
                current_gnn_embedding = all_node_embeddings[source_mapped_id].unsqueeze(0)
            else:
                current_gnn_embedding = torch.zeros(1, all_node_embeddings.size(1)).to(device)
            
            batch_results.append({
                'source_node': source_node,
                'path': path,
                'energy_consumed': energy_consumed,
                'reward': reward,
                'current_state': current_state,
                'current_gnn_embedding': current_gnn_embedding
            })
    
    return batch_results

def calculate_network_metrics(network, round_num):
    """Calculate various network performance metrics"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    
    metrics = {
        'round': round_num,
        'alive_nodes': len(alive_nodes),
        'alive_percentage': len(alive_nodes) / NUM_NODES * 100,
        'total_energy': sum(node['E'] for node in alive_nodes),
        'avg_energy': sum(node['E'] for node in alive_nodes) / len(alive_nodes) if alive_nodes else 0,
        'min_energy': min(node['E'] for node in alive_nodes) if alive_nodes else 0,
        'max_energy': max(node['E'] for node in alive_nodes) if alive_nodes else 0,
        'energy_variance': np.var([node['E'] for node in alive_nodes]) if alive_nodes else 0,
        'total_traffic': sum(node.get('traffic', 0) for node in alive_nodes),
        'avg_traffic': sum(node.get('traffic', 0) for node in alive_nodes) / len(alive_nodes) if alive_nodes else 0
    }
    
    # Sleep scheduling metrics
    if ENABLE_SLEEP_SCHEDULING:
        awake_nodes = [node for node in alive_nodes if node.get('sleep_state') == 'awake']
        listen_nodes = [node for node in alive_nodes if node.get('sleep_state') == 'listen']
        asleep_nodes = [node for node in alive_nodes if node.get('sleep_state') == 'asleep']
        
        metrics.update({
            'awake_nodes': len(awake_nodes),
            'listen_nodes': len(listen_nodes),
            'asleep_nodes': len(asleep_nodes),
            'awake_percentage': len(awake_nodes) / len(alive_nodes) * 100 if alive_nodes else 0,
            'sleep_efficiency': (len(listen_nodes) + len(asleep_nodes)) / len(alive_nodes) * 100 if alive_nodes else 0,
            'avg_duty_cycle': sum(node.get('adaptive_duty', node.get('duty_cycle', DUTY_CYCLE)) 
                                for node in alive_nodes) / len(alive_nodes) if alive_nodes else 0,
            'wake_up_count': sum(node.get('wake_up_count', 0) for node in alive_nodes),
            'coverage_redundancy': sum(node.get('coverage_redundancy', 0) for node in alive_nodes) / len(alive_nodes) if alive_nodes else 0
        })
    
    # Connectivity analysis
    if len(alive_nodes) > 1:
        # Create graph for connectivity analysis
        G = nx.Graph()
        for node in alive_nodes:
            G.add_node(node['id'], pos=(node['x'], node['y']))
        
        # Add edges based on transmission range
        for i, node1 in enumerate(alive_nodes):
            for j, node2 in enumerate(alive_nodes[i+1:], i+1):
                dist = np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                if dist <= TRANSMISSION_RANGE:
                    G.add_edge(node1['id'], node2['id'])
        
        # Check connectivity
        if G.number_of_nodes() > 0:
            metrics['connectivity'] = nx.is_connected(G)
            
            # Calculate sink connectivity
            sink_reachable_nodes = []
            for node in alive_nodes:
                dist_to_sink = np.sqrt((node['x'] - SINK_X)**2 + (node['y'] - SINK_Y)**2)
                if dist_to_sink <= TRANSMISSION_RANGE:
                    sink_reachable_nodes.append(node['id'])
            
            # Count nodes that can reach sink through multi-hop
            reachable_to_sink = 0
            if sink_reachable_nodes:
                # Add sink to graph
                G.add_node(NUM_NODES, pos=(SINK_X, SINK_Y))
                for sink_neighbor_id in sink_reachable_nodes:
                    G.add_edge(sink_neighbor_id, NUM_NODES)
                
                # Count connected component containing sink
                if NUM_NODES in G.nodes:
                    sink_component = nx.node_connected_component(G, NUM_NODES)
                    reachable_to_sink = len([n for n in sink_component if n != NUM_NODES])
            metrics['sink_connectivity'] = reachable_to_sink / len(alive_nodes) * 100
    else:
        metrics['connectivity'] = False
        metrics['sink_connectivity'] = 0
    
    return metrics

def visualize_network(network, round_num, sink_pos, save_plot=True):
    """Visualize the current network state"""
    if not save_plot:
        return
    try:
        plt.figure(figsize=(12, 10))
        alive_nodes = [node for node in network if node['cond'] == 1]
        dead_nodes = [node for node in network if node['cond'] == 0]
        for node in alive_nodes:
            color = 'blue'  # Regular nodes
            marker = 'o'
            size = 50
            if node['role'] == 1:  # Cluster head
                color = 'red'
                marker = 's'
                size = 100
            elif node['role'] == 2:  # Sink
                color = 'green'
                marker = '^'
                size = 150
            alpha = 1.0
            if ENABLE_SLEEP_SCHEDULING:
                if node.get('sleep_state') == 'listen':
                    alpha = 0.7
                elif node.get('sleep_state') == 'asleep':
                    alpha = 0.3
            plt.scatter(node['x'], node['y'], c=color, marker=marker, 
                       s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
        if dead_nodes:
            dead_x = [node['x'] for node in dead_nodes]
            dead_y = [node['y'] for node in dead_nodes]
            plt.scatter(dead_x, dead_y, c='gray', marker='x', s=50, alpha=0.5)
        plt.scatter(sink_pos[0], sink_pos[1], c='green', marker='^', 
                   s=200, edgecolors='black', linewidth=2, label='Sink')
        cluster_heads = [node for node in alive_nodes if node['role'] == 1]
        for ch in cluster_heads:
            circle = plt.Circle((ch['x'], ch['y']), TRANSMISSION_RANGE, 
                              fill=False, linestyle='--', alpha=0.3, color='red')
            plt.gca().add_patch(circle)
        plt.xlim(0, FIELD_X)
        plt.ylim(0, FIELD_Y)
        plt.xlabel('X coordinate (m)')
        plt.ylabel('Y coordinate (m)')
        plt.title(f'WSN Topology - Round {round_num}\nAlive: {len(alive_nodes)}, Dead: {len(dead_nodes)}')
        plt.grid(True, alpha=0.3)
        plt.legend(['Regular Nodes', 'Cluster Heads', 'Dead Nodes', 'Sink'], 
                  loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        filename = f'network_round_{round_num:04d}.png'
        plt.savefig(os.path.join('results', filename), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in visualization: {e}")
        plt.close()

def calculate_reward(path, energy_consumed, network, sink_pos):
    """Calculate reward for DRL agent based on path quality and proactive metrics"""
    if not path:
        return -10.0  # Penalty for failed routing
    
    # Normalize energy consumption to avoid extreme values
    max_possible_energy = PACKET_SIZE * (E_ELEC + E_AMP * (TRANSMISSION_RANGE ** 2))
    normalized_energy_consumed = energy_consumed / max_possible_energy
    
    # Base reward components with better scaling
    energy_efficiency = 1.0 - min(1.0, normalized_energy_consumed)  # Reward between 0-1
    path_length_penalty = min(1.0, len(path) / 10.0)  # Normalize path length penalty
    
    # Network connectivity preservation (normalized)
    total_energy = sum(node['E'] for node in network if node['cond'] == 1)
    max_possible_total = NUM_NODES * INITIAL_ENERGY_MAX
    network_energy_ratio = total_energy / max_possible_total
    connectivity_bonus = network_energy_ratio * 0.5
    
    # Load balancing reward
    path_energies = [node['E'] for node in path]
    if len(path_energies) > 1:
        energy_variance = np.var(path_energies)
        max_variance = (INITIAL_ENERGY_MAX ** 2) / 4
        load_balance_reward = max(0, 1.0 - energy_variance / max_variance) * 0.3
    else:
        load_balance_reward = 0.5
    
    # Future viability reward
    future_viability_reward = 0
    for node in path:
        predicted_energy = node.get('predicted_energy', node['E'])
        current_energy = node['E']
        if current_energy > 0:
            future_ratio = min(2.0, predicted_energy / current_energy)  # Cap the ratio
            future_viability_reward += future_ratio
    future_viability_reward = (future_viability_reward / len(path)) * 0.5  # Scale down
    
    # Distance to sink factor (normalized)
    final_node = path[-1]
    distance_to_sink = np.sqrt((final_node['x'] - sink_pos[0])**2 + 
                              (final_node['y'] - sink_pos[1])**2)
    max_distance = np.sqrt(FIELD_X**2 + FIELD_Y**2)
    distance_reward = (1.0 - distance_to_sink / max_distance) * 1.0
    
    # Combine all reward components with better scaling
    total_reward = (3.0 * energy_efficiency +
                   connectivity_bonus +
                   load_balance_reward +
                   future_viability_reward +
                   distance_reward -
                   2.0 * path_length_penalty)
    
    # Clip the reward to a reasonable range to prevent exploding gradients
    total_reward = np.clip(total_reward, -10.0, 10.0)
    
    return total_reward
# --- End migrated code from missing_functions.py

# Enhanced performance monitoring with detailed tracking
class PerformanceTracker:
    def __init__(self):
        self.timings = {}
        self.cache_hits = {}
        self.cache_misses = {}
        self.performance_history = []
        self.memory_usage = []
        self.round_times = []
        
        # Add new timing tracking for time-based plots
        self.simulation_start_time = None
        self.round_timestamps = []  # Track elapsed time for each round
        self.first_node_death_time = None  # Track time (in seconds) when first node dies
        self.throughput_history = []  # Track throughput over time
        self.time_metrics_history = []  # Store metrics with timestamps
    
    def start_timer(self, name: str):
        self.timings[name] = time.time()
    
    def end_timer(self, name: str):
        if name in self.timings:
            elapsed = time.time() - self.timings[name]
            # Only print detailed timing for important operations
            if name.startswith("round_") and int(name.split("_")[1]) % 50 == 0:
                print(f"{name}: {elapsed:.4f}s")
            elif name in ["garbage_collection", "gnn_training", "data_transmission_batch"]:
                if elapsed > 0.1:  # Only log if significant time
                    print(f"{name}: {elapsed:.4f}s")
            return elapsed
        return 0
    
    def record_cache_hit(self, cache_name: str):
        self.cache_hits[cache_name] = self.cache_hits.get(cache_name, 0) + 1
    
    def record_cache_miss(self, cache_name: str):
        self.cache_misses[cache_name] = self.cache_misses.get(cache_name, 0) + 1
    
    def get_cache_stats(self):
        stats = {}
        for cache_name in set(list(self.cache_hits.keys()) + list(self.cache_misses.keys())):
            hits = self.cache_hits.get(cache_name, 0)
            misses = self.cache_misses.get(cache_name, 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            stats[cache_name] = {'hits': hits, 'misses': misses, 'hit_rate': hit_rate}
        return stats
    
    def record_memory_usage(self):
        """Record current memory usage if available"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
            return memory_mb
        except ImportError:
            return None
    
    def start_simulation_timer(self):
        """Start the overall simulation timer"""
        self.simulation_start_time = time.time()
    
    def get_elapsed_time(self):
        """Get elapsed time since simulation start in seconds"""
        if self.simulation_start_time is None:
            return 0
        return time.time() - self.simulation_start_time
    
    def record_round_timestamp(self):
        """Record timestamp for current round"""
        elapsed_time = self.get_elapsed_time()
        self.round_timestamps.append(elapsed_time)
        return elapsed_time
    
    def record_first_node_death(self):
        """Record the time when first node dies"""
        if self.first_node_death_time is None:
            self.first_node_death_time = self.get_elapsed_time()
    
    def record_throughput(self, bits_transmitted, time_window=1.0):
        """Record throughput in bits per second"""
        elapsed_time = self.get_elapsed_time()
        throughput_bps = bits_transmitted / time_window if time_window > 0 else 0
        self.throughput_history.append({
            'time': elapsed_time,
            'throughput_bps': throughput_bps,
            'bits_transmitted': bits_transmitted
        })
    
    def record_time_metrics(self, metrics):
        """Record metrics with timestamp for time-based analysis"""
        elapsed_time = self.get_elapsed_time()
        time_metrics = metrics.copy()
        time_metrics['elapsed_time'] = elapsed_time
        self.time_metrics_history.append(time_metrics)
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        cache_stats = self.get_cache_stats()
        avg_round_time = sum(self.round_times) / len(self.round_times) if self.round_times else 0
        
        summary = {
            'average_round_time': avg_round_time,
            'total_rounds_processed': len(self.round_times),
            'cache_statistics': cache_stats,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else None,
            'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else None
        }
        return summary
    
    def start_simulation_timer(self):
        """Start the overall simulation timer"""
        self.simulation_start_time = time.time()
    
    def get_elapsed_time(self):
        """Get elapsed time since simulation start in seconds"""
        if self.simulation_start_time is None:
            return 0
        return time.time() - self.simulation_start_time
    
    def record_round_timestamp(self):
        """Record timestamp for current round"""
        elapsed_time = self.get_elapsed_time()
        self.round_timestamps.append(elapsed_time)
        return elapsed_time
    
    def record_first_node_death(self):
        """Record the time when first node dies"""
        if self.first_node_death_time is None:
            self.first_node_death_time = self.get_elapsed_time()
    
    def record_throughput(self, bits_transmitted, time_window=1.0):
        """Record throughput in bits per second"""
        elapsed_time = self.get_elapsed_time()
        throughput_bps = bits_transmitted / time_window if time_window > 0 else 0
        self.throughput_history.append({
            'time': elapsed_time,
            'throughput_bps': throughput_bps,
            'bits_transmitted': bits_transmitted
        })
    
    def record_time_metrics(self, metrics):
        """Record metrics with timestamp for time-based analysis"""
        elapsed_time = self.get_elapsed_time()
        time_metrics = metrics.copy()
        time_metrics['elapsed_time'] = elapsed_time
        self.time_metrics_history.append(time_metrics)

# Initialize global performance tracker
perf_tracker = PerformanceTracker()
    
############################## WSN Parameters ##############################

# Sensing Field Dimensions (meters)
FIELD_X = 100
FIELD_Y = 100

# Network parameters
NUM_NODES = 100
SINK_X = 50
SINK_Y = 50
TRANSMISSION_RANGE = 20
CH_PERCENTAGE = 0.1  # 10% of nodes as Cluster Heads

# GNN parameters - Optimized for performance
GNN_HIDDEN_CHANNELS = 64   # Reduced for faster computation
GNN_NUM_LAYERS = 3         # Reduced layers for speed
PREDICTION_HORIZON = 10    # Reduced horizon for efficiency
GNN_DROPOUT = 0.1          # Reduced dropout
GNN_ATTENTION_HEADS = 2    # Fewer attention heads

# Energy parameters (all in Joules) - More realistic values
INITIAL_ENERGY_MIN = 2.0   # Increased initial energy to match previous code
INITIAL_ENERGY_MAX = 5.0  # Increased max energy for better heterogeneity
E_ELEC = 50e-9        # Energy for running transceiver circuitry (J/bit, as in thesis)
E_AMP = 100e-12       # Energy for transmitter amplifier (J/bit/m², as in thesis)
E_DA = 5e-9           # Energy for data aggregation (J/bit)
PACKET_SIZE = 4000    # Size of data packet (bits, as in thesis)
CONTROL_PACKET_SIZE = 500  # Size of control packets (bits)
E_SLEEP = 0.001       # Energy consumption during sleep mode (J/round, fixed small value)
E_LISTEN = 50e-9      # Energy for listening/idle mode (J/bit)
DEAD_NODE_THRESHOLD = 0.0  # Node is dead only when energy is zero

# Simulation parameters - Optimized
MAX_ROUNDS = 1000      # Reduced for faster testing

# Sleep Scheduling Parameters
ENABLE_SLEEP_SCHEDULING = True  # Enable/disable sleep scheduling
DUTY_CYCLE = 0.3                # Percentage of time a node is active (30%)
SLEEP_ROUND_DURATION = 1        # Duration of each sleep/wake cycle in rounds
WAKE_ROUND_DURATION = 2         # Duration of wake period in rounds
MIN_ACTIVE_NEIGHBORS = 2        # Minimum active neighbors required for coverage
COORDINATOR_DUTY_CYCLE = 0.8    # Cluster heads have higher duty cycle
ADAPTIVE_DUTY_CYCLE = True      # Enable adaptive duty cycling based on traffic
SLEEP_COORDINATION_ENERGY = 10e-9  # Energy for sleep coordination messages
PER_ROUND_VISUALIZATION = False # Toggle for per-round visualization
EXPORT_METRICS_CSV = True       # Toggle for exporting per-round metrics to CSV

# Duty cycle limits for adaptive scheduling
MIN_DUTY_CYCLE = 0.05
MAX_DUTY_CYCLE = 0.95

############################## DRL Parameters - Optimized ##############################

# DQN Hyperparameters - Optimized for faster training
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

# GNN Parameters - Optimized
NODE_FEATURE_SIZE = 9  # [energy_ratio, x, y, dist_to_sink, hop_count, network_energy, congestion, sleep_state, duty_cycle]
EDGE_FEATURE_SIZE = 3  # Edge features: distance, energy ratio, signal strength
GNN_OUTPUT_SIZE = 64   # Output embedding size from GNN (should match hidden channels)

# Proactive planning parameters - Optimized
FUTURE_DISCOUNT = 0.85  # Slightly higher discount factor
PLANNING_HORIZON = 3    # Reduced planning horizon
TRAFFIC_PREDICTION_WINDOW = 5  # Smaller window for efficiency

# Performance optimization parameters
CACHE_SIZE = 1000       # LRU cache size for computations
VISUALIZATION_INTERVAL = 50  # Less frequent visualization
METRICS_CALCULATION_INTERVAL = 5  # Calculate detailed metrics less frequently

############################## Graph Neural Network ##############################

class WSN_GNN(nn.Module):
    """Enhanced Graph Neural Network for WSN topology modeling and prediction"""
    
    def __init__(self, node_features, edge_features, hidden_channels, output_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Optimized GNN layers with fewer parameters for better performance
        self.conv1 = GATConv(node_features, hidden_channels, 
                            heads=GNN_ATTENTION_HEADS, edge_dim=edge_features, 
                            dropout=GNN_DROPOUT, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels * GNN_ATTENTION_HEADS, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization layers with momentum for faster training
        self.bn1 = nn.BatchNorm1d(hidden_channels * GNN_ATTENTION_HEADS, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_channels, momentum=0.1)
        
        # Simplified energy prediction layers for faster computation
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_channels + node_features, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Simplified route scoring layers
        self.route_scorer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Simplified criticality predictor
        self.criticality_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
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
        # Optimized forward pass with fewer operations
        x1 = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr)), inplace=True)
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)), inplace=True)
        node_embeddings = self.conv3(x2, edge_index)
        
        # Simple residual connection if dimensions match
        if x.size(-1) == node_embeddings.size(-1):
            node_embeddings.add_(x)  # In-place addition
        
        # Global pooling for graph embedding
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
        if len(path_embeddings.shape) == 1:
            path_embeddings = path_embeddings.unsqueeze(0)
        path_embedding = path_embeddings.mean(dim=0, keepdim=True)
        return self.route_scorer(path_embedding)
    
    def predict_criticality(self, node_embeddings):
        # Predict how critical each node is for network connectivity
        return self.criticality_predictor(node_embeddings)

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
        
        # Sleep scheduling features
        sleep_state_encoded = 0.0
        if current_node.get('sleep_state') == 'awake':
            sleep_state_encoded = 1.0
        elif current_node.get('sleep_state') == 'listen':
            sleep_state_encoded = 0.5
        # 'asleep' remains 0.0
        
        normalized_duty_cycle = current_node.get('adaptive_duty', current_node.get('duty_cycle', DUTY_CYCLE))
        
        # Return state as tensor with sleep features
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
        
        # Get expected Q values from current experiences
        q_expected = self.q_network(states, gnn_embeddings).gather(1, actions)
        
        # Get next Q values from target network for next states
        with torch.no_grad():
            q_targets_next = self.target_network(next_states, next_gnn_embeddings).max(1)[0].unsqueeze(1)
            # Calculate target Q values using Bellman equation
            q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        
        # Compute Huber loss for better stability (less sensitive to outliers)
        loss = F.smooth_l1_loss(q_expected, q_targets)
        
        # Minimize loss with gradient clipping for stability
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        self.optimizer.step()
        
        # Soft update of target network for better stability
        self.t_step += 1
        if self.t_step % TARGET_UPDATE == 0:
            # Use soft update instead of hard update for better stability
            tau = 0.005  # Soft update parameter
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
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
                      energy_ratio=node['E']/node.get('Eo', node['E']) if node.get('Eo', node['E']) > 0 else 0)
    
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
                0.0,  # Congestion (irrelevant for sink)
                1.0,  # Sleep state (always awake)
                1.0   # Duty cycle (always 100%)
            ])
        else:
            # Find the node in network
            node = next((n for n in network if n['id'] == node_id), None)
            if node:
                normalized_energy = node['E'] / INITIAL_ENERGY_MAX
                normalized_x = node['x'] / FIELD_X
                normalized_y = node['y'] / FIELD_Y
                # Calculate dts if not present or use existing value
                if 'dts' not in node or node['dts'] is None:
                    node['dts'] = np.sqrt((SINK_X - node['x'])**2 + (SINK_Y - node['y'])**2)
                normalized_dist_to_sink = node['dts'] / np.sqrt(FIELD_X**2 + FIELD_Y**2)
                
                # Calculate hop count if not present
                if 'hop' not in node or node['hop'] is None:
                    node['hop'] = np.ceil(node['dts'] / TRANSMISSION_RANGE)
                normalized_hop_count = node['hop'] / (FIELD_X/TRANSMISSION_RANGE)
                
                # Network energy percentage
                total_energy = sum(n['E'] for n in network if n['cond'] == 1)
                max_possible_energy = NUM_NODES * INITIAL_ENERGY_MAX
                network_energy_percentage = total_energy / max_possible_energy
                
                # Congestion
                proximity_nodes = sum(1 for n in network if n['cond'] == 1 and
                                     np.sqrt((n['x'] - node['x'])**2 + (n['y'] - node['y'])**2) <= TRANSMISSION_RANGE)
                normalized_congestion = proximity_nodes / NUM_NODES
                
                # Sleep scheduling features
                sleep_state_encoded = 0.0
                if node.get('sleep_state') == 'awake':
                    sleep_state_encoded = 1.0
                elif node.get('sleep_state') == 'listen':
                    sleep_state_encoded = 0.5
                # 'asleep' remains 0.0
                
                normalized_duty_cycle = node.get('adaptive_duty', node.get('duty_cycle', DUTY_CYCLE))
                
                x.append([
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
    
    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_map

############################## WSN Functions ##############################

def initialize_network(num_nodes, field_x, field_y, sink_x, sink_y, range_c):
    """Initialize WSN with heterogeneous energy and non-uniform placement"""
    network = []
    # Heterogeneous energy: 20% low-energy, 80% high-energy
    low_energy_count = int(0.2 * num_nodes)
    high_energy_count = num_nodes - low_energy_count
    energies = [random.uniform(0.5 * INITIAL_ENERGY_MIN, INITIAL_ENERGY_MIN) for _ in range(low_energy_count)] + \
              [random.uniform(INITIAL_ENERGY_MIN, INITIAL_ENERGY_MAX) for _ in range(high_energy_count)]
    random.shuffle(energies)
    # Non-uniform (Gaussian/clustered) placement
    cluster_centers = [(random.uniform(0.2*field_x,0.8*field_x), random.uniform(0.2*field_y,0.8*field_y)) for _ in range(3)]
    for i in range(num_nodes):
        # Assign to a random cluster center
        cx, cy = random.choice(cluster_centers)
        x = np.clip(np.random.normal(cx, field_x/8), 0, field_x)
        y = np.clip(np.random.normal(cy, field_y/8), 0, field_y)
        node = {}
        node['id'] = i
        node['x'] = x
        node['y'] = y
        node['E'] = energies[i]
        node['Eo'] = node['E']
        node['cond'] = 1
        node['dts'] = np.sqrt((sink_x - x)**2 + (sink_y - y)**2)
        node['hop'] = np.ceil(node['dts'] / range_c)
        node['role'] = 0
        node['closest'] = 0
        node['cluster'] = None
        node['prev'] = 0
        node['traffic'] = 0
        node['predicted_energy'] = node['E']
        node['energy_history'] = []
        
        # Sleep scheduling initialization
        node['sleep_state'] = 'awake'  # 'awake', 'asleep', 'listen'
        node['duty_cycle'] = DUTY_CYCLE if i % 10 != 0 else COORDINATOR_DUTY_CYCLE  # Different duty cycles
        node['sleep_timer'] = 0
        node['wake_timer'] = 0
        node['last_activity'] = 0
        node['sleep_schedule'] = []
        node['coverage_redundancy'] = 0
        node['adaptive_duty'] = DUTY_CYCLE
        node['wake_up_count'] = 0  # Track number of wake-up events
        
        network.append(node)
    return network

############################## Sleep Scheduling Functions ##############################

def calculate_coverage_redundancy(node, network):
    """Calculate how many awake neighbors can provide coverage for this node's area"""
    redundancy = 0
    for neighbor in network:
        if (neighbor['id'] != node['id'] and 
            neighbor['cond'] == 1 and 
            neighbor['sleep_state'] == 'awake'):
            distance = np.sqrt((node['x'] - neighbor['x'])**2 + (node['y'] - neighbor['y'])**2)
            if distance <= TRANSMISSION_RANGE:
                redundancy += 1
    return redundancy

def coordinate_sleep_schedule(network, cluster_heads):
    """Coordinate sleep schedules to maintain network connectivity and stagger CHs"""
    # Stagger cluster head wake times to avoid all sleeping at once
    for i, ch in enumerate(cluster_heads):
        if ch['role'] == 1:
            ch['duty_cycle'] = COORDINATOR_DUTY_CYCLE
            ch['adaptive_duty'] = COORDINATOR_DUTY_CYCLE
            ch['sleep_timer'] = i % (SLEEP_ROUND_DURATION + WAKE_ROUND_DURATION)
            ch['wake_timer'] = 0

def update_sleep_states(network, round_num):
    """Update sleep states for all nodes based on duty cycling and coverage"""
    if not ENABLE_SLEEP_SCHEDULING:
        return
    for node in network:
        if node['cond'] == 0:
            node['sleep_state'] = 'asleep'
            continue
        # Sink node is always awake
        if node['role'] == 2:
            node['sleep_state'] = 'awake'
            continue
        # Calculate adaptive duty cycle if enabled
        if ADAPTIVE_DUTY_CYCLE:
            traffic_level = node.get('traffic', 0)
            energy_ratio = node['E'] / node['Eo'] if node['Eo'] > 0 else 1.0
            node['adaptive_duty'] = adaptive_duty_cycle_adjustment(node, traffic_level, energy_ratio)
            current_duty = node['adaptive_duty']
        else:
            current_duty = node.get('duty_cycle', DUTY_CYCLE)
        # Calculate coverage redundancy
        node['coverage_redundancy'] = calculate_coverage_redundancy(node, network)
        # Update sleep/wake timers and state
        cycle_length = SLEEP_ROUND_DURATION + WAKE_ROUND_DURATION
        wake_duration = int(np.ceil(cycle_length * current_duty))
        sleep_duration = cycle_length - wake_duration
        if node['sleep_state'] == 'awake':
            node['wake_timer'] += 1
            if node['wake_timer'] >= wake_duration:
                # Only sleep if enough awake neighbors for coverage
                if node['coverage_redundancy'] >= MIN_ACTIVE_NEIGHBORS:
                    node['sleep_state'] = 'asleep'
                    node['wake_timer'] = 0
                    node['sleep_timer'] = 0
        elif node['sleep_state'] == 'asleep':
            node['sleep_timer'] += 1
            if node['sleep_timer'] >= sleep_duration:
                node['sleep_state'] = 'listen'  # Wake up to listen before fully awake
                node['sleep_timer'] = 0
        elif node['sleep_state'] == 'listen':
            # Listen for a short period, then become awake
            node['wake_timer'] += 1
            if node['wake_timer'] >= 1:
                node['sleep_state'] = 'awake'
                node['wake_timer'] = 0
                node['wake_up_count'] = node.get('wake_up_count', 0) + 1
        # Energy consumption based on state (fixed per round, not per bit)
        if node['sleep_state'] == 'asleep':
            node['E'] -= E_SLEEP
        elif node['sleep_state'] == 'listen':
            node['E'] -= E_LISTEN
        # Clamp energy to non-negative
        node['E'] = max(0, node['E'])

def get_awake_neighbors(node, network):
    """Return list of awake neighbors within transmission range"""
    awake_neighbors = []
    for neighbor in network:
        if (neighbor['id'] != node['id'] and neighbor['cond'] == 1 and neighbor['sleep_state'] == 'awake'):
            distance = np.sqrt((node['x'] - neighbor['x'])**2 + (node['y'] - neighbor['y'])**2)
            if distance <= TRANSMISSION_RANGE:
                awake_neighbors.append(neighbor)
    return awake_neighbors

def wake_up_nodes_for_routing(network, path):
    """Force nodes along a routing path to be awake for transmission"""
    if not ENABLE_SLEEP_SCHEDULING:
        return
    for node in path:
        if node['cond'] == 1 and node['sleep_state'] != 'awake':
            node['sleep_state'] = 'awake'
            node['wake_timer'] = 0
            node['wake_up_count'] = node.get('wake_up_count', 0) + 1

def predict_traffic_patterns(network, traffic_history, window_size=TRAFFIC_PREDICTION_WINDOW):
    """Predict future traffic for each node using moving average"""
    predictions = {}
    for node in network:
        node_id = node['id']
        history = traffic_history.get(node_id, [])
        if len(history) >= window_size:
            predictions[node_id] = np.mean(history[-window_size:])
        elif history:
            predictions[node_id] = np.mean(history)
        else:
            predictions[node_id] = 0
    return predictions

def select_cluster_heads(network, ch_percentage, energy_predictions=None):
    """Select cluster heads based on highest energy or predicted energy"""
    alive_nodes = [node for node in network if node['cond'] == 1]
    num_ch = max(1, int(len(alive_nodes) * ch_percentage))
    if energy_predictions:
        # Use predicted energy for selection
        sorted_nodes = sorted(alive_nodes, key=lambda n: energy_predictions.get(n['id'], n['E']), reverse=True)
    else:
        sorted_nodes = sorted(alive_nodes, key=lambda n: n['E'], reverse=True)
    cluster_heads = sorted_nodes[:num_ch]
    for node in network:
        node['role'] = 1 if node in cluster_heads else 0
    return cluster_heads

def form_clusters(network, cluster_heads):
    """Assign each node to the nearest cluster head"""
    for node in network:
        if node['cond'] == 1 and node['role'] != 1:
            min_dist = float('inf')
            closest_ch = None
            for ch in cluster_heads:
                dist = np.sqrt((node['x'] - ch['x'])**2 + (node['y'] - ch['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_ch = ch['id']
            node['cluster'] = closest_ch
    # Optionally, return cluster assignments
    return {node['id']: node.get('cluster', None) for node in network if node['cond'] == 1}

def adaptive_duty_cycle_adjustment(node, traffic_level, energy_ratio):
    """Adjust duty cycle based on traffic and energy (simple linear scaling)"""
    base = DUTY_CYCLE
    # Increase duty cycle if traffic is high, decrease if energy is low
    duty = base + 0.2 * traffic_level - 0.2 * (1 - energy_ratio)
    return np.clip(duty, MIN_DUTY_CYCLE, MAX_DUTY_CYCLE)

def train_gnn_model(gnn_model, gnn_optimizer, network_history):
    """Train the GNN model on network history (returns dummy loss or raises error if not implemented)"""
    # Placeholder implementation: raise error to ensure user is aware
    raise NotImplementedError("train_gnn_model is a placeholder. Please implement GNN training logic.")

def predict_future_energies(gnn_model, network):
    """Predict future energies using GNN model (returns current energies or raises error if not implemented)"""
    # Placeholder implementation: raise error to ensure user is aware
    raise NotImplementedError("predict_future_energies is a placeholder. Please implement GNN-based energy prediction.")

def find_optimal_path_drl(source_node, sink_pos, network=None, *args, **kwargs):
    """Find optimal path using DRL (returns direct path and estimated energy or raises error if not implemented)"""
    # Placeholder implementation: raise error to ensure user is aware
    raise NotImplementedError("find_optimal_path_drl is a placeholder. Please implement DRL-based path finding.")

############################## Energy Update Function ##############################

def update_energy_after_transmission(network, path, energy_consumed):
    """Update energy for nodes along a path after transmission (vectorized for performance)"""
    if not path or energy_consumed <= 0:
        return
    per_node_energy = energy_consumed / len(path)
    for node in path:
        node['E'] -= per_node_energy
        node['E'] = max(0, node['E'])

############################## Main Simulation ##############################

def run_proactive_gnn_wsn_simulation():
    """Run the main WSN simulation with proactive GNN-DRL routing and comprehensive optimizations"""
    try:
        print("="*70)
        print("PROACTIVE GNN-DRL WSN ROUTING SIMULATION")
        print("="*70)
        
        # Performance tracking initialization
        simulation_start_time = time.time()
        perf_tracker.start_timer("total_simulation")
        perf_tracker.start_simulation_timer()  # Start time-based tracking
        
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
        state_size = 9  # Updated to match the actual state features including sleep scheduling
        gnn_embedding_size = GNN_HIDDEN_CHANNELS  # Use the hidden channel size as embedding size
        agent = ProactiveDRLAgent(state_size, gnn_embedding_size, GNN_HIDDEN_CHANNELS)
        
        # Initialize tracking variables with memory-efficient structures
        network_history = []
        traffic_history = {node['id']: [] for node in network}
        metrics_history = []
        gnn_losses = []
        drl_losses = []
        energy_predictions_history = []
        first_node_death_round = None  # Initialize first node death tracking
        
        # Initialize round counter
        round_num = 0
        
        print(f"Starting simulation with {NUM_NODES} nodes...")
        print(f"Initial network energy: {sum(node['E'] for node in network):.3f}J")
        print(f"Performance optimizations: Batch processing, Memory management, Vectorized operations")
        
    except Exception as e:
        print(f"Error during simulation initialization: {e}")
        return None
    
    # Initialize snapshots list for visualization
    snapshots = []
    
    # Main simulation loop with comprehensive error handling and memory management
    try:
        with tqdm(total=MAX_ROUNDS, desc="Simulation Progress") as pbar:
            while round_num < MAX_ROUNDS:
                round_num += 1
                pbar.update(1)
                
                # Performance tracking for this round
                perf_tracker.start_timer(f"round_{round_num}")
                
                
                # Record timestamp for time-based analysis
                elapsed_time = perf_tracker.record_round_timestamp()
                
                # Memory management - Force garbage collection every 50 rounds
                if round_num % 50 == 0:
                    perf_tracker.start_timer("garbage_collection")
                    collected = gc.collect()
                    perf_tracker.end_timer("garbage_collection")
                    if round_num % 100 == 0:
                        print(f"\nMemory cleanup: {collected} objects collected")
                
                # Check for node failures
                perf_tracker.start_timer("node_failure_check")
                dead_nodes = check_node_failures(network)
                alive_nodes = [node for node in network if node['cond'] == 1]
                perf_tracker.end_timer("node_failure_check")
                
                # Track first node death round
                if first_node_death_round is None and len(alive_nodes) < NUM_NODES:
                    first_node_death_round = round_num
                    perf_tracker.record_first_node_death()  # Record time of first node death
                    print(f"\nFirst node death detected at round {round_num} (time: {perf_tracker.first_node_death_time:.2f}s)")
                
                               
                
                if len(alive_nodes) == 0:
                    print(f"\nAll nodes dead at round {round_num}")
                    break
                
                # Optimized network state storage with memory management
                perf_tracker.start_timer("network_state_storage")
                # Use shallow copy for better memory efficiency and only store essential data
                essential_network_state = [{
                    'id': node['id'], 'E': node['E'], 'x': node['x'], 'y': node['y'], 
                    'cond': node['cond'], 'role': node['role'], 'traffic': node['traffic']
                } for node in network]
                network_history.append(essential_network_state)
                if len(network_history) > 50:  # Keep only recent history
                    network_history = network_history[-50:]
                perf_tracker.end_timer("network_state_storage")
                
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
                
                # Update sleep scheduling states (before data transmission)
                if ENABLE_SLEEP_SCHEDULING:
                    update_sleep_states(network, round_num)
                
                # Optimized data transmission with enhanced batch processing
                perf_tracker.start_timer("data_transmission_batch")
                num_transmissions = min(10, len(alive_nodes))  # Simulate 10 random transmissions
                
                # Pre-select source nodes for batch processing
                non_ch_nodes = [node for node in alive_nodes if node['role'] == 0]
                if len(non_ch_nodes) >= num_transmissions:
                    # Batch selection of source nodes for efficiency
                    source_nodes = random.sample(non_ch_nodes, num_transmissions)
                    
                    # Create network graph once for all transmissions (optimization)
                    network_graph = create_network_graph(network, sink_pos, TRANSMISSION_RANGE)
                    
                    # Use optimized batch processing function
                    batch_results = process_transmissions_batch(source_nodes, sink_pos, network, 
                                                              agent, gnn_model, network_graph)
                    
                    # Process all successful transmissions
                    for result in batch_results:
                        # Update network energy
                        update_energy_after_transmission(network, result['path'], result['energy_consumed'])
                        
                        # Get next state after energy update
                        next_state = agent.get_state(result['source_node'], network, sink_pos)
                        
                        # Store experience
                        action_idx = 0  # Simplified action representation
                        done = result['source_node']['E'] <= DEAD_NODE_THRESHOLD
                        
                        agent.memory.push(result['current_state'], result['current_gnn_embedding'], 
                                        action_idx, result['reward'], next_state, 
                                        result['current_gnn_embedding'], done)
                        
                        # Update traffic history efficiently
                        for path_node in result['path']:
                            traffic_history[path_node['id']].append(1)
                            if len(traffic_history[path_node['id']]) > 20:
                                traffic_history[path_node['id']] = traffic_history[path_node['id']][-20:]
                    
                    # Batch DRL training (more efficient than individual training)
                    if len(agent.memory) > WARMUP_STEPS and len(batch_results) > 0:
                        # Train multiple times on the batch for better learning
                        training_iterations = min(3, len(batch_results))
                        for _ in range(training_iterations):
                            loss = agent.learn()
                            if loss is not None:
                                drl_losses.append(loss)
                        
                        # Update exploration rate once per batch
                        agent.update_epsilon()
                    
                    # Record performance metrics
                    perf_tracker.round_times.append(perf_tracker.end_timer("data_transmission_batch"))
                    
                    # Record memory usage periodically
                    if round_num % 20 == 0:
                        memory_usage = perf_tracker.record_memory_usage()
                        if memory_usage and round_num % 100 == 0:
                            print(f"Memory usage: {memory_usage:.1f} MB")
                
                else:
                    perf_tracker.end_timer("data_transmission_batch")
                
                # Optimized GNN training with memory management
                perf_tracker.start_timer("gnn_training")
                if len(network_history) > 1 and round_num % GNN_TRAINING_INTERVAL == 0:
                    # Clear gradients and free unused memory before training
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    loss = train_gnn_model(gnn_model, gnn_optimizer, network_history)
                    if loss is not None:
                        gnn_losses.append(loss)
                    
                    # Clean up after training
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                perf_tracker.end_timer("gnn_training")
                
                # Performance-optimized metrics calculation
                perf_tracker.start_timer("metrics_calculation")
                metrics = calculate_network_metrics(network, round_num)
                metrics_history.append(metrics)
                
                # Record time-based metrics for new plots
                perf_tracker.record_time_metrics(metrics)
                
                # Calculate and record throughput (bits per second)
                if len(metrics_history) >= 2:
                    prev_traffic = metrics_history[-2]['total_traffic']
                    current_traffic = metrics['total_traffic']
                    traffic_increase = current_traffic - prev_traffic
                    bits_transmitted = traffic_increase * PACKET_SIZE  # Convert packets to bits
                    
                    # Record throughput (assuming 1 second per measurement interval)
                    if round_num > 1:
                        time_diff = perf_tracker.round_timestamps[-1] - perf_tracker.round_timestamps[-2] if len(perf_tracker.round_timestamps) >= 2 else 1.0
                        perf_tracker.record_throughput(bits_transmitted, time_diff)
                
                # Periodic memory cleanup for metrics history
                if len(metrics_history) > 1000:  # Keep only last 1000 metrics
                    metrics_history = metrics_history[-1000:]
                perf_tracker.end_timer("metrics_calculation")
                
                # End round performance tracking
                round_time = perf_tracker.end_timer(f"round_{round_num}")
                
                # Enhanced progress reporting with performance metrics
                if round_num % 10 == 0 or round_num == MAX_ROUNDS - 1:
                    print(f"\nRound {round_num}/{MAX_ROUNDS}")
                    print(f"Alive Nodes: {metrics['alive_nodes']} ({metrics['alive_percentage']:.1f}%)")
                    print(f"Total Energy: {metrics['total_energy']:.2f}J")
                    print(f"Avg Energy: {metrics['avg_energy']:.2f}J")
                    print(f"Min Energy: {metrics['min_energy']:.2f}J")
                    print(f"Max Energy: {metrics['max_energy']:.2f}J")
                    print(f"Energy Variance: {metrics['energy_variance']:.2f}")
                    print(f"Total Traffic: {metrics['total_traffic']}")
                    print(f"Avg Traffic: {metrics['avg_traffic']:.2f}")
                    
                    if ENABLE_SLEEP_SCHEDULING and 'awake_nodes' in metrics:
                        print(f"Sleep Status - Awake: {metrics['awake_nodes']}, Listen: {metrics['listen_nodes']}, Asleep: {metrics['asleep_nodes']}")
                        print(f"Awake Percentage: {metrics['awake_percentage']:.1f}%")
                        print(f"Sleep Efficiency: {metrics['sleep_efficiency']:.1f}%")
                        print(f"Avg Duty Cycle: {metrics['avg_duty_cycle']:.3f}")
                        print(f"Wake-up Events: {metrics['wake_up_count']}")
                    
                    if 'connectivity' in metrics:
                        print(f"Network Connectivity: {'Connected' if metrics['connectivity'] else 'Disconnected'}")
                    if 'sink_connectivity' in metrics:
                        print(f"Sink Connectivity: {metrics['sink_connectivity']:.1f}%")
                
                # Store metrics every 100 rounds for later visualization
                if round_num % 100 == 0 or round_num == 1:
                    # Save a snapshot of the network and metrics
                    import copy
                    snapshots.append({
                        'round': round_num,
                        'network': copy.deepcopy(network),
                        'metrics': metrics.copy(),
                        'gnn_losses': gnn_losses[:],
                        'drl_losses': drl_losses[:],
                        'alive_nodes': [node['id'] for node in network if node['cond'] == 1],
                        'dead_nodes': [node['id'] for node in network if node['cond'] == 0],
                        'first_node_death_round': first_node_death_round
                    })
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Set default value for first_node_death_round if no nodes died
    if first_node_death_round is None:
        first_node_death_round = MAX_ROUNDS

    # Visualize snapshots if available
    for snapshot in snapshots:
        round_num_snap = snapshot['round']
        network_snap = snapshot['network']
        visualize_network(network_snap, round_num_snap, sink_pos, save_plot=True)
    
    print("Simulation completed.")
    print(f"Network lifetime: {round_num} rounds")
    print(f"First node death occurred at round: {first_node_death_round}")
    print(f"Final alive nodes: {metrics_history[-1]['alive_nodes']}/{NUM_NODES}")
    
    # Create comprehensive results
    df = pd.DataFrame(metrics_history)
    
    # Create separate directories for different result sets
    results_dirs = ['results1']
    
    for fig_dir in results_dirs:
        os.makedirs(fig_dir, exist_ok=True)

        # 1. Round vs Network Lifetime (Alive nodes)
        plt.figure(figsize=(10, 6))
        plt.plot(df['round'], df['alive_nodes'], 'b-', linewidth=2, label='Alive Nodes')
        plt.xlabel('Round')
        plt.ylabel('Number of Alive Nodes')
        plt.title('Network Lifetime: Alive Nodes Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'round_vs_network_lifetime.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Round vs Network Energy Consumption
        plt.figure(figsize=(10, 6))
        plt.plot(df['round'], df['total_energy'], 'r-', linewidth=2, label='Total Residual Energy')
        plt.xlabel('Round')
        plt.ylabel('Total Residual Energy (J)')
        plt.title('Network Energy Depletion Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'round_vs_network_energy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Round vs Network Connectivity
        plt.figure(figsize=(10, 6))
        connectivity_values = df['connectivity'].astype(int)
        plt.plot(df['round'], connectivity_values, 'g-', linewidth=2, label='Network Connectivity')
        plt.xlabel('Round')
        plt.ylabel('Network Connectivity (1=Connected, 0=Disconnected)')
        plt.title('Network Connectivity Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'round_vs_network_connectivity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Round vs Traffic Distribution (Total Traffic)
        plt.figure(figsize=(10, 6))
        plt.plot(df['round'], df['total_traffic'], 'm-', linewidth=2, label='Total Traffic')
        plt.xlabel('Round')
        plt.ylabel('Total Traffic (Packets)')
        plt.title('Network Traffic Distribution Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'round_vs_traffic_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Training Iteration vs GNN Training Loss
        if len(gnn_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(gnn_losses)), gnn_losses, 'c-', linewidth=2, label='GNN Training Loss')
            plt.xlabel('Training Iteration')
            plt.ylabel('GNN Loss')
            plt.title('GNN Training Loss Over Iterations')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'iteration_vs_gnn_loss.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 6. Training Iteration vs DRL Training Loss
        if len(drl_losses) > 0:
            plt.figure(figsize=(10,  6))
            plt.plot(range(len(drl_losses)), drl_losses, 'orange', linewidth=2, label='DRL Training Loss')
           
            plt.xlabel('Training Iteration')
            plt.ylabel('DRL Loss')

            plt.title('DRL Training Loss Over Iterations')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'iteration_vs_drl_loss.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 7. Round vs First Node Death Time
        plt.figure(figsize=(10, 6))
        plt.plot(df['round'], df['alive_nodes'], 'b-', linewidth=2, label='Alive Nodes')
        plt.axvline(first_node_death_round, color='r', linestyle='--', linewidth=2, 
                   label=f'First Node Death (Round {first_node_death_round})')
        plt.xlabel('Round')
        plt.ylabel('Number of Alive Nodes')
        plt.title('Network Lifetime with First Node Death Marker')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'round_vs_first_node_death.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 8. Round vs Network Throughput (Average Traffic)
        plt.figure(figsize=(10, 6))
        plt.plot(df['round'], df['avg_traffic'], 'purple', linewidth=2, label='Average Traffic per Node')
        plt.xlabel('Round')
        plt.ylabel('Average Traffic per Node')
        plt.title('Network Throughput Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'round_vs_network_throughput.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 9. Energy Distribution Analysis
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(df['round'], df['avg_energy'], 'g-', linewidth=2, label='Average Energy')
        plt.xlabel('Round')
        plt.ylabel('Average Energy (J)')
        plt.title('Average Node Energy')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(df['round'], df['min_energy'], 'r-', linewidth=2, label='Minimum Energy')
        plt.xlabel('Round')
        plt.ylabel('Minimum Energy (J)')
        plt.title('Minimum Node Energy')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(df['round'], df['max_energy'], 'b-', linewidth=2, label='Maximum Energy')
        plt.xlabel('Round')
        plt.ylabel('Maximum Energy (J)')
        plt.title('Maximum Node Energy')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(df['round'], df['energy_variance'], 'm-', linewidth=2, label='Energy Variance')
        plt.xlabel('Round')
        plt.ylabel('Energy Variance')
        plt.title('Energy Distribution Variance')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'energy_distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 10. Sink Connectivity Analysis
        if 'sink_connectivity' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['round'], df['sink_connectivity'], 'navy', linewidth=2, label='% Nodes Connected to Sink')
            plt.xlabel('Round')
            plt.ylabel('Sink Connectivity (%)')
            plt.title('Percentage of Nodes Connected to Sink Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'round_vs_sink_connectivity.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 11. Sleep Scheduling Analysis (if enabled)
        if ENABLE_SLEEP_SCHEDULING and 'awake_nodes' in metrics_history[-1]:
            # Sleep State Distribution
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(df['round'], df['awake_nodes'], 'g-', linewidth=2, label='Awake Nodes')
            plt.plot(df['round'], df['listen_nodes'], 'orange', linewidth=2, label='Listen Nodes')
            plt.plot(df['round'], df['asleep_nodes'], 'r-', linewidth=2, label='Asleep Nodes')
            plt.xlabel('Round')
            plt.ylabel('Number of Nodes')
            plt.title('Sleep State Distribution Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            plt.plot(df['round'], df['awake_percentage'], 'g-', linewidth=2, label='Awake %')
            plt.plot(df['round'], df['sleep_efficiency'], 'r-', linewidth=2, label='Sleep Efficiency %')
            plt.xlabel('Round')
            plt.ylabel('Percentage (%)')
            plt.title('Awake vs Sleep Efficiency')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            plt.plot(df['round'], df['avg_duty_cycle'], 'purple', linewidth=2, label='Avg Duty Cycle')
            plt.xlabel('Round')
            plt.ylabel('Duty Cycle')
            plt.title('Average Duty Cycle Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            plt.plot(df['round'], df['wake_up_count'], 'cyan', linewidth=2, label='Wake-up Events')
            plt.xlabel('Round')
            plt.ylabel('Number of Wake-ups')
            plt.title('Wake-up Events Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'sleep_scheduling_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Energy Efficiency with Sleep Scheduling
            plt.figure(figsize=(10, 6))
            energy_efficiency = df['total_energy'] / df['awake_nodes']  # Energy per awake node
            plt.plot(df['round'], energy_efficiency, 'darkgreen', linewidth=2, label='Energy per Awake Node')
            plt.xlabel('Round')
            plt.ylabel('Energy per Awake Node (J)')
            plt.title('Energy Efficiency with Sleep Scheduling')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'sleep_energy_efficiency.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 12. First Node Death Time vs Epoch (NEW)
        if perf_tracker.first_node_death_time is not None:
            plt.figure(figsize=(10, 6))
            epochs = list(range(1, len(perf_tracker.round_timestamps) + 1))
            first_death_epoch = first_node_death_round if first_node_death_round else len(epochs)
            first_death_time = perf_tracker.first_node_death_time
            
            plt.axhline(y=first_death_time, color='r', linestyle='--', linewidth=2, 
                       label=f'First Node Death Time ({first_death_time:.2f}s)')
            plt.axvline(x=first_death_epoch, color='orange', linestyle='--', linewidth=2, 
                       label=f'First Node Death Epoch ({first_death_epoch})')
            plt.scatter([first_death_epoch], [first_death_time], color='red', s=100, zorder=5)
            
            plt.xlabel('Epoch (Round)')
            plt.ylabel('Time (seconds)')
            plt.title('First Node Death Time vs Epoch')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'first_node_death_time_vs_epoch.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 13. Alive Nodes vs Time (seconds) (NEW)
        if len(perf_tracker.time_metrics_history) > 0:
            plt.figure(figsize=(10, 6))
            time_data = [m['elapsed_time'] for m in perf_tracker.time_metrics_history]
            alive_data = [m['alive_nodes'] for m in perf_tracker.time_metrics_history]
            
            plt.plot(time_data, alive_data, 'b-', linewidth=2, label='Alive Nodes')
            if perf_tracker.first_node_death_time is not None:
                plt.axvline(x=perf_tracker.first_node_death_time, color='r', linestyle='--', linewidth=2, 
                           label=f'First Node Death ({perf_tracker.first_node_death_time:.2f}s)')
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Number of Alive Nodes')
            plt.title('Network Lifetime: Alive Nodes vs Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'alive_nodes_vs_time_seconds.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 14. Throughput (bit/sec) vs Time (seconds) (NEW)
        if len(perf_tracker.throughput_history) > 0:
            plt.figure(figsize=(10, 6))
            throughput_times = [t['time'] for t in perf_tracker.throughput_history]
            throughput_values = [t['throughput_bps'] for t in perf_tracker.throughput_history]
            
            plt.plot(throughput_times, throughput_values, 'g-', linewidth=2, label='Throughput (bits/sec)')
            plt.fill_between(throughput_times, throughput_values, alpha=0.3, color='green')
            
            # Add average throughput line
            if throughput_values:
                avg_throughput = sum(throughput_values) / len(throughput_values)
                plt.axhline(y=avg_throughput, color='orange', linestyle='--', linewidth=2, 
                           label=f'Average Throughput ({avg_throughput:.0f} bits/sec)')
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Throughput (bits/sec)')
            plt.title('Network Throughput vs Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'throughput_vs_time_seconds.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Performance metrics plots saved in: {', '.join(results_dirs)}/")
    print(f"New time-based plots added:")
    print(f"  - First Node Death Time vs Epoch")
    print(f"  - Alive Nodes vs Time (seconds)")
    print(f"  - Throughput (bits/sec) vs Time (seconds)")
    
    # Save simulation summary
    summary = {
        'simulation_parameters': {
            'num_nodes': NUM_NODES,
            'field_size': f"{FIELD_X}x{FIELD_Y}m",
            'max_rounds': MAX_ROUNDS,
            'transmission_range': TRANSMISSION_RANGE,
            'initial_energy_range': f"{INITIAL_ENERGY_MIN}-{INITIAL_ENERGY_MAX}J",
            'packet_size': PACKET_SIZE,
            'sleep_scheduling_enabled': ENABLE_SLEEP_SCHEDULING,
            'duty_cycle': DUTY_CYCLE,
            'adaptive_duty_cycle': ADAPTIVE_DUTY_CYCLE,
            'coordinator_duty_cycle': COORDINATOR_DUTY_CYCLE,
        },
        'results': {
            'network_lifetime': round_num,
            'first_node_death_round': first_node_death_round,
            'final_alive_nodes': metrics_history[-1]['alive_nodes'],
            'final_energy_percentage': (metrics_history[-1]['total_energy'] / 
                                       (NUM_NODES * INITIAL_ENERGY_MAX)) * 100,
            'total_packets_transmitted': metrics_history[-1]['total_traffic'],
        }
    }
    
    # Add sleep scheduling results if enabled
    if ENABLE_SLEEP_SCHEDULING and 'awake_nodes' in metrics_history[-1]:
        summary['results'].update({
            'final_awake_percentage': metrics_history[-1]['awake_percentage'],
            'final_sleep_efficiency': metrics_history[-1]['sleep_efficiency'],
            'final_avg_duty_cycle': metrics_history[-1]['avg_duty_cycle'],
            'total_wake_up_events': metrics_history[-1]['wake_up_count'],
            'avg_coverage_redundancy': metrics_history[-1]['coverage_redundancy'],
        })
    
    # Save summary to JSON file
    import json
    for fig_dir in results_dirs:
        with open(os.path.join(fig_dir, 'simulation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"Simulation summary saved in each results directory.")
    
    # Add comprehensive performance summary with optimization metrics
    perf_summary = perf_tracker.get_performance_summary()
    cache_stats = perf_tracker.get_cache_stats()
    
    # Enhanced performance reporting
    print("\n" + "="*70)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("="*70)
    
    if perf_summary['average_round_time'] > 0:
        print(f"Average Round Processing Time: {perf_summary['average_round_time']:.4f}s")
        print(f"Total Rounds Processed: {perf_summary['total_rounds_processed']}")
        estimated_total_time = perf_summary['average_round_time'] * round_num
        print(f"Estimated Total Simulation Time: {estimated_total_time:.2f}s")
    
    if perf_summary['peak_memory_mb']:
        print(f"Peak Memory Usage: {perf_summary['peak_memory_mb']:.1f} MB")
        print(f"Average Memory Usage: {perf_summary['avg_memory_mb']:.1f} MB")
    
    # Cache performance statistics
    if cache_stats:
        print(f"\nCache Performance:")
        for cache_name, stats in cache_stats.items():
            hit_rate = stats['hit_rate']
            print(f"  {cache_name}: {stats['hits']} hits, {stats['misses']} misses, {hit_rate:.1f}% hit rate")
    
    # Optimization impact analysis
    print(f"\nOptimization Impact:")
    print(f"  Batch Processing: Enabled (10 transmissions per batch)")
    print(f"  Memory Management: Active (GC every 50 rounds)")
    print(f"  GNN Embeddings: Pre-computed and cached")
    print(f"  Distance Calculations: Vectorized with LRU cache")
    print(f"  Network State Storage: Optimized (essential data only)")
    
    # Add performance metrics to simulation summary
    summary['performance_metrics'] = {
        'average_round_time_seconds': perf_summary['average_round_time'],
        'peak_memory_mb': perf_summary['peak_memory_mb'],
        'cache_statistics': cache_stats,
        'optimization_features': [
            'batch_processing',
            'memory_management', 
            'vectorized_computations',
            'pre_computed_embeddings',
            'lru_caching'
        ],
        # Add new time-based metrics
        'first_node_death_time_seconds': perf_tracker.first_node_death_time,
        'total_simulation_time_seconds': perf_tracker.get_elapsed_time(),
        'average_throughput_bps': sum(t['throughput_bps'] for t in perf_tracker.throughput_history) / len(perf_tracker.throughput_history) if perf_tracker.throughput_history else 0,
        'total_bits_transmitted': sum(t['bits_transmitted'] for t in perf_tracker.throughput_history) if perf_tracker.throughput_history else 0
    }
    
    # Save enhanced summary with performance metrics
    import json
    for fig_dir in results_dirs:
        with open(os.path.join(fig_dir, 'simulation_summary_with_performance.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"\nEnhanced simulation summary with performance metrics saved.")
    print("="*70)
    
    return summary

# Run the optimized simulation
if __name__ == "__main__":
    print("Starting optimized DRL-GNN-WSN simulation...")
    simulation_results = run_proactive_gnn_wsn_simulation()
