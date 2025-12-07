"""
MoE Inference with Experts on Different Machines
Demonstrates Distributed Expert Parallelism using torch.distributed.rpc

Architecture:
- Master node (rank 0): Runs router and coordinates
- Worker nodes (rank 1, 2, 3...): Each hosts one or more experts

Communication: PyTorch Distributed RPC (Remote Procedure Call)
"""

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote, rpc_sync, rpc_async
import os
from typing import List, Dict


class ExpertFFN(nn.Module):
    """Single Expert Feed-Forward Network"""
    def __init__(self, hidden_dim=2048, ffn_dim=8192, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()
        
        # Move to GPU if available on this worker
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, x):
        """Process tokens assigned to this expert"""
        # x shape: [num_tokens, hidden_dim]
        if torch.cuda.is_available():
            x = x.cuda()
        
        output = self.w2(self.activation(self.w1(x)))
        
        # Move back to CPU for network transfer
        return output.cpu()


class RemoteExpertWorker:
    """
    Worker class that runs on each remote machine
    Hosts one or more experts
    """
    def __init__(self, expert_ids: List[int], hidden_dim=2048, ffn_dim=8192):
        self.expert_ids = expert_ids
        self.experts = nn.ModuleDict({
            str(expert_id): ExpertFFN(hidden_dim, ffn_dim, expert_id)
            for expert_id in expert_ids
        })
        
        rank = rpc.get_worker_info().id
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Worker {rank}: Initialized experts {expert_ids} on {device}")
    
    def process_tokens(self, expert_id: int, tokens: torch.Tensor, 
                       weights: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through the specified expert
        
        Args:
            expert_id: Which expert to use
            tokens: [num_tokens, hidden_dim]
            weights: [num_tokens] - routing weights
        
        Returns:
            weighted_output: [num_tokens, hidden_dim]
        """
        expert = self.experts[str(expert_id)]
        
        # Process tokens
        output = expert(tokens)  # [num_tokens, hidden_dim]
        
        # Apply weights
        weighted_output = weights.unsqueeze(-1) * output
        
        return weighted_output


class DistributedMoELayer(nn.Module):
    """
    MoE Layer with experts distributed across multiple machines
    
    This runs on the master node (rank 0)
    """
    def __init__(self, hidden_dim=2048, ffn_dim=8192, num_experts=5, 
                 expert_worker_map: Dict[int, str] = None):
        """
        Args:
            hidden_dim: Model hidden dimension
            ffn_dim: Expert FFN intermediate dimension
            num_experts: Total number of experts
            expert_worker_map: Maps expert_id -> worker_name
                Example: {0: 'worker0', 1: 'worker1', 2: 'worker0', ...}
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_worker_map = expert_worker_map
        
        # Router stays on master node
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        if torch.cuda.is_available():
            self.gate.cuda()
        
        # Create remote references to expert workers
        self.expert_rrefs = {}
        for expert_id, worker_name in expert_worker_map.items():
            # Get reference to remote expert worker
            self.expert_rrefs[expert_id] = rpc.remote(
                worker_name,
                lambda: RemoteExpertWorker([expert_id], hidden_dim, ffn_dim)
            )
        
        print(f"Master: Created {num_experts} remote experts")
    
    def forward(self, hidden_states):
        """
        Forward pass with distributed expert routing
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Step 1: Route tokens (on master)
        tokens = hidden_states.reshape(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
        
        router_logits = self.gate(tokens)
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Select top-2 experts
        top_k_weights, top_k_indices = torch.topk(router_probs, k=2, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        num_tokens = tokens.shape[0]
        
        # Step 2: Group tokens by expert
        expert_inputs = {}
        expert_weights = {}
        expert_token_ids = {}
        
        for expert_id in range(self.num_experts):
            mask = (top_k_indices == expert_id)
            token_ids, k_position = torch.where(mask)
            
            if len(token_ids) > 0:
                expert_inputs[expert_id] = tokens[token_ids].cpu()  # Move to CPU for RPC
                expert_weights[expert_id] = top_k_weights[token_ids, k_position].cpu()
                expert_token_ids[expert_id] = token_ids
        
        # Step 3: Send to remote experts ASYNCHRONOUSLY (key for performance!)
        expert_futures = {}
        
        for expert_id, expert_input in expert_inputs.items():
            worker_name = self.expert_worker_map[expert_id]
            
            # Asynchronous RPC call
            expert_futures[expert_id] = rpc_async(
                worker_name,
                RemoteExpertWorker.process_tokens,
                args=(
                    self.expert_rrefs[expert_id],
                    expert_id,
                    expert_input,
                    expert_weights[expert_id]
                )
            )
        
        # Step 4: Wait for all experts to finish and collect results
        expert_outputs = {}
        for expert_id, future in expert_futures.items():
            expert_outputs[expert_id] = future.wait()  # Block until result ready
        
        # Step 5: Combine expert outputs
        combined_output = torch.zeros(num_tokens, hidden_dim)
        
        for expert_id, weighted_output in expert_outputs.items():
            token_ids = expert_token_ids[expert_id]
            combined_output.index_add_(0, token_ids, weighted_output)
        
        # Move back to original device
        combined_output = combined_output.to(device)
        
        # Reshape and return
        output = combined_output.reshape(batch_size, seq_len, hidden_dim)
        return output + hidden_states


# ============================================================================
# INITIALIZATION AND LAUNCHER CODE
# ============================================================================

def run_worker(rank, world_size, master_addr, master_port, expert_assignments):
    """
    Run on each worker node
    
    Args:
        rank: Worker rank (0 = master, 1+ = workers)
        world_size: Total number of nodes
        master_addr: Master node address
        master_port: Master node port
        expert_assignments: Dict mapping rank -> list of expert IDs
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Initialize RPC
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=300  # 5 minutes timeout
    )
    
    worker_name = f'worker{rank}'
    
    if rank == 0:
        # Master node
        print(f"Initializing master node: {worker_name}")
        rpc.init_rpc(
            worker_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        print("Master node ready, running inference...")
        run_master_inference(world_size, expert_assignments)
        
    else:
        # Worker node
        expert_ids = expert_assignments.get(rank, [])
        print(f"Initializing worker node: {worker_name} with experts {expert_ids}")
        
        rpc.init_rpc(
            worker_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        # Create expert worker (will be accessed remotely)
        worker = RemoteExpertWorker(expert_ids, hidden_dim=512, ffn_dim=2048)
        
        print(f"Worker {rank} ready and waiting for requests...")
    
    # Block until shutdown
    rpc.shutdown()


def run_master_inference(world_size, expert_assignments):
    """Run inference on master node"""
    
    # Create expert-to-worker mapping
    expert_worker_map = {}
    for rank, expert_ids in expert_assignments.items():
        for expert_id in expert_ids:
            expert_worker_map[expert_id] = f'worker{rank}'
    
    print(f"Expert-to-worker mapping: {expert_worker_map}")
    
    # Create distributed MoE layer
    moe_layer = DistributedMoELayer(
        hidden_dim=512,
        ffn_dim=2048,
        num_experts=len(expert_worker_map),
        expert_worker_map=expert_worker_map
    )
    
    if torch.cuda.is_available():
        moe_layer.cuda()
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    hidden_dim = 512
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    if torch.cuda.is_available():
        hidden_states = hidden_states.cuda()
    
    print(f"\nInput shape: {hidden_states.shape}")
    print("Running distributed inference...\n")
    
    # Run inference
    with torch.no_grad():
        output = moe_layer(hidden_states)
    
    print(f"\nOutput shape: {output.shape}")
    print("Distributed inference complete!")


# ============================================================================
# LAUNCHER SCRIPT (Run this on each machine)
# ============================================================================

"""
To run this on multiple machines:

1. On Master Machine (e.g., 192.168.1.100):
   python moe_multi_machine.py --rank 0 --world-size 3 --master-addr 192.168.1.100

2. On Worker Machine 1 (e.g., 192.168.1.101):
   python moe_multi_machine.py --rank 1 --world-size 3 --master-addr 192.168.1.100

3. On Worker Machine 2 (e.g., 192.168.1.102):
   python moe_multi_machine.py --rank 2 --world-size 3 --master-addr 192.168.1.100

Note: Ensure all machines can reach each other over the network
      and have the same Python environment.
"""

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed MoE Inference')
    parser.add_argument('--rank', type=int, default=0, 
                       help='Rank of this process (0 = master)')
    parser.add_argument('--world-size', type=int, default=3,
                       help='Total number of processes')
    parser.add_argument('--master-addr', type=str, default='localhost',
                       help='Address of master node')
    parser.add_argument('--master-port', type=int, default=29500,
                       help='Port for master node')
    
    args = parser.parse_args()
    
    # Define which experts run on which worker
    # In this example: 5 experts distributed across 2 workers (ranks 1 and 2)
    expert_assignments = {
        1: [0, 1, 2],  # Worker 1 hosts experts 0, 1, 2
        2: [3, 4],     # Worker 2 hosts experts 3, 4
    }
    
    print("=" * 80)
    print(f"Starting Distributed MoE - Rank {args.rank}/{args.world_size}")
    print("=" * 80)
    
    run_worker(
        rank=args.rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        expert_assignments=expert_assignments
    )
    
    print(f"Rank {args.rank} shutting down.")

