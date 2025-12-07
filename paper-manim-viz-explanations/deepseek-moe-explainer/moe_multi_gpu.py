"""
MoE Inference with Experts on Different GPUs (Same Machine)
Demonstrates Expert Parallelism
"""

import torch
import torch.nn as nn
from typing import List, Dict

class ExpertFFN(nn.Module):
    """Single Expert Feed-Forward Network"""
    def __init__(self, hidden_dim=2048, ffn_dim=8192):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x shape: [num_tokens, hidden_dim]
        return self.w2(self.activation(self.w1(x)))


class MoEGate(nn.Module):
    """Router that decides which tokens go to which experts"""
    def __init__(self, hidden_dim=2048, num_experts=5):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.num_experts = num_experts
    
    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_dim]
        # Flatten for routing
        batch_size, seq_len, hidden_dim = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
        
        # Compute router logits
        router_logits = self.gate(tokens)  # [batch*seq_len, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Select top-2 experts
        top_k_weights, top_k_indices = torch.topk(router_probs, k=2, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, tokens


class MultiGPUMoELayer(nn.Module):
    """
    MoE Layer with experts distributed across multiple GPUs
    
    Strategy: Each expert lives on a different GPU
    - Expert 0 → GPU 0
    - Expert 1 → GPU 1
    - Expert 2 → GPU 2
    - etc.
    """
    def __init__(self, hidden_dim=2048, ffn_dim=8192, num_experts=5, 
                 gpu_devices=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Determine available GPUs
        if gpu_devices is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available!")
            gpu_devices = [f'cuda:{i}' for i in range(min(num_gpus, num_experts))]
        
        self.gpu_devices = gpu_devices
        print(f"Distributing {num_experts} experts across {len(gpu_devices)} GPUs")
        
        # Router stays on GPU 0 (or CPU)
        self.gate = MoEGate(hidden_dim, num_experts).to('cuda:0')
        
        # Create experts and place each on a different GPU
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = ExpertFFN(hidden_dim, ffn_dim)
            # Distribute experts across GPUs (round-robin if more experts than GPUs)
            device = gpu_devices[i % len(gpu_devices)]
            expert.to(device)
            self.experts.append(expert)
            print(f"  Expert {i} → {device}")
    
    def forward(self, hidden_states):
        """
        Forward pass with cross-GPU expert routing
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] on cuda:0
        
        Returns:
            output: [batch, seq_len, hidden_dim] on cuda:0
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Ensure input is on cuda:0 (where router is)
        hidden_states = hidden_states.to('cuda:0')
        
        # Step 1: Route tokens to experts
        top_k_weights, top_k_indices, tokens = self.gate(hidden_states)
        # top_k_weights: [batch*seq_len, 2]
        # top_k_indices: [batch*seq_len, 2]
        # tokens: [batch*seq_len, hidden_dim]
        
        num_tokens = tokens.shape[0]
        
        # Step 2: Group tokens by expert
        expert_inputs = {}  # expert_id -> tensor of tokens
        expert_weights = {}  # expert_id -> weights for those tokens
        expert_token_ids = {}  # expert_id -> original token positions
        
        for expert_id in range(self.num_experts):
            # Find which tokens are assigned to this expert
            mask = (top_k_indices == expert_id)
            token_ids, k_position = torch.where(mask)
            
            if len(token_ids) > 0:
                # Gather tokens for this expert
                expert_inputs[expert_id] = tokens[token_ids]
                expert_weights[expert_id] = top_k_weights[token_ids, k_position]
                expert_token_ids[expert_id] = token_ids
        
        # Step 3: Process each expert IN PARALLEL (key optimization!)
        expert_outputs = {}
        
        # Option A: Sequential (simple but slower)
        # for expert_id, expert_input in expert_inputs.items():
        #     device = self.gpu_devices[expert_id % len(self.gpu_devices)]
        #     expert_input_gpu = expert_input.to(device)
        #     output = self.experts[expert_id](expert_input_gpu)
        #     expert_outputs[expert_id] = output.to('cuda:0')
        
        # Option B: Parallel using threads (better!)
        import concurrent.futures
        
        def process_expert(expert_id):
            if expert_id not in expert_inputs:
                return None
            
            # Move input to expert's GPU
            device = self.gpu_devices[expert_id % len(self.gpu_devices)]
            expert_input_gpu = expert_inputs[expert_id].to(device)
            
            # Process on expert's GPU
            with torch.cuda.device(device):
                output = self.experts[expert_id](expert_input_gpu)
            
            # Move output back to cuda:0 for combining
            return output.to('cuda:0')
        
        # Execute all experts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_experts) as executor:
            futures = {executor.submit(process_expert, i): i 
                      for i in range(self.num_experts)}
            
            for future in concurrent.futures.as_completed(futures):
                expert_id = futures[future]
                result = future.result()
                if result is not None:
                    expert_outputs[expert_id] = result
        
        # Step 4: Combine expert outputs (weighted sum)
        combined_output = torch.zeros(num_tokens, hidden_dim, device='cuda:0')
        
        for expert_id, output in expert_outputs.items():
            token_ids = expert_token_ids[expert_id]
            weights = expert_weights[expert_id].unsqueeze(-1)  # [num_tokens, 1]
            weighted_output = weights * output  # [num_tokens, hidden_dim]
            
            # Scatter back to original positions
            combined_output.index_add_(0, token_ids, weighted_output)
        
        # Step 5: Reshape to original dimensions
        output = combined_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Residual connection
        return output + hidden_states


class SimpleMoEModel(nn.Module):
    """Simple MoE model for demonstration"""
    def __init__(self, hidden_dim=2048, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(50000, hidden_dim).to('cuda:0')
        self.layers = nn.ModuleList([
            MultiGPUMoELayer(hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim).to('cuda:0')
    
    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        hidden_states = self.embedding(input_ids.to('cuda:0'))
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Multi-GPU MoE Inference Example")
    print("=" * 80)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if num_gpus < 2:
        print("\nWarning: Need at least 2 GPUs for this demo!")
        print("Running with available GPUs...")
    
    # Create model with experts distributed across GPUs
    print("\n" + "=" * 80)
    print("Creating MoE Model with Distributed Experts")
    print("=" * 80)
    
    model = SimpleMoEModel(hidden_dim=512, num_layers=2)
    model.eval()
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)
    
    # Memory usage
    print("\nGPU Memory Usage:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

