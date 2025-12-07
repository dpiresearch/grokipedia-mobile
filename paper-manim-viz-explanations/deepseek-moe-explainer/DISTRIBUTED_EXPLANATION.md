# Distributed MoE Inference: Complete Guide

## Overview of PyTorch Distribution Strategies

### 1. **Data Parallelism (DP/DDP)**
- **What**: Same model replicated on each GPU, different data batches
- **Use case**: Training with large batch sizes
- **Not ideal for MoE**: Replicates all experts (wastes memory)

### 2. **Model Parallelism**
- **What**: Different parts of model on different devices
- **Use case**: Models too large for single GPU
- **For MoE**: Perfect! Each expert on different device

### 3. **Tensor Parallelism**
- **What**: Split individual tensors/layers across devices
- **Use case**: Very large individual layers
- **For MoE**: Can split expert FFNs if they're huge

### 4. **Pipeline Parallelism**
- **What**: Different layers on different devices, process in pipeline
- **Use case**: Deep models
- **For MoE**: Can combine with expert parallelism

### 5. **FSDP (Fully Sharded Data Parallel)**
- **What**: Shards model parameters, optimizer states, and gradients
- **Use case**: Training very large models
- **For MoE**: Can shard experts, but adds communication overhead

---

## Comparison Table

| Strategy | Same Machine | Multi-Machine | Communication | Best For |
|----------|--------------|---------------|---------------|----------|
| **Manual Device Placement** | ✅ Easy | ❌ No | None (GPU-GPU) | Multi-GPU, same node |
| **DataParallel** | ✅ Yes | ❌ No | AllReduce | Not for MoE |
| **DistributedDataParallel** | ✅ Yes | ✅ Yes | AllReduce | Training, not ideal for MoE |
| **RPC** | ✅ Yes | ✅ Yes | Point-to-point | **Perfect for MoE inference** |
| **FSDP** | ✅ Yes | ✅ Yes | AllGather/Reduce | MoE training |
| **Pipeline Parallel** | ✅ Yes | ✅ Yes | Send/Recv | Deep models |

---

## Architecture Details

### Multi-GPU (Same Machine)

```
┌─────────────────────────────────────────┐
│             Master Process              │
│  ┌──────────────────────────────────┐  │
│  │   Router (GPU 0)                 │  │
│  │   Tokens → [Exp 1, 2] [Exp 3, 4] │  │
│  └──────────────────────────────────┘  │
│              ↓         ↓                │
│    ┌─────────────┐  ┌─────────────┐   │
│    │   GPU 0     │  │   GPU 1     │   │
│    │ Expert 0, 1 │  │ Expert 2, 3 │   │
│    └─────────────┘  └─────────────┘   │
│              ↓         ↓                │
│  ┌──────────────────────────────────┐  │
│  │   Combine Results                │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘

Communication: PCIe/NVLink (fast!)
Latency: ~μs
Bandwidth: 32-900 GB/s (NVLink)
```

### Multi-Machine (Distributed)

```
┌──────────────────┐
│  Master Node     │
│  (Rank 0)        │
│  ┌────────────┐  │
│  │  Router    │  │
│  └────────────┘  │
│       ↓  ↓       │
└───────┼──┼───────┘
        │  │
    Network (RPC)
        │  │
    ┌───┘  └────┐
    ↓           ↓
┌────────┐  ┌────────┐
│Worker 1│  │Worker 2│
│(Rank 1)│  │(Rank 2)│
│Expert  │  │Expert  │
│ 0,1,2  │  │ 3,4    │
└────────┘  └────────┘

Communication: Ethernet/InfiniBand
Latency: ~ms
Bandwidth: 1-200 GB/s (IB)
```

---

## Code Walkthrough

### Multi-GPU Implementation

#### 1. **Router Computation** (GPU 0)
```python
# All on GPU 0
router_logits = self.gate(tokens)  # [40, 5]
top_k_weights, top_k_indices = torch.topk(router_probs, k=2)

# Result:
# Token 0 → Experts [1, 2]  (Expert 1 on GPU 0, Expert 2 on GPU 1)
# Token 1 → Experts [2, 4]  (Expert 2 on GPU 1, Expert 4 on GPU 2)
```

#### 2. **Token Dispatch** (Grouping)
```python
# Group tokens by target GPU
expert_inputs = {
    0: tokens[[0, 5, 8], :],    # 3 tokens for Expert 0 (GPU 0)
    1: tokens[[0, 2, 7], :],    # 3 tokens for Expert 1 (GPU 0)
    2: tokens[[0, 1, 9], :],    # 3 tokens for Expert 2 (GPU 1)
    ...
}
```

#### 3. **Parallel Expert Processing**
```python
def process_expert(expert_id):
    device = f'cuda:{expert_id % num_gpus}'
    
    # Move to expert's GPU
    x = expert_inputs[expert_id].to(device)
    
    # Process on that GPU
    output = experts[expert_id](x)  # Runs in parallel!
    
    # Move back to GPU 0
    return output.to('cuda:0')

# All experts run simultaneously on different GPUs
with ThreadPoolExecutor() as executor:
    results = executor.map(process_expert, range(num_experts))
```

**Key**: `ThreadPoolExecutor` + `torch.cuda.Stream` enable true parallelism

#### 4. **Result Combination** (GPU 0)
```python
# Scatter results back to token positions
for expert_id, output in expert_outputs.items():
    token_ids = expert_token_ids[expert_id]
    weights = expert_weights[expert_id]
    
    combined[token_ids] += weights * output
```

---

### Multi-Machine Implementation

#### 1. **Initialization** (Each Node)
```python
# On each machine:
rpc.init_rpc(
    name=f'worker{rank}',
    rank=rank,
    world_size=total_nodes,
    backend=TensorPipeRpcBackend  # Supports CPU and CUDA tensors
)
```

#### 2. **Remote Expert Creation** (Master → Workers)
```python
# Master creates remote references
expert_rref = rpc.remote(
    to='worker1',  # Target machine
    func=RemoteExpertWorker,  # Class to instantiate
    args=([0, 1], hidden_dim, ffn_dim)  # Expert IDs 0, 1
)

# This creates the expert ON worker1, not locally
```

#### 3. **Async RPC Calls** (Parallel Network Requests)
```python
# Send tokens to all workers in parallel
futures = {}
for expert_id in range(num_experts):
    futures[expert_id] = rpc_async(
        to=expert_worker_map[expert_id],  # e.g., 'worker1'
        func=RemoteExpertWorker.process_tokens,
        args=(expert_rref, tokens[expert_id])
    )

# Wait for all to complete
results = {i: f.wait() for i, f in futures.items()}
```

**Key**: `rpc_async` doesn't block, all requests sent simultaneously

#### 4. **Networking Details**
```python
# TensorPipe automatically:
# 1. Serializes tensors
# 2. Sends over network (TCP/IB/CUDA IPC)
# 3. Deserializes on remote
# 4. Executes function
# 5. Sends result back

# No manual socket programming needed!
```

---

## Performance Characteristics

### Multi-GPU (Same Machine)

**Pros:**
- ✅ Very low latency (~10-100 μs)
- ✅ High bandwidth (32-900 GB/s with NVLink)
- ✅ Simple to implement
- ✅ No network bottleneck

**Cons:**
- ❌ Limited to GPUs in one machine (typically 8-16)
- ❌ All experts must fit in total GPU memory

**Bottlenecks:**
- PCIe bandwidth (if not using NVLink)
- Memory copies between GPUs

### Multi-Machine (Distributed)

**Pros:**
- ✅ Scale to many machines (100s of experts)
- ✅ Each machine can have multiple GPUs
- ✅ Flexible deployment

**Cons:**
- ❌ Higher latency (~1-10 ms per RPC)
- ❌ Network bandwidth limit (1-200 GB/s)
- ❌ More complex setup

**Bottlenecks:**
- Network latency (dominant factor)
- Serialization/deserialization overhead
- Can be mitigated with:
  - Batching multiple tokens per RPC
  - Using InfiniBand instead of Ethernet
  - Async RPC to hide latency

---

## When to Use Which?

### Use Multi-GPU (Same Machine) When:
- ✅ You have 2-8 GPUs in one machine
- ✅ Your model has 5-50 experts
- ✅ Latency is critical (real-time inference)
- ✅ You want simplicity

**Example**: DeepSeek-16B with 5 experts/layer on 8x A100

### Use Multi-Machine When:
- ✅ You have 100+ experts
- ✅ Each expert is very large (>10GB)
- ✅ You need to scale beyond single machine
- ✅ Batch inference (latency less critical)

**Example**: Mixtral-8x22B (176B params) across multiple nodes

---

## FSDP for MoE (Bonus)

FSDP (Fully Sharded Data Parallel) is mainly for **training**, but can be used for inference:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

# Wrap model with FSDP
model = FSDP(
    model,
    mixed_precision=MixedPrecision(...),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    # For MoE: can use custom policy to shard experts
)

# FSDP automatically:
# - Shards parameters across GPUs/machines
# - AllGathers parameters when needed
# - Frees memory after forward pass
```

**For MoE Inference, RPC is usually better than FSDP because:**
- ✅ Less communication (point-to-point vs AllGather)
- ✅ Better for sparse activation (only used experts communicate)
- ✅ More flexible routing

**Use FSDP for MoE when:**
- Training (needs gradients)
- Need memory efficiency more than speed
- Using ZeRO-3 style sharding

---

## Optimization Tips

### 1. **Batching Tokens**
```python
# Bad: Send each token separately
for token in tokens:
    rpc_async(worker, process_token, (token,))

# Good: Batch tokens to same expert
tokens_for_expert = tokens[expert_mask]  # [N, hidden_dim]
rpc_async(worker, process_tokens, (tokens_for_expert,))
```

### 2. **Async Everywhere**
```python
# Use async RPC to hide network latency
futures = [rpc_async(...) for expert in experts]
results = [f.wait() for f in futures]  # Parallel wait
```

### 3. **Prefetching**
```python
# Start next batch's routing while waiting for current expert results
with torch.cuda.stream(stream1):
    expert_results = process_experts_async(batch1)

with torch.cuda.stream(stream2):
    routing = compute_routing(batch2)  # Overlaps with batch1
```

### 4. **Compression**
```python
# Use FP16 for network transfer
tokens_fp16 = tokens.half()
rpc_async(worker, process, (tokens_fp16,))
```

---

## Summary

| Aspect | Multi-GPU | Multi-Machine |
|--------|-----------|---------------|
| **Framework** | Manual placement + ThreadPool | torch.distributed.rpc |
| **Communication** | CUDA-to-CUDA | Network (TensorPipe) |
| **Latency** | 10-100 μs | 1-10 ms |
| **Bandwidth** | 32-900 GB/s | 1-200 GB/s |
| **Scale** | 2-16 GPUs | Unlimited machines |
| **Complexity** | Low | Medium |
| **Best for** | Real-time inference | Large-scale batch inference |

**Recommendation for DeepSeek-MoE:**
- **Development/Single Node**: Multi-GPU script
- **Production/Large Scale**: Multi-Machine RPC
- **Training**: Consider FSDP + expert parallelism

