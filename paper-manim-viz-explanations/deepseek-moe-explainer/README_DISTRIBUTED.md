# Distributed MoE Inference - Usage Guide

This directory contains implementations for running MoE (Mixture of Experts) inference across multiple GPUs and multiple machines.

## Files

1. **`moe_multi_gpu.py`** - Experts on different GPUs (same machine)
2. **`moe_multi_machine.py`** - Experts on different machines (distributed)
3. **`DISTRIBUTED_EXPLANATION.md`** - Detailed technical explanation
4. **`test_multi_gpu.sh`** - Test script for multi-GPU setup

---

## Quick Start

### Option 1: Multi-GPU (Same Machine)

**Requirements:**
- 2+ NVIDIA GPUs
- PyTorch with CUDA support

**Run:**
```bash
# Simple test
python moe_multi_gpu.py

# Or use the test script
./test_multi_gpu.sh
```

**Expected Output:**
```
================================================================================
Multi-GPU MoE Inference Example
================================================================================

Available GPUs: 2
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB

================================================================================
Creating MoE Model with Distributed Experts
================================================================================
Distributing 5 experts across 2 GPUs
  Expert 0 → cuda:0
  Expert 1 → cuda:1
  Expert 2 → cuda:0
  Expert 3 → cuda:1
  Expert 4 → cuda:0
...
Inference Complete!
```

### Option 2: Multi-Machine (Distributed)

**Requirements:**
- Multiple machines with network connectivity
- Same Python environment on all machines
- Open port for communication (default: 29500)

**Setup:**

1. **On Master Machine** (192.168.1.100):
```bash
python moe_multi_machine.py \
    --rank 0 \
    --world-size 3 \
    --master-addr 192.168.1.100 \
    --master-port 29500
```

2. **On Worker Machine 1** (192.168.1.101):
```bash
python moe_multi_machine.py \
    --rank 1 \
    --world-size 3 \
    --master-addr 192.168.1.100 \
    --master-port 29500
```

3. **On Worker Machine 2** (192.168.1.102):
```bash
python moe_multi_machine.py \
    --rank 2 \
    --world-size 3 \
    --master-addr 192.168.1.100 \
    --master-port 29500
```

**Expected Output (Master):**
```
================================================================================
Starting Distributed MoE - Rank 0/3
================================================================================
Initializing master node: worker0
Expert-to-worker mapping: {0: 'worker1', 1: 'worker1', 2: 'worker1', 3: 'worker2', 4: 'worker2'}
Master: Created 5 remote experts

Input shape: torch.Size([2, 10, 512])
Running distributed inference...

Output shape: torch.Size([2, 10, 512])
Distributed inference complete!
```

**Expected Output (Workers):**
```
================================================================================
Starting Distributed MoE - Rank 1/3
================================================================================
Initializing worker node: worker1 with experts [0, 1, 2]
Worker 1: Initialized experts [0, 1, 2] on cuda
Worker 1 ready and waiting for requests...
```

---

## Adapting to Your Use Case

### 1. Change Model Dimensions

Edit the model creation:

```python
# In moe_multi_gpu.py or moe_multi_machine.py

# Change from:
model = SimpleMoEModel(hidden_dim=512, num_layers=2)

# To your dimensions:
model = SimpleMoEModel(hidden_dim=2048, num_layers=27)
```

### 2. Change Number of Experts

Modify the MoE layer:

```python
moe_layer = MultiGPUMoELayer(
    hidden_dim=2048,
    ffn_dim=8192,
    num_experts=16,  # Change this
    gpu_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
)
```

### 3. Custom Expert Assignment (Multi-Machine)

Edit `expert_assignments` in `moe_multi_machine.py`:

```python
# Distribute 16 experts across 4 workers
expert_assignments = {
    1: [0, 1, 2, 3],      # Worker 1: experts 0-3
    2: [4, 5, 6, 7],      # Worker 2: experts 4-7
    3: [8, 9, 10, 11],    # Worker 3: experts 8-11
    4: [12, 13, 14, 15],  # Worker 4: experts 12-15
}
```

### 4. Load Real Model Weights

Replace the random initialization:

```python
# Load checkpoint
checkpoint = torch.load('deepseek_model.pt')

# Load into experts
for expert_id, expert in enumerate(experts):
    expert_state_dict = checkpoint[f'expert_{expert_id}']
    expert.load_state_dict(expert_state_dict)
```

---

## Integration with DeepSeek Model

To integrate with your actual DeepSeek model:

### Step 1: Extract Expert Weights

```python
# From your loaded DeepSeek model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/deepseek-moe-16b-base",
    trust_remote_code=True
)

# Extract expert weights from a specific layer
layer_idx = 0
moe_layer = model.model.layers[layer_idx].mlp

# moe_layer contains the experts and router
# Save them separately
torch.save({
    'router': moe_layer.gate.state_dict(),
    'experts': [expert.state_dict() for expert in moe_layer.experts]
}, f'layer_{layer_idx}_moe.pt')
```

### Step 2: Load into Distributed Model

```python
# Load saved weights
checkpoint = torch.load(f'layer_{layer_idx}_moe.pt')

# Initialize distributed MoE layer
dist_moe = MultiGPUMoELayer(...)

# Load router
dist_moe.gate.load_state_dict(checkpoint['router'])

# Load experts
for expert_id, expert_state in enumerate(checkpoint['experts']):
    dist_moe.experts[expert_id].load_state_dict(expert_state)
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Use fewer/smaller experts per GPU
```python
# Reduce ffn_dim
moe_layer = MultiGPUMoELayer(ffn_dim=4096)  # Instead of 8192
```

**Solution 2**: Use CPU offloading for some experts
```python
gpu_devices = ['cuda:0', 'cuda:1', 'cpu']  # Some experts on CPU
```

### Issue: "Connection refused" (Multi-Machine)

**Check:**
1. All machines can ping each other
2. Firewall allows port 29500
3. Same `--master-addr` on all machines
4. Same `--world-size` on all machines

**Test connectivity:**
```bash
# On worker machine
telnet <master-addr> 29500
```

### Issue: "Timeout" in RPC calls

**Solution**: Increase timeout
```python
rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
    rpc_timeout=600  # 10 minutes instead of 5
)
```

### Issue: Slow inference

**Check:**
1. Are experts truly running in parallel?
2. Use NVIDIA NVLink if available:
   ```bash
   nvidia-smi nvlink --status
   ```
3. Profile with:
   ```python
   with torch.profiler.profile() as prof:
       output = model(input)
   print(prof.key_averages().table())
   ```

---

## Performance Tips

### 1. Use Mixed Precision
```python
with torch.cuda.amp.autocast():
    output = model(input)
```

### 2. Batch Inference
```python
# Process multiple requests together
batch_size = 32  # Instead of 1
inputs = torch.cat([input1, input2, ...])
outputs = model(inputs)
```

### 3. Pin Memory (Multi-Machine)
```python
# Speed up CPU-GPU transfers
input = input.pin_memory()
output = expert(input.cuda())
```

### 4. Use InfiniBand (if available)
```bash
# Check InfiniBand
ibstat

# Set environment variable
export NCCL_IB_DISABLE=0
```

---

## Benchmarking

Compare different strategies:

```python
import time

def benchmark(model, input, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = model(input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        _ = model(input)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    
    return avg_time

# Run benchmark
benchmark(model, sample_input)
```

---

## Next Steps

1. **Read** `DISTRIBUTED_EXPLANATION.md` for technical details
2. **Test** with `test_multi_gpu.sh`
3. **Modify** for your specific model
4. **Benchmark** to find optimal configuration
5. **Scale** to production workload

## Questions?

Check the detailed explanation in `DISTRIBUTED_EXPLANATION.md` or refer to PyTorch documentation:
- [Distributed RPC](https://pytorch.org/docs/stable/rpc.html)
- [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [FSDP](https://pytorch.org/docs/stable/fsdp.html)






