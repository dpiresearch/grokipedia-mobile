#!/bin/bash
# Test script for multi-GPU MoE inference

echo "========================================="
echo "Testing Multi-GPU MoE Inference"
echo "========================================="

# Check CUDA availability
python3 << EOF
import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo ""
echo "Running multi-GPU MoE example..."
python3 moe_multi_gpu.py

echo ""
echo "Test complete!"

