import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from collections import defaultdict

# Storage for routing information
routing_info = defaultdict(list)

def capture_routing_hook(module, input, output):
    """Hook to capture expert routing decisions"""
    # Check if output is a tuple (common for custom modules)
    if isinstance(output, tuple):
        hidden_states = output[0]
        # Some MoE implementations return (hidden_states, router_logits, expert_indices)
        if len(output) > 1:
            for item in output[1:]:
                if isinstance(item, torch.Tensor) and item.dtype in [torch.long, torch.int32, torch.int64]:
                    # These are likely expert indices
                    routing_info[f"{module.__class__.__name__}_indices"].append(item.cpu().tolist())
                elif isinstance(item, torch.Tensor) and item.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                    # These are likely router weights/logits
                    # Get top-k experts
                    k = min(2, item.shape[-1]) if len(item.shape) > 0 else 1
                    top_experts = torch.topk(item, k=k, dim=-1)
                    routing_info[f"{module.__class__.__name__}_routing"].append({
                        'expert_indices': top_experts.indices.cpu().tolist(),
                        'expert_weights': top_experts.values.cpu().tolist(),
                    })
    
    # Also check if output has router_logits attribute
    if hasattr(output, 'router_logits') and output.router_logits is not None:
        router_logits = output.router_logits
        top_experts = torch.topk(router_logits, k=min(2, router_logits.shape[-1]), dim=-1)
        routing_info[f"{module.__class__.__name__}_attr"].append({
            'expert_indices': top_experts.indices.cpu().tolist(),
            'expert_weights': top_experts.values.cpu().tolist(),
        })
    
    return output

model_name = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# Register hooks on MoE layers to capture routing
hooked_layers = []
for name, module in model.named_modules():
    if 'mlp' in name.lower() or 'moe' in name.lower() or 'expert' in name.lower():
        module.register_forward_hook(capture_routing_hook)
        hooked_layers.append(name)

print(f"Registered hooks on {len(hooked_layers)} layers")
if hooked_layers:
    print(f"Sample hooked layers: {hooked_layers[:3]}")
    # 27 (NUMBER OF MOE LAYERS) * 65 (64 experts + 1 shared)  * 5 (5 EXPERT, ACT, UP, DOWN, GATE) + 27 * 3 (MLP.EXPERT + MLP + MLP>=.GATE) + 5 (LAYER0) = 8861

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")

# First, do a simple forward pass to inspect output structure
print("\nRunning single forward pass to inspect outputs...")
with torch.no_grad():
    forward_output = model(**inputs.to(model.device))
    print(f"Forward output type: {type(forward_output)}")
    if hasattr(forward_output, '__dict__'):
        print(f"Forward output attributes: {list(forward_output.__dict__.keys())}")

print("\nRunning generation...")
routing_info.clear()  # Clear any routing info from the forward pass
with torch.no_grad():
    outputs = model.generate(**inputs.to(model.device), max_new_tokens=100, return_dict_in_generate=True)

result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print("\n" + "="*80)
print("GENERATED TEXT:")
print("="*80)
print(result)

# Print routing information
if routing_info:
    print("\n" + "="*80)
    print("EXPERT ROUTING INFORMATION:")
    print("="*80)
    for layer_name, routes in routing_info.items():
        print(f"\n{layer_name}:")
        print(f"  Total routing decisions: {len(routes)}")
        if routes:
            # Show first few routing decisions as example
            for i, route in enumerate(routes[:3]):
                print(f"  Token {i}: Expert indices: {route['expert_indices']}, Weights: {route['expert_weights']}")
else:
    print("\nNote: No routing information captured. Checking model architecture...")
    # Alternative: directly inspect model output
    with torch.no_grad():
        direct_output = model(**inputs.to(model.device), output_router_logits=True)
        if hasattr(direct_output, 'router_logits') and direct_output.router_logits is not None:
            print("\nRouter logits found in model output!")
            for layer_idx, router_logit in enumerate(direct_output.router_logits):
                top_experts = torch.topk(router_logit, k=2, dim=-1)
                print(f"\nLayer {layer_idx}:")
                print(f"  Shape: {router_logit.shape}")
                print(f"  Top-2 experts for first token: {top_experts.indices[0, 0].cpu().tolist()}")
                print(f"  Expert weights: {top_experts.values[0, 0].cpu().tolist()}")
