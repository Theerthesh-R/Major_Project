import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("✅ GPU is available")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("❌ GPU not available — running on CPU only")

# Quick tensor test on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(3, 3).to(device)
print(f"Tensor allocated on: {device}")
print(x)
