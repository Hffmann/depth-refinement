# Depth Refinement with GAIL-CNN

Hybrid model combining ZeroDepth monocular estimation with GAIL-based refinement.
- Pre-train a CNN for initial corrections, then fine-tune with RL for context-aware refinements.
- Use a CNN as the policy network in GAIL to combine adversarial training with efficient inference.

## Features
- Sliding window processing
- LiDAR-free refinement
- Real-time visualization
- Multi-domain training

## Installation
```bash
git clone https://github.com/Hffmann/depth-refinement.git
cd depth-refinement
pip install -r requirements.txt
