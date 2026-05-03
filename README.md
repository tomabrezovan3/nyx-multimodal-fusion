# 🛰️ Multimodal Sensor Fusion (IR + Radar)

This project explores multimodal fusion of infrared (IR) and radar data for target detection in low-visibility environments.

## Results

- IR-only: 0.625
- Radar-only: 0.700
- Late fusion: 0.700
- Intermediate fusion: 0.675
- Gated fusion: 0.750

## Key Idea

Fusion improves performance only when modalities are complementary.

## Features

- Synthetic IR dataset generator
- Synthetic radar (Range-Doppler) generator
- Single-modality classifiers
- Multiple fusion strategies:
  - Late fusion
  - Intermediate fusion
  - Gated fusion
- Complementarity analysis tool

## Tech Stack

- Python
- PyTorch
- NumPy
- SciPy