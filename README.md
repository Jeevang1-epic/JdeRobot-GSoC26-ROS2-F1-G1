# Deep-F1: End-to-End Behavioral Cloning for Autonomous Racing
**GSoC 2026 Phase II Technical Report | JdeRobot Visual Control Module**

## Project Overview
This repository contains the foundational research and implementation of an end-to-end visual servoing pipeline for high-speed autonomous racing. The project utilizes monocular vision and deep neural networks to perform direct regression of vehicle dynamics from raw pixel data.

## Theoretical Framework
The navigation problem is formulated as a mapping $\pi: I \rightarrow \mathcal{U}$, where $I$ represents the monocular camera input space and $\mathcal{U}$ represents the continuous control space defined by the linear velocity $v$ and angular velocity $\omega$.

### Mathematical Optimization
The policy is optimized by minimizing the Mean Squared Error (MSE) objective function:
$$L = \frac{1}{N} \sum_{i=1}^{N} || \mathbf{u}_i - \Phi(I_i; \theta) ||^2$$

Where $\Phi$ is the neural network parameterized by $\theta$.

## Repository Structure
- **src/visual_control_module/**: Core logic including inference nodes and network architecture.
- **dataset/**: Primary training samples containing expert driving trajectories.
- **dataset_brittle/**: Edge-case dataset for track-limit recovery training.
- **models/**: Pre-trained weights and performance telemetry.
## 📁 Repository Structure


    jderobot_gsoc26/
    ├── dataset/                        # Primary expert driving dataset (Images + CSV)
    ├── dataset_brittle/                # Edge-case recovery dataset
    ├── src/
    │   ├── visual_control_module/      # Main ROS 2 Intelligence Package
    │   │   ├── config/                 # Network & PID hyperparameters
    │   │   │   └── hyperparams.yaml
    │   │   ├── launch/                 # ROS 2 Launch files for Gazebo integration
    │   │   │   └── visual_control.launch.py
    │   │   ├── models/                 # ❗ TRAINING OUTPUTS & WEIGHTS
    │   │   │   ├── robust_model.pth    # Optimized PyTorch weights (177MB)
    │   │   │   ├── model_brittle.pth   # Recovery-trained weights
    │   │   │   ├── training_loss.png   # Convergence visualization
    │   │   │   └── model_comparison.png 
    │   │   ├── resource/               # ROS 2 index files
    │   │   ├── vision_core/            # ❗ PRIMARY SCRIPT ENGINE
    │   │   │   ├── __init__.py
    │   │   │   ├── data_recorder.py    # Telemetry acquisition script
    │   │   │   ├── network.py          # CNN Architecture definition (PyTorch)
    │   │   │   ├── train.py            # Model training pipeline
    │   │   │   ├── visual_node.py      # Core Inference Node (Standard)
    │   │   │   ├── visual_node_v2.py   # Advanced Inference Node (Optimized)
    │   │   │   └── visual_test_cam.py  # Diagnostic camera utility
    │   │   ├── package.xml             # ROS 2 dependencies
    │   │   ├── setup.cfg               # Package configuration
    │   │   └── setup.py                # Installation script
    │   └── CustomRobots/               # JdeRobot 3D environments & assets
    ├── .gitattributes                  # LFS configuration for .pth files
    ├── .gitignore                      # Workspace exclusion rules
    └── README.md                       # Technical Documentation

## Methodology

### 1. Perception and Data Pipeline
The perception layer utilizes a ROS 2 (Humble) subscriber model to intercept camera feeds.
- **Normalization**: Images are standardized to zero-mean and unit variance.
- **Data Augmentation**: To prevent overfitting to specific track features, random brightness shifts and geometric perturbations were applied.

### 2. Network Architecture
The architecture employs a deep convolutional backbone followed by a regression-head for real-time control.
- **Input Dimensions**: 640 x 480 RGB
- **Backbone**: ResNet-derived feature extractor.
- **Actuation**: Outputs a `geometry_msgs/Twist` message for real-time physics interaction.

## Results and Evaluation

### Training Convergence
The model demonstrates high-fidelity convergence over 100 epochs, with validation loss tracking the training distribution accurately.

![Training Loss Convergence](src/visual_control_module/models/training_loss_brittle.png)

### Performance Benchmarking
Comparative analysis between model variants (standard vs. robust) shows significant improvement in cornering stability with the current iteration.

![Model Performance Benchmark](src/visual_control_module/models/model_comparison_bar.png)

### Inference Robustness Analysis
The following scatter plot correlates the network's predicted steering response against the ground-truth labels provided by the expert driver.

![Inference Robustness Scatter](src/visual_control_module/models/robust_inference_scatter.png)

## Execution Instructions
Ensure the ROS 2 workspace environment variables are properly initialized.

1. Source the environment:
   `source /opt/ros/humble/setup.bash`
2. Run the visual controller:
   `python3 src/visual_control_module/vision_core/visual_node.py`

---
