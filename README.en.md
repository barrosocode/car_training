[🇧🇷 Português](README.md) | **🇺🇸 English**
***

# Path Learning
*Deep Reinforcement Learning in Autonomous Vehicles*

[![Status](https://github.com/barrosocode/car_training/actions/workflows/blank.yml/badge.svg)](https://github.com/barrosocode/car_training/actions/workflows/blank.yml) [![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/barrosocode/car_training/blob/main/LICENSE) ![Language](https://img.shields.io/github/languages/top/barrosocode/car_training)

## Abstract

This project consists of developing an autonomous driving agent using Deep Reinforcement Learning. The agent was trained to navigate the track in the *CarRacing* environment from start to finish, learning optimal control policies through trial and error.

**Key technologies and methods used:**

* **Environment:** Built and managed with the **`Gymnasium (gym)`** library.
* **Algorithm:** Proximal Policy Optimization (**PPO**) from **`Stable Baselines3`**.
* **Training Optimization:** Vectorized environments with `make_vec_env` for parallelism and **`VecFrameStack`** to provide the agent with visual memory.
* **Preprocessing:** Custom wrappers (`gym.Wrapper`) and array manipulation with **`NumPy`**.
* **Model Persistence:** Saving checkpoints during training with **`CheckpointCallback`**.

## Introduction
**Project for Unit III - Course: Computer Vision 2025.1**

Autonomous driving represents one of the most significant and transformative challenges in the fields of artificial intelligence and engineering. Developing vehicles capable of navigating complex environments safely and efficiently has the potential to revolutionize transportation systems, reduce accidents, and optimize traffic flow. Traditional approaches to this task often rely on a massive set of pre-programmed rules and expensive sensors, making them brittle and difficult to generalize to unexpected scenarios.

As a powerful alternative, **Deep Reinforcement Learning (DRL)** emerges as a promising methodology, inspired by how humans learn: through interaction and experience. Instead of following explicit instructions, a DRL agent learns to take the best actions through a system of rewards and penalties, optimizing its decision-making policy over time.

This project applies DRL concepts to train an agent to drive a car in the **`CarRacing-v2`** simulation environment from the `Gymnasium` library. This environment is particularly challenging as it requires the agent to learn to control the vehicle (continuous steering, acceleration, and braking actions) based solely on raw visual data (pixels), simulating what a real car's onboard camera would capture.

The specific objectives of this work are:
1.  To configure and customize the `CarRacing-v2` environment for training a DRL agent.
2.  To implement and train an agent using the **Proximal Policy Optimization (PPO)** algorithm, known for its robustness and efficiency in continuous control environments.
3.  To develop a final model capable of autonomously completing the track, demonstrating coherent driving behavior.
4.  To document the process, architecture, and results as a practical guide for applying `Stable Baselines3` to vision-based control problems.

## Methodology

This section details the system architecture for training and the structural organization of the source code, which were designed to ensure a clear and reproducible workflow.

### System Architecture

The project's architecture is centered on the standard interaction loop of a Reinforcement Learning agent, where the Agent and the Environment continuously exchange information. This process is orchestrated by the `Stable Baselines3` library, which abstracts much of the complexity of the training loop.

The data and process flow can be visualized in the diagram below:

````mermaid
graph TD;
    A["Gymnasium Environment (CarRacing-v2)"] -- Observation (Frame) --> B["Preprocessing (Wrappers)"];
    B -- Processed State --> C["PPO Agent (Neural Network)"];
    C -- "Actions (Steer, Accel., Brake)" --> A;
    C -- "Training Data" --> D["Training Loop (Stable Baselines3)"];
    D -- "Weight Updates" --> C;
    style A fill:#11111,stroke:#333,stroke-width:2px;
    style C fill:#11111,stroke:#333,stroke-width:2px;
````
# Result

# Installation

### 1. Prerequisites

Before installing the Python dependencies, ensure you have the following software installed:

* **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
* **Git**: [Download Git](https://git-scm.com/downloads/)

#### **Operating System Specific Prerequisites:**

<details>
<summary><strong>For Linux (Debian/Ubuntu-based)</strong></summary>

You will need the essential build tools and the SWIG library, which are dependencies for the `CarRacing` environment.

```bash
sudo apt-get update
sudo apt-get install -y build-essential swig
```
</details>

<details>
<summary><strong>For Windows</strong></summary>

Installation on Windows requires C++ build tools for the `Box2D` library.

1.  **Microsoft C++ Build Tools**:
    * Download the [Visual Studio Installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    * Run the installer, and in the "Workloads" tab, select the **"Desktop development with C++"** option.
    * Proceed with the installation.

2.  **SWIG**:
    * Download **SWIG for Windows** (look for `swigwin`) from the [official website](http://swig.org/download.html).
    * Unzip the file (e.g., to `C:\swigwin`).
    * Add the unzipped folder to your system's **PATH** so that `pip` can find it.
        * Search for "Edit the system environment variables" in Windows.
        * Click on "Environment Variables...".
        * In the "System variables" section, select the `Path` variable and click "Edit".
        * Click "New" and add the path to the SWIG folder (e.g., `C:\swigwin`).
</details>
<br>

---

### 2. Project Installation

Follow the steps below in your terminal or PowerShell.

**a. Clone the repository:**
```bash
git clone [https://github.com/barrosocode/car_training.git](https://github.com/barrosocode/car_training.git)
cd car_training
```

**b. Create and activate a virtual environment (Recommended):**
```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Linux or Windows (Git Bash):
source venv/bin/activate
```

**c. Install Python libraries:**

The original `requirements.txt` file contains GPU-specific packages for Linux. **Do not use it directly**. Follow the option that corresponds to your hardware.

<details>
<summary><strong>Option A: Installation with NVIDIA GPU (Recommended for Training)</strong></summary>

This option uses the acceleration of your NVIDIA graphics card for much faster training.

1.  **Install PyTorch with CUDA support:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the exact installation command for your CUDA version. For CUDA 12.1, the command is usually:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

2.  **Install other dependencies:**
    ```bash
    pip install stable_baselines3[extra] gymnasium[box2d] tensorboard
    ```
    * `stable_baselines3[extra]` installs the library with its common dependencies.
    * `gymnasium[box2d]` ensures the correct installation of the `CarRacing` environment.

</details>

<details>
<summary><strong>Option B: CPU-Only Installation</strong></summary>

Use this option if you do not have an NVIDIA graphics card or do not wish to set up CUDA. Training will be significantly slower.

1.  **Install the CPU version of PyTorch:**
    ```bash
    pip install torch torchvision torchaudio
    ```

2.  **Install other dependencies:**
    ```bash
    pip install stable_baselines3[extra] gymnasium[box2d] tensorboard
    ```
</details>
<br>

---

### 3. Execution

**a. To train a new model:**

Run the training script. Progress will be displayed in the terminal, and the models and logs will be saved in the `models/` and `logs/` folders.
```bash
python training.py
```
* The `training.py` script is configured to run for 2,400,000 timesteps and save a checkpoint every 100,000 steps.
* The model uses the "CnnPolicy" and various hyperparameters such as `learning_rate=0.0003`, `gamma=0.99`, and `batch_size=64`.

**b. To evaluate a pre-trained model:**

Run the test script. A Pygame window will appear showing the car being controlled by the agent.
```bash
python test.py
```
* **Important**: Before running, open the `test.py` file and ensure the `model_path` variable points to the model (`.zip` file) you want to test.
* The test environment is configured to render in `human` mode for visualization.

---

### Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Ag0ds"><img src="https://github.com/Ag0ds.png?size=100" alt="Foto de Gabriel Arthur"/><br/><sub><b>Gabriel Arthur</b></sub></a><br/><sub>Developer</sub>
    </td>
    <td align="center">
        <a href="https://github.com/barrosocode"><img src="https://github.com/barrosocode.png?size=100" alt="Foto de SGabriel Barroso"/><br/><sub><b>Gabriel Barroso</b></sub></a><br/><sub>Developer</sub>
    </td>
    <td align="center">
        <a href="https://github.com/heltonmaia"><img src="https://github.com/heltonmaia.png?size=100" alt="Foto de Helton Maia"/><br/><sub><b>Helton Maia</b></sub></a><br/><sub>Advisor</sub>
    </td>
  </tr>
</table>

---
