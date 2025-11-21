# Dynamic Resource Allocation for LEO Satellite Quantum Networks using Deep Reinforcement Learning

This repository contains the official source code for the research paper:

**"Dynamic Resource Allocation for LEO Satellite Quantum Networks using Deep Reinforcement Learning"**

**Author:**
*   Phuc Hao Do ([ORCID: 0000-0003-0645-0021](https://orcid.org/0000-0003-0645-0021))
    *   Danang Architecture University, Da Nang, Vietnam
    *   The Bonch-Bruevich Saint Petersburg State University of Telecommunications, Saint Petersburg, Russia

**Status:** Submitted to *Computer Networks* (Elsevier).

---

## 1. Overview

Satellite Quantum Communication (SQC) is an emerging network architecture for future global secure communications. However, the performance of Low Earth Orbit (LEO) SQC links is highly volatile due to dynamic channel conditions from orbital mechanics and atmospheric turbulence. This makes the static allocation of network resources (such as transmission power and rate) sub-optimal.

This paper addresses this challenge by framing the dynamic control of link parameters as a **resource allocation problem** and solving it using **Deep Reinforcement Learning (RL)**. We propose a framework where an RL agent learns an adaptive policy to maximize the total Secret Key Rate (SKR) generated during a satellite pass.

This repository provides the complete simulation environment and the RL agent implementation used to obtain the results presented in our paper, facilitating the reproducibility of our work and enabling further research in this domain.

## 2. Key Results

Our experiments demonstrate that the trained Reinforcement Learning agent (using the Proximal Policy Optimization algorithm) successfully learns a sophisticated, state-dependent policy that outperforms static baseline strategies.

### 2.1. Performance Comparison

The RL agent was benchmarked against three baseline policies over 50 independent, randomly generated episodes. The results validate the superiority of the adaptive approach.

**Summary of Results:**
| Policy          | Mean Total SKR (Mbps per pass) | Standard Deviation |
|-----------------|:------------------------------:|:------------------:|
| **PPO Agent**   | **44,918.56**                  | 22,177.91          |
| Low-Power (LP)  | 42,472.04                      | 24,176.31          |
| High-Power (HP) | 4,565.52                       | 2,736.24           |
| Random          | 25,388.92                      | 12,527.33          |

![Policy Performance Comparison](figures/policy_comparison_bar.png)
*Figure 1: Comparison of the average total SKR achieved per pass by the PPO agent and three baseline policies. The PPO agent achieves the highest mean performance.*

### 2.2. Analysis of the Learned Adaptive Policy

The PPO agent's performance gain comes from its ability to learn a non-trivial, adaptive strategy. It intelligently balances the trade-off between maximizing transmission rate and ensuring link robustness.

![PPO Agent Behavior](figures/ppo_agent_behavior.png)
*Figure 2: Analysis of the PPO agent's behavior during a representative pass. The agent primarily uses the high-throughput mode (Action 0) but strategically switches to the high-robustness mode (Action 2) during severe channel degradation (low transmittance) to preserve the link.*

---

## 3. Project Structure

```
.
├── figures/                # Contains generated plots for the paper
│   ├── policy_comparison_bar.png
│   └── ppo_agent_behavior.png
├── src/                    # Contains all source code
│   ├── environment.py      # Defines the custom Gymnasium SQC Link environment
│   ├── train.py            # Script to train the PPO agent
│   └── evaluate.py         # Script to evaluate the agent and generate figures
├── models/                 # Stores the trained model files (created by train.py)
├── logs/                   # Stores TensorBoard training logs
├── environment.yml         # Conda environment definition for reproducibility
└── README.md               # This file
```

## 4. How to Reproduce the Results

### 4.1. Prerequisites
*   An NVIDIA GPU is recommended for faster training.
*   [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### 4.2. Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/ai-sqc-survey.git
    cd ai-sqc-survey
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file specifies all necessary packages.
    ```bash
    conda env create -f environment.yml
    conda activate sqc_rl
    ```

### 4.3. Running the Scripts
All scripts should be run from the root directory of the project.

1.  **Train the model:**
    This script will train the PPO agent for 200,000 timesteps and save the best model in the `models/` directory.
    ```bash
    python src/train.py
    ```
    You can monitor the training progress using TensorBoard:
    ```bash
    tensorboard --logdir logs/
    ```

2.  **Evaluate the model and generate figures:**
    This script loads the best-trained model from `models/best_model.zip`, runs the evaluation against baseline policies, and saves the result plots in the `figures/` directory.
    ```bash
    python src/evaluate.py
    ```
