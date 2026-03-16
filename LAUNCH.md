# Parkour RL: Deployment & Execution Guide

This guide provides complete instructions for setting up the environment via Docker and running reinforcement learning experiments (Value Iteration, Policy Iteration) and comparing them with baselines in the "Parkour" environment.

---

## Prerequisites

Before starting, ensure you have the following installed:
- **Docker**
- **Git**

```bash
git clone git@github.com:GibaDuliya/Parkour.git
```

---
## Step 1: Docker Environment Setup

### 1.1 Access Configuration
The `credentials` file maps your local user ID to the container to prevent file permission issues. Verify its content:
```bash
cat credentials
```

### 1.2 Build the Image
Build the lightweight Docker image (this takes 1-2 minutes):
```bash
chmod +x build.sh
./build.sh
```

### 1.3 Launch the Container
Start the container. The script mounts the current directory to `/app` inside the container:
```bash
chmod +x launch_container.sh
./launch_container.sh
```
*Once launched, you will be automatically redirected to the container bash terminal.*

---

## Step 2: Landscape Generation

Before training agents, you must generate maps (landscapes). The script generates only traversable maps where the target is reachable.

Inside the container terminal, execute:
```bash
# Generate 10 valid landscapes (100x100)
chmod +x ./scripts/generate_landscapes.sh  
./scripts/generate_landscapes.sh 10
```
**Result:** Directories `landscape_1`, `landscape_2`, etc., will appear in the `landscape/` folder, containing height maps (`height_map.npy`) and spawn points (`eval_cells.npy`).

---

## Step 3: Training RL Agents

Now, train the Dynamic Programming algorithms on all generated maps. Configuration files are located in `/configs`.

### Option A: Batch Training (Recommended)
Train the chosen algorithm on all generated landscapes:
```bash
chmod +x ./scripts/train_all_landscapes.sh

# Train Value Iteration
./scripts/train_all_landscapes.sh value_iteration

# Train Policy Iteration
./scripts/train_all_landscapes.sh policy_iteration
```

### Option B: Manual Single-Map Training
```bash
python run/train.py value_iteration --landscape_id 1
```
**Result:** Trained policies and convergence plots are saved in `agents/{algorithm_name}/{timestamp}/`.

---

##  Step 4: Method Comparison and Table Generation

This final stage compares the trained models against heuristic baselines (**Random, Safest Path, Shortest Path, Budget Greedy**) on the same maps and start points.

Execute:
```bash
python run/compare.py
```

**What you will get:**
1.  **Console table** showing Success Rate, Avg Steps, and Avg Reward for each method.
2.  **Log file** in the project root: `compare_results_YYYY-MM-DD.txt`.

*Note: The `hp_start` parameter in `configs/env.yaml` defines the health points for all agents in this test.*

---

##  Step 5: Additional Experiments and Jupyter

When running `./launch_container.sh`, a Jupyter server automatically starts on port **8890**.
1. Open your browser: `http://localhost:8890`
2. Navigate to the `experiments/` directory.
3. You will find notebooks for interactive map visualization, trajectory playback, and detailed Value Function analysis.

---

## Project Structure (Summary)

-   `src/environment/`: ParkourEnv logic (HP, falls, and rewards).
-   `src/algorithms/`: Implementations of VI, PI, and `baselines.py`.
-   `landscape/`: Generated worlds and evaluation points.
-   `agents/`: Storage for trained models and metrics.
-   `run/`: Scripts for training, comparison, and visualization.
-   `configs/`: YAML settings for the environment and algorithms.

---
**Happy Parkouring!**