# PULSE: Personalized Opportunistic Learning for Wearable Devices

> **Paper:** PULSE: Personalized Opportunistic Learning for Wearable Devices  
> **Authors:** Anonymous authors  
> **Status:** Under review for IEEE PerCom 2027

This repository contains the anonymized artifact for **PULSE**, a personalized opportunistic learning method for wearable devices. The artifact supports the reproducible evaluation of learning over intermittent device encounters and the measurement of the method on the SenseWear embedded platform.

The repository is organized into the following main components:

- [`simulations/`](simulations/): software experiments and analysis for the paper.
- [`Implementation/`](Implementation/): embedded implementation and hardware measurements on the SenseWear board.
- [`data/`](data/): experiment inputs and pretrained artifacts, including encoder checkpoints with training logs and summaries, as well as mobility traces and their generation summaries.

## Repository organization

### `data/`

This directory stores shared inputs and artifacts used by the experiments:

- `checkpoints/`: pretrained encoder weights for the supported source datasets, together with architecture metadata, training logs, and training summaries.
- `mobility/`: encounter traces and accompanying JSON summaries, including Cambridge traces and seeded synthetic scenarios generated with edge-Markov and community-based models.

### `simulations/`

This directory contains the end-to-end simulation and data-analysis workflow:

- [`opportunistic_simulator/`](simulations/opportunistic_simulator/): event-driven simulator for PULSE and the comparison methods, including local-only learning, OppAVG, Opportunistic Federated Learning, and OPTIMIST. It models user nodes, intermittent encounters, local training, peer selection, evaluation, budgets, and experiment logging.
- [`scripts/`](simulations/scripts/): data preparation and experiment utilities, including HAR dataset preprocessing, encoder pretraining, mobility-trace generation, and experiment setup.
- [`opportunistic_simulator/configs/`](simulations/opportunistic_simulator/configs/): YAML configurations defining datasets, models, algorithms, mobility, training, and output settings.
- [`results/`](simulations/results/): generated experiment outputs and aggregated result tables.
- [`plots/`](simulations/plots/): plots and summary tables used to inspect mobility characteristics and experimental results.
- [`results.ipynb`](simulations/results.ipynb): notebook for summarizing and visualizing simulation results.
- `generate_experiment_configs.sh` and `run_generated_experiments.sh`: helpers for generating experiment configurations and running experiment batches.

The simulation workflow separates the source dataset used to pretrain the shared encoder from the target data partitioned among simulated users. Mobility traces provide encounter events in the form `time,node1,node2`; they can be generated synthetically or prepared from available traces. Each run records configuration, metrics, training activity, contact outcomes, node assignments, and a seed-specific summary.

For detailed simulator requirements and commands, see [`simulations/opportunistic_simulator/README.md`](simulations/opportunistic_simulator/README.md). For dataset preparation and encoder pretraining, see [`simulations/scripts/README.md`](simulations/scripts/README.md).

### `Implementation/`

This directory contains the Zephyr-based embedded implementation for the SenseWear wearable platform:

- board support, device drivers, sensor and power-management components;
- the application and Bluetooth interfaces used by the device;
- configuration through CMake, Kconfig, and devicetree;
- on-target tests and hardware bring-up utilities;
- build and deployment configuration for the nRF54L15 platform;
- measurement workflows and recorded results for the paper, including energy consumption, memory footprint, and execution time.

The implementation is intended to verify the feasibility and resource cost of the learning workflow on the target hardware. It is complementary to the simulator: the simulator evaluates learning behavior at scale and across encounter patterns, while the embedded implementation measures device-level resource usage.

Start with [`Implementation/README.md`](Implementation/README.md) for the firmware architecture. Build and setup instructions are available in [`Implementation/SETUP.md`](Implementation/SETUP.md), and the source-tree organization is documented in [`Implementation/ORGANIZATION.md`](Implementation/ORGANIZATION.md).

## Reproducibility

The experiments are configured through versioned scripts and YAML files. Results should be associated with the configuration and random seed used to produce them. Dataset files, pretrained encoder checkpoints, mobility traces, and generated summaries are kept in the repository's data and results directories where applicable.

The repository is intentionally anonymized for review. It does not include author names, affiliations, acknowledgements, or other identifying information.

## License

See [`Implementation/LICENSE`](Implementation/LICENSE) for the license distributed with the implementation.
