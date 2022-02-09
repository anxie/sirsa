## Getting Started

Create and activate conda environment, install softlearning to enable command line interface.
```
cd ${SIRSA_PATH}
conda env create -f environment.yml
conda activate sirsa
pip install -e ${SIRSA_PATH}
```

## Examples
### Training and simulating an agent
1. To train the agent
```
python -m examples.development.main \
--algorithm SAC \
--universe gym \
--domain Reacher2D \
--task Default-v0 \
--exp-name exp-name \
--checkpoint-frequency 500 \
--observe-context true \
--env-type random \
--predict-context true \
--history-length 5 \
--latent-dim 2 \
--ensemble-size 4 \
--cvar-alpha 0.25 \
--threshold-iterations 25000 \
--num-tasks 20 \
--local-dir "~/ray_results" \
--trial-name-template "trial=0"
```

2. To simulate the resulting policy:
First, find the path that the checkpoint is saved to. The next command assumes that this path is found from `${CHECKPOINT_DIR}` environment variable.

```
python -m examples.development.simulate_policy \
    ${CHECKPOINT_DIR} \
    --max-path-length 500 \
    --num-rollouts 1 \
    --deterministic true \
    --task-id 0 1 \
    --env-type uniform
```
