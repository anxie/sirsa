python -m examples.development.main \
--algorithm SAC \
--universe gym \
--domain Minitaur \
--task Default-v0 \
--exp-name exp-name \
--checkpoint-frequency 500 \
--observe-context true \
--env-type random \
--predict-context true \
--history-length 5 \
--latent-dim 6 \
--state-dim 12 \
--ensemble-size 4 \
--cvar-alpha 0.25 \
--threshold-iterations 150000 \
--num-qs 8 \
--num-tasks 80 \
--local-dir "~/ray_results" \
--trial-name-template 'trial=0'