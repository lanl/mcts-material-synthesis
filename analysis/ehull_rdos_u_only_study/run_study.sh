#!/bin/bash
# Reproduce the U-only ehull_rdos study: sharp tanh E_hull + DOSCAR reward
# Reward = beta*ehull_reward(E_hull) + gamma*r_DOS, where ehull_reward = -tanh(300*(E_hull-0.05))
# beta/gamma default to 1.0/0.0001 (see config.json/config.example.json - gamma is a single
# value shared by the MCTS run and all analysis/plotting scripts)
# F-block mode: u_only (108 composition design space)
#
# Starting material: Pb6U1W6 (centrally located between target compounds)
#
# Requires (see README.md "Data Availability" for schema/how to obtain):
#   - high_throughput_mace_results.full.csv (repo root)
#   - doscar_rewards.csv (repo root)
#   - a Materials Project API key, supplied via config.json (preferred, gitignored)
#     or the MP_API_KEY environment variable below

set -e

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../../" && pwd)"
OUTPUT_DIR="${STUDY_DIR}"
ITERATIONS=150
TRANSITION_METAL="W"
GROUP_IV="Pb"
F_BLOCK_MODE="u_only"

echo "=================================================="
echo "EHULL + RDOS STUDY (U-only)"
echo "Reward = beta*ehull_reward(E_hull) + gamma*r_DOS"
echo "where ehull_reward = -tanh(300 * (E_hull - 0.05))"
echo "=================================================="

MP_API_KEY_ARG=()
if [ -n "${MP_API_KEY}" ]; then
    MP_API_KEY_ARG=(--mp-api-key "${MP_API_KEY}")
elif [ ! -f "${REPO_ROOT}/config.json" ]; then
    echo "Note: no config.json found and MP_API_KEY env var not set."
    echo "Copy config.example.json to config.json and add your Materials Project API key,"
    echo "or export MP_API_KEY=your_key before running this script."
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
python run_mcts.py \
    --structure examples/mat_Pb6U1W6_sg191.cif \
    --transition-metal ${TRANSITION_METAL} \
    --group-iv ${GROUP_IV} \
    --f-block-mode ${F_BLOCK_MODE} \
    --iterations ${ITERATIONS} \
    --exploration-constant 0.1 \
    --epsilon 0.1 \
    --termination-limit 25 \
    --rollout-method ehull_rdos \
    --seed 42 \
    "${MP_API_KEY_ARG[@]}" \
    --rollout-depth 3 \
    --n-rollout 1 \
    --output "${OUTPUT_DIR}"

echo ""
echo "=================================================="
echo "MCTS RUN COMPLETE - Generating figures..."
echo "=================================================="
cd "${STUDY_DIR}"
bash generate_plots.sh
