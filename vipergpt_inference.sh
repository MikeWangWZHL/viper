DATASET_NAME="None"
SUBSET_SIZE=100
SLEEP_RATE=15

export DATA_ROOT="/shared/nas2/wangz3/llava_data"
export OPENAI_API_KEY=$(cat /shared/nas2/wangz3/openai_school_account_api_key)
echo "API key: $OPENAI_API_KEY"
export CUDA_VISIBLE_DEVICES=7

# TASKS="svg_probing_lines svg_probing_angles shapeworld_relational-spatial_twoshapes nlvr maze_g2 maze_g3 geoclidean_2-shot_elements shapeworld_selection-superlative shapeworld_relational-spatial"
# TASKS="svg_probing_lines svg_probing_angles shapeworld_relational-spatial_twoshapes nlvr maze_g2 maze_g3 geoclidean_2-shot_elements shapeworld_selection-superlative shapeworld_relational-spatial"
TASKS="maze_g2 maze_g3 geoclidean_2-shot_elements shapeworld_selection-superlative shapeworld_relational-spatial"

# TASKS="shapeworld_relational-spatial_twoshapes maze_g3 geoclidean_2-shot_elements"
# TASKS="geoclidean_2-shot_elements"

# TASKS="nlvr"

PREFIX="base_layer_downstream"
INPUT_TYPE="image"
MODEL_TYPE="vipergpt-gpt4"
for TASK in $TASKS
do
    TASK_NAME="${PREFIX}__${TASK}__${INPUT_TYPE}"
    OUTPUT_DIR="/shared/nas2/wangz3/ecole-gvs-method/results/base_layer_downstream_tasks/march_3_batch/${TASK_NAME}__${MODEL_TYPE}"
    echo "Task name: ${TASK_NAME}"
    python vipergpt_inference.py \
        --subset_size ${SUBSET_SIZE} \
        --output-dir ${OUTPUT_DIR} \
        --run-task ${TASK_NAME}  \
        --dataset-name ${DATASET_NAME} \
        --sleep_rate $SLEEP_RATE \
        --max_tokens 2048
done