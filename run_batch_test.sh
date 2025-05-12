#!/bin/bash

DATA_DIR="../assets/"
IMAGE_FILE="max.jpg" # Assuming max.jpg is in the same directory as the script, or adjust path accordingly

MODEL_FILENAMES=(
    "multimodal_image_transformer/model_checkpoint_epoch_1_val_loss_3.0019.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_2_val_loss_2.8036.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_3_val_loss_2.7074.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_4_val_loss_2.6526.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_5_val_loss_2.6176.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_6_val_loss_2.5925.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_7_val_loss_2.5645.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_8_val_loss_2.5563.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_9_val_loss_2.5503.safetensors"
    "multimodal_image_transformer/model_checkpoint_epoch_10_val_loss_2.5425.safetensors"
)

# Ensure the script is executable: chmod +x run_batch_test.sh
# Ensure inference.py is in a location findable by python, or specify its path e.g., ../src/inference.py

for MODEL_FILENAME in "${MODEL_FILENAMES[@]}"; do
    FULL_MODEL_PATH="${DATA_DIR}${MODEL_FILENAME}"
    
    echo ""
    echo "======================================================================"
    echo "RUNNING INFERENCE FOR MODEL: ${FULL_MODEL_PATH}"
    echo "USING IMAGE: ${IMAGE_FILE}"
    echo "======================================================================"
    
    # Check if the model file exists before trying to run inference
    if [ -f "${FULL_MODEL_PATH}" ]; then
        python inference.py --image_path "${IMAGE_FILE}" --checkpoint_path "${FULL_MODEL_PATH}"
    else
        echo "ERROR: Model file not found at ${FULL_MODEL_PATH}"
    fi
    
    echo "----------------------------------------------------------------------"
done

echo ""
echo "Batch test finished."
