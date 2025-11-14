#!/bin/bash
# train_qwen.sh - Script to run Qwen model training

# Default values
OUTPUT_DIR=""
BATCH_SIZE=4
NUM_EPOCHS=3
NUM_GPUS=8
CLAUDE_DATA_PATH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --claude_data)
      CLAUDE_DATA_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --output_dir DIR    Output directory for the model"
      echo "  --batch_size SIZE   Batch size per GPU [default: 4]"
      echo "  --epochs NUM        Number of training epochs [default: 3]"
      echo "  --gpus NUM          Number of GPUs to use [default: 8]"
      echo "  --claude_data PATH  Path to Claude reasoning traces JSON file (optional)"
      echo "  --help              Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate arguments
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./qwen-7b-finetuned"
  echo "No output directory specified, using: $OUTPUT_DIR"
fi

MODEL_TYPE="qwen-7b"

# Build command with optional parameters
CMD="src/main.py sft --model $MODEL_TYPE --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS"

# Add Claude data path if specified
if [ -n "$CLAUDE_DATA_PATH" ]; then
  CMD="$CMD --claude_data_path $CLAUDE_DATA_PATH"
  echo "Using Claude reasoning traces from: $CLAUDE_DATA_PATH"
fi

# Launch training with torchrun
DISTRIBUTED_ARGS="--nproc_per_node=$NUM_GPUS --master_port=6000"

echo "Starting training for Qwen 7B with $NUM_GPUS GPUs..."
echo "Output directory: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"

# Run the training
torchrun $DISTRIBUTED_ARGS $CMD

echo "Training complete!" 