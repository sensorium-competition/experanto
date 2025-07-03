#!/bin/bash
BASE="./"
IMG="experanto.sif"
DEF="apptainer.def"

# Build the image if it doesn't exist
if [ ! -f "$BASE$IMG" ]; then
    echo "[INFO] Building Apptainer image..."
    export APPTAINER_TMPDIR=/dev/shm                    # This is needed for GWDG HPC, change as needed
    apptainer build --fakeroot "$BASE$IMG" "$BASE$DEF"
fi

# Run the image with GPU access and bind the current directory
apptainer exec \
    --nv \
    --bind "$BASE":/project \
    --bind /mnt/vast-react/projects/neural_foundation_model:/data \
    "$BASE$IMG" \
    jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='1234' --notebook-dir='/project'
    # echo "Running inside Apptainer container with GPU support, change apptainer-compose.sh to run your application."