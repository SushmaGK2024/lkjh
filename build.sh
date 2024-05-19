#!/bin/bash
set -e

# Set writable directories for cargo
export CARGO_HOME=$HOME/.cargo
export CARGO_TARGET_DIR=$HOME/.cargo_target

# Install dependencies
pip install numpy
pip install tokenizers
pip install -r requirements.txt
