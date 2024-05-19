#!/bin/bash
set -e

# Update Rust toolchain
rustup update stable

# Set writable directories for cargo
export CARGO_HOME=/home/render/.cargo
export CARGO_TARGET_DIR=/home/render/.cargo_target

# Install system dependencies
apt-get update && apt-get install -y build-essential libssl-dev pkg-config

# Install Python dependencies
pip install -r requirements.txt
