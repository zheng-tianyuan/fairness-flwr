#!/bin/bash

TMP_DIR="/home/zty/Mdata/flwr-mt/tmp"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Starting server"
python server.py --algorithm "ditto" --num_rounds 200 &
sleep 3  # Sleep for 3s to give the server enough time to start

export CUDA_VISIBLE_DEVICES=0
python ditto_client.py --i 0 &
export CUDA_VISIBLE_DEVICES=1
python ditto_client.py --i 1 &
export CUDA_VISIBLE_DEVICES=2
python ditto_client.py --i 2 &
python ditto_client.py --i 3 &
export CUDA_VISIBLE_DEVICES=3
python ditto_client.py --i 4 &
python ditto_client.py --i 5 &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
