#!/bin/bash

echo "Starting server"
python server.py --algorithm "local" --num_rounds 30 &
sleep 3  # Sleep for 3s to give the server enough time to start

export CUDA_VISIBLE_DEVICES=0
python local_client.py --i 0 &
python local_client.py --i 1 &
python local_client.py --i 2 &

export CUDA_VISIBLE_DEVICES=1
python local_client.py --i 3 &
python local_client.py --i 4 &

export CUDA_VISIBLE_DEVICES=2
python local_client.py --i 5 &
python local_client.py --i 6 &
python local_client.py --i 7 &

export CUDA_VISIBLE_DEVICES=3
python local_client.py --i 8 &
python local_client.py --i 9 &


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait