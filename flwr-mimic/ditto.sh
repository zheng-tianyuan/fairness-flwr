#!/bin/bash

# Ditto
NUM_CLIENTS=5
SEED=1
ROUND=40
BATCH=32
EPOCH=1
LR=0.002
LOCAL_LR=0.002
LAM=10
SAVE="/home/zty/Mdata/tmp/ditto/mimic"

echo "Starting server"
python ditto_server.py --num_users $NUM_CLIENTS --num_rounds $ROUND \
    --batch_size $BATCH --epochs $EPOCH --learning_rate $LR --frac 1.0 --random_seed $SEED \
    --local_lr $LOCAL_LR --lam $LAM --tmp_save $SAVE &

sleep 3  # Sleep for 3s to give the server enough time to start

for cid in `seq 0 4`; do
    echo "Starting client $cid"
    python -u -m ditto_client --num_users $NUM_CLIENTS --cid $cid --num_rounds $ROUND \
    --batch_size $BATCH --epochs $EPOCH --learning_rate $LR --frac 1.0 --random_seed $SEED \
    --local_lr $LOCAL_LR --lam $LAM \
    --tmp_save $SAVE &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

