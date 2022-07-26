max=10
for i in `seq 0 $max`
do
    echo "$i"
    python -m main --config ddp_config.yaml 
done