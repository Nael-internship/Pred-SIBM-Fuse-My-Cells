METHODS=(
    "gaussian_filter sigma 400 600 2 1000"
)

for METHOD_PARAMS in "${METHODS[@]}"
do
    # Split the method and its parameters
    METHOD=$(echo $METHOD_PARAMS | awk '{print $1}')
    ARGNAME=$(echo $METHOD_PARAMS | awk '{print $2}')
    START=$(echo $METHOD_PARAMS | awk '{print $3}')
    STOP=$(echo $METHOD_PARAMS | awk '{print $4}')
    INCREMENT=$(echo $METHOD_PARAMS | awk '{print $5}')
    DIVIDER=$(echo $METHOD_PARAMS | awk '{print $6}')

    for ARG in $(python3 -c "print(' '.join(['$ARGNAME'+'='+str(x / $DIVIDER) for x in range($START, $STOP, $INCREMENT)]))")
    do
        echo "sbatch fmc.slurm $METHOD $ARG"
        sbatch fmc.slurm $METHOD $ARG
    done
done
