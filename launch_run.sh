menu=/tmp/garbus-runs
echo "new run" > $menu
ls --reverse tensorboard_logs >> $menu
run=$(cat $menu | fzf)
echo $run
rm /tmp/garbus-runs
if [[ $run == "new run" ]]; then
    read -e -p "Enter run name: " -i "$(date +"%Y-%m-%d %H:%M")" run
    resume=""
else
    echo "loading run $run"
    resume="--resume"
fi

echo -n "TensorBoard logging? y/n: "
read tensorboard_logging
if [[ $tensorboard_logging == "y" ]]; then
    tb="--tb"
else
    tb=""
fi

cp batch-template.sh batch.sh
echo "julia --project=. run-experiment.jl $tb $resume --max-steps=2000000 --episode-len=500 \"$run\"" >> batch.sh
sbatch batch.sh

