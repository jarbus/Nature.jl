menu=/tmp/garbus-runs
echo "new run" > $menu
ls --reverse tensorboard_logs >> $menu
run=$(cat $menu | fzf)
echo $run
rm /tmp/garbus-runs

NUM_FRAMES=4
tb="--tb"
while [[ "$1" ]]; do
    case "$1" in
        "--num-frames") NUM_FRAMES="$2";;
        "--no-tb") tb="" ;;
    esac
    shift
done

if [[ $run == "new run" ]]; then
    #read -e -p "Enter run name: " -i "$(date +"%Y-%m-%d %H:%M")" run
    run="$(date +"%Y-%m-%d %H:%M") num_frames=$NUM_FRAMES"
fi

cp batch-template.sh batch.sh
echo "julia --project=. run-experiment.jl $tb $resume --num-frames=$NUM_FRAMES --max-steps=2000000 --episode-len=1000 \"$run\"" >> batch.sh
sbatch --output="outs/$run.out" --job-name="$run" batch.sh
