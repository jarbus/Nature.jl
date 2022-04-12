run=$(ls --reverse tensorboard_logs | fzf)

echo "[r]ename [d]elete [q]uit"

echo $run
read -p "Select an action: " action

if [[ $action == "q" ]]; then
    echo "Quitting."
elif [[ $action == "d" ]]; then
    read -p "Are you sure you want to remove $run? [y/n]" confirm
    if [[ $confirm == "y" ]]; then
        rm -r "checkpoints/$run"
        rm -r "tensorboard_logs/$run" && echo "rm tensorboard_logs/$run"
        rm -r "outs/$run" && echo "rm tensorboard_logs/$run"
    fi
elif [[ $action == "r" ]]; then
    read -p "New nun name: " -i "$run" new_name
    mv "checkpoints/$run.jls" "checkpoints/$new_name" && echo "mv checkpoints/$run checkpoints/$new_name"
    mv "tensorboard_logs/$run" "tensorboard_logs/$new_name" && echo "mv tensorboard_logs/$run tensorboard_logs/$new_name"
    mv "outs/$run" && echo "mv tensorboard_logs/$run" && echo "mv outs/$run outs/$new_name"
fi
