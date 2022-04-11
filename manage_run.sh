run=$(ls --reverse tensorboard_logs | fzf)

echo "[r]ename [d]elete [q]uit"

echo $run
read -p "Select an action: " action

if [[ $action == "q" ]]; then
    echo "Quitting."
elif [[ $action == "d" ]]; then
    read -p "Are you sure you want to remove $run? [y/n]" confirm
    if [[ $confirm == "y" ]]; then
        rm "policies/$run.jls"
        rm -r "tensorboard_logs/$run"&& echo "rm tensorboard_logs/$run"
    fi
elif [[ $action == "r" ]]; then
    read -p "New nun name: " -i "$run" new_name
fi
