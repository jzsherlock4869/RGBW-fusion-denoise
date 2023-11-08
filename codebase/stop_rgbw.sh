ps -ef | grep train_rgbw | grep $1 | awk -F" " '{print $2}' | xargs kill
