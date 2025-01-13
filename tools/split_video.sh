#!/bin/bash

# データタイプの選択を動的に表示
data_dir="../data"
directories=($(ls -d $data_dir/*/ 2>/dev/null | xargs -n 1 basename))

if [ ${#directories[@]} -eq 0 ]; then
    echo "No directories found in $data_dir. Exiting."
    exit 1
fi

echo "Select data type:"
for i in "${!directories[@]}"; do
    echo "$((i+1))) ${directories[$i]}"
done

read -p "Enter choice [1-${#directories[@]}]: " choice

# 入力に基づいてDATA_TYPEを設定
if [[ $choice -ge 1 && $choice -le ${#directories[@]} ]]; then
    DATA_TYPE="${directories[$((choice-1))]}"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "Selected data type: $DATA_TYPE"

input_dir=$(realpath "$data_dir/$DATA_TYPE/videos")

# 出力ディレクトリを作成
output_dir=$(realpath "$data_dir/$DATA_TYPE/images")
mkdir -p "$output_dir"

# 左カメラ動画のフレームを画像に変換
ffmpeg -i "${input_dir}/Left.webm" -qscale:v 2 "$output_dir/left_%06d.png"

# 右カメラ動画のフレームを画像に変換
ffmpeg -i "${input_dir}/Right.webm" -qscale:v 2 "$output_dir/right_%06d.png"
