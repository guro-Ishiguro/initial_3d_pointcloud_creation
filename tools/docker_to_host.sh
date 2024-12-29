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

CONTAINER_ID="555abba83550"
echo "Selected data type: $DATA_TYPE"

# フォルダの削除とコピー処理
echo "Deleting old drone folder..."
sudo rm -rf $data_dir/$DATA_TYPE/images/drone

echo "Copying new drone_image folder from Docker container..."
sudo docker cp $CONTAINER_ID:/catkin_ws/src/process_unity_data/output/stereo/drone_image/ $data_dir/$DATA_TYPE/images/
sudo mv $data_dir/$DATA_TYPE/images/drone_image $data_dir/$DATA_TYPE/images/drone
echo "Drone images copied successfully."

echo "Deleting old camera_pose_log.txt..."
sudo rm -rf $data_dir/$DATA_TYPE/txt/camera_pose_log.txt
echo "Copying camera_pose_log.txt from Docker container..."
sudo docker cp $CONTAINER_ID:/catkin_ws/src/process_unity_data/output/stereo/camera_pose_log.txt $data_dir/$DATA_TYPE/txt/
echo "camera_pose_log.txt copied successfully."

echo "Deleting old drone_image_log.txt..."
sudo rm -rf $data_dir/$DATA_TYPE/txt/drone_image_log.txt
echo "Copying drone_image_log.txt from Docker container..."
sudo docker cp $CONTAINER_ID:/catkin_ws/src/process_unity_data/output/stereo/drone_image_log.txt $data_dir/$DATA_TYPE/txt/
echo "drone_image_log.txt copied successfully."

echo "Deleting old KeyFrameTrajectory.txt..."
sudo rm -rf $data_dir/$DATA_TYPE/txt/KeyFrameTrajectory.txt
echo "Copying KeyFrameTrajectory.txt from Docker container..."
sudo docker cp $CONTAINER_ID:/catkin_ws/ORB_SLAM3/KeyFrameTrajectory.txt $data_dir/$DATA_TYPE/txt/
echo "KeyFrameTrajectory.txt copied successfully."

echo "All tasks completed successfully."
