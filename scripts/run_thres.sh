#!/bin/bash
source ~/catkin_ws2/devel/setup.bash

# TODO: load different thres configs and save each config and outcome file in a folder

for i in {0..120..1}; do
	echo "echo $i"

  tmp=0

  echo "ee ${tmp}"

	sleep 2

#	rosrun cont2 cont2_batch_bin_test _runid:=" $i"
	rosrun cont2 cont2_batch_para_bin_test _runid:=" $i"


	stop_streak=0
	while true; do
#		if [[ $(pidof cont2_batch_bin_test) ]]; then
		if [[ $(pidof cont2_batch_para_bin_test) ]]; then
			echo "still running"
			stop_streak=0
		else 
			echo "not running anymore"
			((stop_streak++))

			if [[ $stop_streak == '4' ]]; then
				break
			fi
		fi 

		sleep 1

	done


	# rostopic pub cont2_status std_msgs/String "end" --once

done

