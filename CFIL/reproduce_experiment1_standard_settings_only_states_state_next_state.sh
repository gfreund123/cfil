#!/bin/bash

#note this currently assumes a cluster of 4 gpu's with sufficent memory (each python call is about 900 mega)
#if you only have 1, change all 'dev' below appropriately and remove '&' from the end of the command.

declare -a env_names=("HalfCheetah-v2" "Hopper-v2"  "Walker2d-v2" "Ant-v2" "Humanoid-v2")


for ((seed=0;seed<=4;seed+=1))
do

for env_name in "${env_names[@]}"
do

if [ $env_name = "Humanoid-v2" ]; then
    data_set="Spinningup"
else
    data_set="Value_Dice"
fi



if [ $env_name = "Humanoid-v2" ]; then
    dev="cuda:0"
fi
if [ $env_name = "HalfCheetah-v2" ]; then
    dev="cuda:1"
fi
if [ $env_name = "Walker2d-v2" ]; then
    dev="cuda:2"
fi
if [ $env_name = "Ant-v2" ]; then
    dev="cuda:3"
fi
if [ $env_name = "Hopper-v2" ]; then
    dev="cuda:3"
fi   



python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 15 6 --title "seed$seed" --outputdir 'standard_setting_tanh_15_6_reg1_smooth05' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5 --flow_reg &

#only states (option 0)
python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 15 6 --title "state_only_seed$seed" --outputdir 'state_only_tanh_15_6_reg1_smooth05' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5 --flow_reg  --option 0&

#state next_state (option 2
python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 15 6 --title "state_next_state_seed$seed" --outputdir 'state_next_state_tanh_15_6_reg1_smooth05' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5 --flow_reg  --option 2&

done
wait

done

