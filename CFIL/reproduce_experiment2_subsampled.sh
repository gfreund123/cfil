#!/bin/bash

#note this currently assumes a cluster of 4 gpu's with sufficent memory (each python call is about 900 mega)
#if you only have 1, change all 'dev' below appropriately and remove '&' from the end of the command.

declare -a env_names=("Hopper-v2" "Walker2d-v2" "Ant-v2" "Humanoid-v2") #cheetah handled down below...


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
    dev="cuda:1"
fi
if [ $env_name = "HalfCheetah-v2" ]; then
    dev="cuda:2"
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


#subsample20
python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 10 3 --title "subsample20_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5 --train_skip_steps 20&

#subsample50
python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 10 3 --title "subsample50_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5  --train_skip_steps 50 &

#subsample10
python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 10 3 --title "subsample10_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5 --train_skip_steps 10 &

#subsample100
python rewarder.py --rewarder_type 'coupledflow' --env_name $env_name --device $dev --seed $seed --use_tanh --tanh_scale 10 3 --title "subsample100_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name $data_set --smooth 0.5 --train_skip_steps 100&

done


#handle cheetah

#subsample20
python rewarder.py --rewarder_type 'coupledflow' --env_name 'HalfCheetah-v2' --device 'cuda:2' --seed $seed --use_tanh --tanh_scale 10 10 --title "subsample20_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name "Value_Dice" --flow_reg --train_skip_steps 20&

#subsample50
python rewarder.py --rewarder_type 'coupledflow' --env_name 'HalfCheetah-v2' --device 'cuda:2' --seed $seed --use_tanh --tanh_scale 10 10 --title "subsample50_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name "Value_Dice" --flow_reg  --train_skip_steps 50 &

#subsample10
python rewarder.py --rewarder_type 'coupledflow' --env_name 'HalfCheetah-v2' --device 'cuda:2' --seed $seed --use_tanh --tanh_scale 10 10 --title "subsample10_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name "Value_Dice" --flow_reg --train_skip_steps 10 &

#subsample100
python rewarder.py --rewarder_type 'coupledflow' --env_name 'HalfCheetah-v2' --device 'cuda:2' --seed $seed --use_tanh --tanh_scale 10 10 --title "subsample100_seed$seed" --outputdir 'all_subsampled' --num_train_trajs 1 --start_steps 2000 --tanh_shift --flow_norm 'none' --data_set_name "Value_Dice" --flow_reg --train_skip_steps 100&






wait

done






