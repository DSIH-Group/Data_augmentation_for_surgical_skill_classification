#!/usr/bin/env bash
NOVICE_FOLDS=('B_OUT' 'G_OUT' 'H_OUT' 'I_OUT' 'ALL_NOVICES')
INTERMEDIATE_FOLDS=('C_OUT' 'F_OUT' 'ALL_INTERMEDIATES')
EXPERT_FOLDS=('D_OUT' 'E_OUT' 'ALL_EXPERTS')
SURGERY='Suturing'
PATH_TO_DATASET="data/jigsaws-data/${SURGERY}"

# for fold in "${NOVICE_FOLDS[@]}"; do 
#     mkdir "LOUO/${fold}"
# done

for fold in "${INTERMEDIATE_FOLDS[@]}"; do 
    mkdir -p "LOUO/${SURGERY}/${fold}"
done

for fold in "${EXPERT_FOLDS[@]}"; do 
    mkdir -p "LOUO/${SURGERY}/${fold}"
done

# python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
# --hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
# --model_save_path "models/B_OUT.ckpt" --skill Novice --type Knot_Tying \
# --data_path "${PATH_TO_DATASET}/splits/Novice/B_OUT" --results_save_path "LOUO/B_OUT" &
# python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
# --hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
# --model_save_path "models/G_OUT.ckpt" --skill Novice --type Knot_Tying \
# --data_path "${PATH_TO_DATASET}/splits/Novice/G_OUT" --results_save_path "LOUO/G_OUT" &
# python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
# --hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
# --model_save_path "models/H_OUT.ckpt" --skill Novice --type Knot_Tying \
# --data_path "${PATH_TO_DATASET}/splits/Novice/H_OUT" --results_save_path "LOUO/H_OUT" 

# wait

# python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
# --hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
# --model_save_path "models/I_OUT.ckpt" --skill Novice --type Knot_Tying \
# --data_path "${PATH_TO_DATASET}/splits/Novice/I_OUT" --results_save_path "LOUO/I_OUT" &
# python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
# --hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
# --model_save_path "models/ALL_NOVICES.ckpt" --skill Novice --type Knot_Tying \
# --data_path "${PATH_TO_DATASET}/splits/Novice/ALL_NOVICES" --results_save_path "LOUO/ALL_NOVICES" 

# wait

python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
--hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
--model_save_path "models/C_OUT.ckpt" --skill Novice --type Needle_Passing \
--data_path "${PATH_TO_DATASET}/splits/Intermediate/C_OUT" --results_save_path "LOUO/${SURGERY}/C_OUT" &
python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
--hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
--model_save_path "models/F_OUT.ckpt" --skill Novice --type Needle_Passing \
--data_path "${PATH_TO_DATASET}/splits/Intermediate/F_OUT" --results_save_path "LOUO/${SURGERY}/F_OUT" &
python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
--hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
--model_save_path "models/ALL_INTERMEDIATES.ckpt" --skill Novice --type Needle_Passing \
--data_path "${PATH_TO_DATASET}/splits/Intermediate/ALL_INTERMEDIATES" --results_save_path "LOUO/${SURGERY}/ALL_INTERMEDIATES"

wait

python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
--hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
--model_save_path "models/D_OUT.ckpt" --skill Novice --type Needle_Passing \
--data_path "${PATH_TO_DATASET}/splits/Expert/D_OUT" --results_save_path "LOUO/${SURGERY}/D_OUT" &
python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
--hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
--model_save_path "models/E_OUT.ckpt" --skill Novice --type Needle_Passing \
--data_path "${PATH_TO_DATASET}/splits/Expert/E_OUT" --results_save_path "LOUO/${SURGERY}/E_OUT" &
python3 main_timegan.py --data_name jigsaws --seq_len 24 --module gru \
--hidden_dim 24 --num_layer 3 --iteration 4000 --batch_size 1024 \
--model_save_path "models/ALL_EXPERTS.ckpt" --skill Novice --type Needle_Passing \
--data_path "${PATH_TO_DATASET}/splits/Expert/ALL_EXPERTS" --results_save_path "LOUO/${SURGERY}/ALL_EXPERTS"