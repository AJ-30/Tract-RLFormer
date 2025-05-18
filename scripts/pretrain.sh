mkdir -p trained_models/new_trlf/sc_params/

python T-RLF/experiment.py \
    --batch_size 512 \
    --env ttl2 \
    --device cuda \
    --model_type dt \
    --max_iters 10 \
    --num_steps_per_iter 2500 \
    --num_eval_episodes 100 \
    --embed_dim 128 \
    --learning_rate 1e-4 \
    --K 40 \
    --n_layer 3 \
    --dataset data/pretrain/merged_1.pkl \
    --save_path trained_models/new_trlf/sc_params/M_pretrain_TRLF.pt \
    --save_sc_params_path trained_models/new_trlf/sc_params/sc_params/scale_params_TRLF.pth


python T-RLF/experiment.py \
    --batch_size 512 \
    --env ttl2 \
    --device cuda \
    --model_type dt \
    --max_iters 10 \
    --num_steps_per_iter 2500 \
    --num_eval_episodes 100 \
    --embed_dim 128 \
    --learning_rate 1e-4 \
    --K 40 \
    --n_layer 3 \
    --dataset data/pretrain/merged_2.pkl \
    --load_model 1 \
    --model_path trained_models/new_trlf/sc_params/M_pretrain_TRLF.pt \
    --save_path trained_models/new_trlf/sc_params/M_pretrain_TRLF_x2.pt \
    --save_sc_params_path trained_models/new_trlf/sc_params/sc_params/scale_params_TRLF_x2.pth


python T-RLF/experiment.py \
    --batch_size 512 \
    --env ttl2 \
    --device cuda \
    --model_type dt \
    --max_iters 10 \
    --num_steps_per_iter 2500 \
    --num_eval_episodes 100 \
    --embed_dim 128 \
    --learning_rate 1e-4 \
    --K 40 \
    --n_layer 3 \
    --dataset data/pretrain/merged_3.pkl \
    --load_model 1 \
    --model_path trained_models/new_trlf/sc_params/M_pretrain_TRLF_x2.pt \
    --save_path trained_models/new_trlf/sc_params/M_pretrain_TRLF_x3.pt \
    --save_sc_params_path trained_models/new_trlf/sc_params/sc_params/scale_params_TRLF_x3.pth


echo "Pretraining done..."