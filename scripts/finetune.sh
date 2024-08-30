BUNDLES=("CC_Fr_1" "AF_L" "AF_R" "CG_L" "CG_R" "PYT_L" "PYT_R" )
MODEL_BUNDLES=("CC" "AF_L" "AF_R" "CG_L" "CG_R" "PYT_L" "PYT_R" )
mkdir -p trained_models/new_trlf/finetuned/sc_params/

for ((i=0; i<${#BUNDLES[@]}; i++)); do
    BUNDLE=${BUNDLES[$i]}
    MODEL_BUNDLE=${MODEL_BUNDLES[$i]}

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
        --n_layer 4 \
        --dataset data/finetune/data_${MODEL_BUNDLE}.pkl \
        --from_pretrained 1 \
        --pretrained_path trained_models/new_trlf/sc_params/M_pretrain_TRLF_x3.pt \
        --save_path trained_models/new_trlf/finetuned/M_finetuned_TRLF_${MODEL_BUNDLE}.pt \
        --save_sc_params_path trained_models/new_trlf/finetuned/sc_params/scale_params_TRLF_${MODEL_BUNDLE}.pth

    echo Finetuned on ${BUNDLE}

    done