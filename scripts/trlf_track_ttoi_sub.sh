TESTSUB=sub-1160

BUNDLE=PYT_L 
MODEL_BUNDLE=PYT_L


mkdir -p exp_tracts/${BUNDLE}

FINETUNED_DATA="data/tractwise_trajs/data4DT_${BUNDLE}.pkl"

echo ${TESTSUB}

python T-RLF/inference_tracking.py \
    --trlf_model_load_path trained_models/T-RLF/M_basisL_4Layer_${MODEL_BUNDLE}.pt \
    --offline_trajectories ${FINETUNED_DATA} \
    --load_sc_params_path trained_models/T-RLF/sc_params/scale_params_basisL_4Layer_${MODEL_BUNDLE}.pth \
    --max_episode_return 300 --scale_rtg 100 --timesteps_embed_trlf_model_dim 530 \
    --input_fodf_signal data/${TESTSUB}/fodf/${TESTSUB}__fodf.nii.gz \
    --target_sh_order 8 \
    --seeding_mask data/tractoinferno/${TESTSUB}/mrm_masks/${TESTSUB}-generated_approximated_mask_${BUNDLE}.nii.gz \
    --tracking_mask data/tractoinferno/${TESTSUB}/mrm_masks/${TESTSUB}-generated_approximated_mask_${BUNDLE}.nii.gz \
    --bundle_mask data/tractoinferno/${TESTSUB}/mrm_masks/${TESTSUB}-generated_approximated_mask_${BUNDLE}.nii.gz \
    --peaks data/${TESTSUB}/fodf/${TESTSUB}__peaks.nii.gz \
    --step_size 0.375 \
    --theta 60 \
    --min_length 20.0 \
    --max_length 200.0 \
    --npv 7 \
    --reference_file_fa data/${TESTSUB}/dti/${TESTSUB}__fa.nii.gz \
    --voxel_size 1 \
    --tracking_batch_size 5000 \
    --save_trk_path exp_tracts/${BUNDLE}/trk_trlf_${TESTSUB}.trk \
    --device cuda:0

echo "Tracked"