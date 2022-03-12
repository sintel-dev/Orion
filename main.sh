#!/bin/bash

LOCAL=0

# -----BASE 2.0 SPEED UP EXPERIMENTS------
#cp orion/primitives/tadgan_speedup.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan'}"
#GPU=0

#cp orion/primitives/tadgan_speedup.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder'}"
#GPU=0


# -----BASE 2.0 EXPERIMENTS------
#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan'}"
#GPU=0

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan'}"
#GPU=0

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_gpu'}"
#GPU=1


# -----TRANSFORMER DYNAMIC THRESHOLD EXPERIMENTS------
#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_lstm_dynamic_threshold_2.0_transformer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'lstm_dynamic_threshold':'transformer_dynamic_threshold'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_lstm_dynamic_threshold_2.0_transformer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'lstm_dynamic_threshold':'transformer_dynamic_threshold'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_lstm_dynamic_threshold_2.0_transformer_v2_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'lstm_dynamic_threshold':'transformer_dynamic_threshold_v2'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_lstm_dynamic_threshold_2.0_transformer_v2_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'lstm_dynamic_threshold':'transformer_dynamic_threshold_v2'}"
#GPU=1


# -----BASE 2.0 + ATTENTION LAYER EXPERIMENTS------
#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_attention"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_attention'}"
#GPU=0

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_attention"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_attention'}"
#GPU=0

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_attention_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_attention_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_attention_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_attention_gpu'}"
#GPU=1


# -----ENCODER (1:1) EXPERIMENTS------
#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder'}"
#GPU=0

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder'}"
#GPU=0

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_optimizer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_optimizer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_optimizer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_optimizer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_embedding_optimizer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_embedding_optimizer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_embedding_optimizer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_embedding_optimizer'}"
#GPU=1


# -----TIME ENCODER (1:1) EXPERIMENTS------
#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_time_encoder_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_time_encoder'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_time_encoder_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_time_encoder'}"
#GPU=1


# -----ENCODER DOWNSAMPLE EXPERIMENTS------

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_downsample_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_downsample_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_downsample_optimizer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_downsample_optimizer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer'}"
#GPU=1


# -----TRANSFORMER EXPERIMENTS------

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_transformer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_transformer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_transformer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_transformer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_transformer_optimizer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_transformer_optimizer'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_transformer_optimizer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_transformer_optimizer'}"
#GPU=1


# -----MULTIGAN EXPERIMENTS------

#cp orion/primitives/tadgan_multigen.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_multigen_lstm_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_multigen_lstm'}"
#GPU=1

#cp orion/primitives/tadgan_multigen.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_multigen_lstm_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_multigen_lstm'}"
#GPU=1

#cp orion/primitives/tadgan_multigen.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_multigen_encoder_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_multigen_encoder'}"
#GPU=1

#cp orion/primitives/tadgan_multigen.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_multigen_encoder_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_multigen_encoder'}"
#GPU=1


# -----MULTI-TO-MULTI EXPERIMENTS------

#cp orion/primitives/tadgan_2d.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_2d_lstm_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_2d_lstm'}"
#GPU=1

#cp orion/primitives/tadgan_2d.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_2d_lstm_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_2d_lstm'}"
#GPU=1

#cp orion/primitives/tadgan_2d.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_2d_encoder_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_2d_encoder'}"
#GPU=1

#cp orion/primitives/tadgan_2d.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_2d_encoder_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_2d_encoder'}"
#GPU=1


# -----INPUT SIZE EXPERIMENTS------

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_25_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_25_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_25_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_25_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_50_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_50_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_50_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_50_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_200_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_200_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_200_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_200_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_400_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_400_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_400_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_400_gpu'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_downsample_optimizer_25_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_25'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_downsample_optimizer_25_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_25'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_downsample_optimizer_50_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_50'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_downsample_optimizer_50_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_50'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_downsample_optimizer_200_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_200'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_downsample_optimizer_200_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_200'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="U_tadgan_2.0_encoder_downsample_optimizer_400_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_400'}"
#GPU=1

#cp orion/primitives/tadgan_org.py orion/primitives/tadgan.py
#EXPERIMENT_NAME="M_tadgan_2.0_encoder_downsample_optimizer_400_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan':'tadgan_encoder_downsample_optimizer_400'}"
#GPU=1


# -----SEMI-SUPERVISED EXPERIMENTS------

#EXPERIMENT_NAME="U_tadgan_2.0_semi_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan_semi':'tadgan_semi_gpu'}"
#GPU=1

#EXPERIMENT_NAME="U_tadgan_2.0_semi_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan_semi':'tadgan_semi_gpu'}"
#GPU=1

#EXPERIMENT_NAME="U_tadgan_2.0_semi_encoder_downsample_optimizer_gpu"
#DATASETS="univariate_datasets"
#PIPELINES="{'tadgan_semi':'tadgan_semi_encoder_downsample_optimizer'}"
#GPU=1

#EXPERIMENT_NAME="M_tadgan_2.0_semi_encoder_downsample_optimizer_gpu"
#DATASETS="multivariate_datasets"
#PIPELINES="{'tadgan_semi':'tadgan_semi_encoder_downsample_optimizer'}"
#GPU=1

echo "$EXPERIMENT_NAME"
if [[ $LOCAL = 1 ]]; then
    python experiments.py "$EXPERIMENT_NAME" "$DATASETS" "$PIPELINES"
else
    if [[ $GPU = 0 ]]; then
        sed -e "s/\$EXPERIMENT_NAME/$EXPERIMENT_NAME/" \
            -e "s/\$DATASETS/$DATASETS/" \
            -e "s/\$PIPELINES/$PIPELINES/" \
            -e "s/\#SBATCH --gres=gpu:volta:1//" \
            run_template.sh > ./run.sh
    else
        sed -e "s/\$EXPERIMENT_NAME/$EXPERIMENT_NAME/" \
            -e "s/\$DATASETS/$DATASETS/" \
            -e "s/\$PIPELINES/$PIPELINES/" \
            run_template.sh > ./run.sh
    fi
    LLsub run.sh
fi
