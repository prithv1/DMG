# DomainNet Experiments
# To run experiments on PACS, change configs/DomainNet -> configs/PACS

# Aggregate Training (AlexNet)
python train_model.py \
    --phase aggregate_training \
    --config-yml configs/DomainNet/aggregate_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real \
    DATA.TARGET_DOMAINS sketch HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL alexnet

# Aggregate Training (ResNet-18)
python train_model.py \
    --phase aggregate_training \
    --config-yml configs/DomainNet/aggregate_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real \
    DATA.TARGET_DOMAINS sketch HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL resnet18

# Aggregate Training (ResNet-50)
python train_model.py \
    --phase aggregate_training \
    --config-yml configs/DomainNet/aggregate_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real \
    DATA.TARGET_DOMAINS sketch HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL resnet50

# Multi-Head Training (AlexNet)
python train_model.py \
    --phase multihead_training \
    --config-yml configs/DomainNet/multihead_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real \
    DATA.TARGET_DOMAINS sketch HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL alexnet \
    MODEL.SPLIT_LAYER classifier.6

# Multi-Head Training (ResNet-18)
python train_model.py \
    --phase multihead_training \
    --config-yml configs/DomainNet/multihead_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real \
    DATA.TARGET_DOMAINS sketch HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL resnet18 \
    MODEL.SPLIT_LAYER fc

# Multi-Head Training (ResNet-50)
python train_model.py \
    --phase multihead_training \
    --config-yml configs/DomainNet/multihead_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real \
    DATA.TARGET_DOMAINS sketch HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL resnet50 \
    MODEL.SPLIT_LAYER fc

# Domain-Specific Masks for Generalization (AlexNet)
python train_model.py \
    --phase supermask_training \
    --config-yml configs/DomainNet/supermask_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real DATA.TARGET_DOMAINS sketch \
    HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL alexnet \
    MODEL.MASK_LAYERS classifier.1,classifier.4,classifier.6 \
    MODEL.MASK_INIT_SETTING random_uniform MODEL.POLICY_CONV_MODE False

# Domain-Specific Masks for Generalization (ResNet-18)
python train_model.py \
    --phase supermask_training \
    --config-yml configs/DomainNet/supermask_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real DATA.TARGET_DOMAINS sketch \
    HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL resnet18 \
    MODEL.MASK_LAYERS layer3,layer4,fc \
    MODEL.MASK_INIT_SETTING random_uniform MODEL.POLICY_CONV_MODE True

# Domain-Specific Masks for Generalization (ResNet-50)
python train_model.py \
    --phase supermask_training \
    --config-yml configs/DomainNet/supermask_training.yml \
    --config-override DATA.DOMAIN_LIST clipart,infograph,painting,quickdraw,real DATA.TARGET_DOMAINS sketch \
    HJOB.JOB_STRING dmnt_v1 MODEL.BASE_MODEL resnet50 \
    MODEL.MASK_LAYERS layer3,layer4,fc \
    MODEL.MASK_INIT_SETTING random_uniform MODEL.POLICY_CONV_MODE True