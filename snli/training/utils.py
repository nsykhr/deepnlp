from transformers import AdamW


def unfreeze_parameters(model):
    for param in model.base_model.parameters():
        param.requires_grad = True


def freeze_parameters(model):
    for param in model.base_model.parameters():
        param.requires_grad = False


def get_encoder_grouped_parameters(model):
    optimizer_grouped_parameters = []
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters.extend([
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': 1e-5, 'weight_decay': 0.0}
    ])

    return optimizer_grouped_parameters


def get_adamw_optimizer(model, lr: float = 1e-4):
    optimizer_grouped_parameters = get_encoder_grouped_parameters(model)

    if hasattr(model, 'fc_hidden'):
        optimizer_grouped_parameters.extend([
            {'params': [p for n, p in model.fc_hidden.named_parameters() if 'bias' in n], 'lr': lr,
             'weight_decay': 0.0},
            {'params': [p for n, p in model.fc_hidden.named_parameters() if 'bias' not in n], 'lr': lr,
             'weight_decay': 1e-4}
        ])
    optimizer_grouped_parameters.extend([
            {'params': [p for n, p in model.fc_output.named_parameters() if 'bias' in n], 'lr': lr,
             'weight_decay': 0.0},
            {'params': [p for n, p in model.fc_output.named_parameters() if 'bias' not in n], 'lr': lr,
             'weight_decay': 1e-4}
        ])

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    return optimizer
