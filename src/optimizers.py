import torch.optim as optim

def get_optimizer(model, optimizer_config):
    optimizer_type = optimizer_config['OPTIMIZER_TYPE']

    if optimizer_type == 'radam':
        try:
            import torch_optimizer as optim_t
        except ModuleNotFoundError:
            print('torch_optimizer is not installed')
        optimizer = optim_t.RAdam([{'params': model.parameters()}],
                                    lr=optimizer_config['LR'],
                                    betas=(0.9, 0.999),
                                    eps=1e-8,
                                    weight_decay=optimizer_config['WEIGHT_DECAY'])
    elif optimizer_type == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()}], 
                                lr=optimizer_config['LR'],
                                weight_decay=optimizer_config['WEIGHT_DECAY'])
    else:
        optimizer = optim.SGD([{'params': model.parameters()}], 
                                lr=optimizer_config['LR'],
                                momentum=optimizer_config['MOMENTUM'], 
                                nesterov=True,
                                weight_decay=optimizer_config['WEIGHT_DECAY'])
    return optimizer