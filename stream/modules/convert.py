import torch


def convert_to_stream(stream_model, model):
    state_dict = model.state_dict()
    new_state_dict = stream_model.state_dict()

    for key in stream_model.state_dict().keys():
        if key in state_dict.keys():
            new_state_dict[key] = state_dict[key]

        elif key.replace('Conv1d.', '') in state_dict.keys():
            new_state_dict[key] = state_dict[key.replace('Conv1d.', '')]
    
        elif key.replace('Conv2d.', '') in state_dict.keys():
            new_state_dict[key] = state_dict[key.replace('Conv2d.', '')]

        elif key.replace('ConvTranspose2d.', '') in state_dict.keys():
            if key.endswith('weight'):
                groups = getattr(stream_model, key.replace('.weight', '')).groups
                if groups == 1:
                    new_state_dict[key] = torch.flip(state_dict[key.replace('ConvTranspose2d.', '')].permute([1,0,2,3]), dims=[-2,-1])
                else:
                    # new_state_dict[key] = torch.flip(state_dict[key.replace('ConvTranspose2d.', '')], dims=[-2,-1])
                    raise ValueError('Invalid group size.')
                
            else:
                new_state_dict[key] = state_dict[key.replace('ConvTranspose2d.', '')]

        else:
            raise(ValueError('key error!'))
        
    stream_model.load_state_dict(new_state_dict)
