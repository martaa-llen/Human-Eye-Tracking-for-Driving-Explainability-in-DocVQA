from differential_privacy.dp_utils import clip_parameters, flatten_params, get_shape, reconstruct_shape, add_dp_noise
import torch
import numpy as np

def test_get_shape():
    test_shapes = [(1, )]
    tensor_list = [torch.empty(test_shapes[0])]
    assert np.all(np.equal(test_shapes, get_shape(tensor_list)))

    test_shapes = [(3*i+2, 30*i) for i in range(20)]
    tensor_list = [torch.empty(test_shapes[i]) for i in range(20)]
    assert np.all(np.equal(test_shapes, get_shape(tensor_list)))


def test_flatten_params():
    test_shapes = [(3*i+2, 30*i) for i in range(20)]
    tensor_list = [torch.empty(test_shapes[i]) for i in range(20)]
    flatted_params = flatten_params(tensor_list)
    assert sum([x.numel() for x in tensor_list]) == flatted_params.numel()
    assert flatted_params.dim() == 1


def test_flatten_reconstruct_params():
    test_shapes = [(3*i+2, 30*i) for i in range(20)]
    tensor_list = [torch.randn(test_shapes[i]) for i in range(20)]
    flatted_params = flatten_params(tensor_list)
    reconstructed = reconstruct_shape(flatted_params, test_shapes)
    assert np.all([torch.equal(x, y) for x, y in zip(tensor_list, reconstructed)])
    assert np.all([np.equal(x.shape, shape) for x, shape in zip(tensor_list, test_shapes)])


def test_clip_parameters():
    test_shapes = [(i*10) for i in range(10)]
    clipping_norms = np.linspace(0, 1, num=10)
    for i, shape in enumerate(test_shapes):
        tensor = torch.rand(shape) + 20
        clipped_tensor = clip_parameters(tensor, clipping_norms[i])

        # compare with torch
        assert torch.linalg.vector_norm(clipped_tensor, ord=2) <= clipping_norms[i] + 1e-4

        # compare with numpy and add a bit of slack because of different precision
        assert np.linalg.norm(clipped_tensor.numpy()) <= clipping_norms[i] + 1e-4


def test_restore_frozen_weights():

    import copy
    from torch.utils.data import DataLoader
    from datasets.PFL_DocVQA import collate_fn
    from utils import  parse_args, load_config
    from build_utils import build_model, build_dataset, build_optimizer
    args = parse_args()
    config = load_config(args)

    model = build_model(config)
    optimizer = build_optimizer(model, config)
    parameters = copy.deepcopy(list(model.model.state_dict().values()))

    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]

    dataset = build_dataset(config, split="imdb_val")
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Do a step:
    for batch in data_loader:
        outputs, pred_answers, _ = model.forward(batch, return_pred_answer=True)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        break

    new_update = [w - w_0 for w, w_0 in zip(list(model.model.state_dict().values()), parameters)]  # Get model update

    # All the updates of frozen layers are equal to 0
    assert max([upd.sum().item() for upd, params, is_frozen in zip(new_update, parameters, frozen_parameters) if is_frozen]) == 0
    print(f'Frozen layers are equal to 0, assert passed')

    # If previous assert fails, this can help us visualize the error.
    param_keys = list(model.model.state_dict().keys())
    for k, upd, params, is_frozen in zip(param_keys, new_update, parameters, frozen_parameters):
        if upd.sum().item() != 0 and is_frozen:
            print('\n\t', k, upd.max().item(), is_frozen, '\n')
        else:
            print(k, upd.sum().item(), is_frozen)

    assert torch.all(torch.Tensor([torch.all(torch.eq(new_param, original_param)).item() if is_frozen else not torch.all(torch.eq(new_param, original_param)).item() for new_param, original_param, is_frozen in zip(list(model.model.state_dict().values()), parameters, frozen_parameters)]))
    print(f'New frozen params are equal to original frozen params, assert passed')


    for k, upd, original_param, is_frozen in zip(param_keys, list(model.model.state_dict().values()), parameters, frozen_parameters):
        equal_params = torch.all(torch.eq(upd, original_param)).item()
        if is_frozen and not equal_params:
            print(k, equal_params, is_frozen)

        if not is_frozen and equal_params:
            print(k, equal_params, is_frozen)

    shapes = get_shape(new_update)
    new_update = flatten_params(new_update)
    new_update = clip_parameters(new_update, clip_norm=config.dp_parameters['sensitivity'])
    agg_update = new_update

    agg_update = add_dp_noise(agg_update, noise_multiplier=config.dp_parameters['noise_multiplier'], sensitivity=config.dp_parameters['sensitivity'])

    # Divide the noisy aggregated update by the number of providers (Average update).
    agg_update = torch.div(agg_update, 10)

    # Add the noisy update to the original model
    agg_update = reconstruct_shape(agg_update, shapes)

    # Restore original weights (without noise) from frozen layers.
    agg_upd = [upd if not is_frozen else params for upd, params, is_frozen in zip(agg_update, parameters, frozen_parameters)]

    assert all([torch.all(params == new_params).item() == is_frozen for params, new_params, is_frozen in zip(parameters, agg_upd, frozen_parameters)])
    print('Aggregated update check passed')

    # upd_weights = [torch.add(agg_upd, w_0).cpu() for agg_upd, w_0 in zip(agg_update, copy.deepcopy(parameters))]
    upd_weights = [torch.add(agg_upd, w_0) for agg_upd, w_0 in zip(agg_update, copy.deepcopy(parameters))]


    assert torch.all(torch.Tensor([torch.all(torch.eq(new_param, original_param)).item() if is_frozen else not torch.all(torch.eq(new_param, original_param)).item() for new_param, original_param, is_frozen in zip(upd_weights, parameters, frozen_parameters)]))


if __name__ == '__main__':
    test_get_shape()
    print('get_shape passed')

    test_flatten_params()
    print('flatten_params passed')

    test_flatten_reconstruct_params()
    print('flatten_reconstruct_params passed')

    test_clip_parameters()
    print('clip_parameters passed')

    test_restore_frozen_weights()
    print('restore_frozen_weights passed')
