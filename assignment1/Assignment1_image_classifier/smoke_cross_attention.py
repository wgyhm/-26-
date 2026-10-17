"""Run a short check without executing the notebook's full training loop.

Usage: /mnt/nas/gezuhao/zhengyuqian/starvla/.venv/bin/python smoke_cross_attention.py
       Add --device cuda:0 for a short GPU check.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import ast
import json
import os
from pathlib import Path
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    os.chdir(project)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    notebook = json.loads((project / 'image-classifier.ipynb').read_text())
    scope = {}
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, f'cell_{i}', 'exec')
        tree = ast.parse(source)
        # Load the actual definitions without starting training or creating CUDA models.
        nodes = [n for n in tree.body if isinstance(
            n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
        if i == 3:
            # Execute data setup only; DataLoader workers start when iterated, not here.
            exec(compile(source, 'data_setup', 'exec'), scope)
        else:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), f'cell_{i}', 'exec'), scope)

    assert (len(scope['train_dataset']), len(scope['val_dataset']),
            len(scope['test_dataset'])) == (45000, 5000, 10000)
    train_ids = scope['train_indices']
    val_ids = scope['val_indices']
    assert len(set(train_ids)) == 45000 and len(set(val_ids)) == 5000
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(50000))
    filenames = scope['train_source'].img_names
    val_counts = Counter(Path(filenames[i]).stem.rsplit('_', 1)[0] for i in val_ids)
    assert len(val_counts) == 100 and set(val_counts.values()) == {50}
    assert scope['train_source'].transform is scope['transform_train']
    assert scope['val_source'].transform is scope['transform_test']
    repeated = dict(scope)
    exec(compile(''.join(notebook['cells'][3]['source']), 'repeat_split', 'exec'), repeated)
    assert repeated['split_info'] == scope['split_info'], 'Split changes across runs'
    assert repeated['train_source'].img_names == filenames
    del repeated

    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is unavailable in this environment'
        print('GPU:', torch.cuda.get_device_name(device))
        torch.cuda.reset_peak_memory_stats(device)
    scope['device'] = device
    dataset = scope['mydataset'](str(project / 'cifar100'), 'train', scope['transform_test'])
    indices = []
    for name in ('beetle', 'woman', 'aquarium_fish', 'sweet_pepper'):
        filename = next((project / 'cifar100' / 'train').glob(name + '_*.jpg'))
        index = dataset.img_names.index(str(filename))
        image, gradient, label = dataset[index]
        assert dataset.tags[label] == name, (name, dataset.tags[label])
        assert image.shape == gradient.shape == (3, 32, 32)
        assert torch.isfinite(gradient).all() and 0 <= gradient.min() <= gradient.max() <= 1
        indices.append(index)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices[:2]), batch_size=2, num_workers=0)
    batch = next(iter(loader))
    model = scope['WRNDualConvNet'](classes_num=100).to(device)
    rgb_modules = [model.prepareRGB, model.layerRGB1, model.layerRGB2, model.layerRGB3]
    gradient_modules = [model.prepare, model.layer1, model.layer2, model.layer3]
    rgb_parameters = {id(p) for m in rgb_modules for p in m.parameters()}
    gradient_parameters = {id(p) for m in gradient_modules for p in m.parameters()}
    assert rgb_parameters.isdisjoint(gradient_parameters), 'Branches share parameters'
    assert (model.cross_attention.rgb_attention.in_proj_weight
            is not model.cross_attention.gradient_attention.in_proj_weight)

    features = {}
    handles = []
    expected_shapes = {
        'prepareRGB': (2, 16, 32, 32), 'prepare': (2, 16, 32, 32),
        'layerRGB1': (2, 32, 32, 32), 'layer1': (2, 32, 32, 32),
        'layerRGB2': (2, 64, 16, 16), 'layer2': (2, 64, 16, 16),
        'layerRGB3': (2, 128, 8, 8), 'layer3': (2, 128, 8, 8),
        'cross_attention': (2, 128, 8, 8),
    }
    assert all(len(getattr(model, name)) == 4 for name in
               ('layer1', 'layer2', 'layer3', 'layerRGB1', 'layerRGB2', 'layerRGB3'))
    assert sum(p.numel() for p in model.parameters()) < 3_500_000
    for name in expected_shapes:
        handles.append(getattr(model, name).register_forward_hook(
            lambda module, inputs, output, key=name: features.update({key: output.detach()})))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    before = model.cross_attention.rgb_attention.in_proj_weight.detach().clone()
    loss = scope['train']([batch], model, optimizer, criterion, epoch=0)
    assert torch.isfinite(torch.tensor(loss)), 'Non-finite loss'
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for parameter in (model.prepareRGB[0].weight, model.prepare[0].weight,
                      model.cross_attention.rgb_attention.in_proj_weight,
                      model.cross_attention.gradient_attention.in_proj_weight):
        assert parameter.grad.abs().sum() > 0, 'A branch or attention direction is disconnected'
    assert not torch.equal(before, model.cross_attention.rgb_attention.in_proj_weight)
    assert set(features) == set(expected_shapes)
    assert all(tuple(features[name].shape) == shape for name, shape in expected_shapes.items())
    accuracy = scope['test']([batch], model, criterion)
    assert 0 <= accuracy <= 100
    image, gradient, _ = (item.to(device) for item in batch)
    with torch.no_grad():
        logits = model(image, gradient)
        changed_rgb = model(torch.zeros_like(image), gradient)
        changed_gradient = model(image, torch.zeros_like(gradient))
    assert logits.shape == (2, 100) and torch.isfinite(logits).all()
    assert not torch.allclose(logits, changed_rgb), 'RGB input is ignored'
    assert not torch.allclose(logits, changed_gradient), 'Gradient input is ignored'
    for handle in handles:
        handle.remove()

    # Validation must neither accumulate gradients nor update BatchNorm statistics.
    optimizer.zero_grad(set_to_none=True)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    val_loss, val_acc = scope['validate']([batch], model, criterion)
    assert torch.isfinite(torch.tensor(val_loss)) and 0 <= val_acc <= 100
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())

    # Exercise the single-image cell with its actual Sobel/batch/eval handling.
    scope.update(net=model, test_dataset=dataset)
    exec(compile(''.join(notebook['cells'][9]['source']), 'single_image', 'exec'), scope)
    assert scope['out'].shape == (1, 100)
    with tempfile.TemporaryDirectory(prefix='.cross-attention-smoke-', dir=project) as output:
        scope['save_checkpoint']({'architecture': type(model).__name__, 'net': model.state_dict()},
                                 True, output)
        state = torch.load(Path(output) / 'best.pt', map_location='cpu', weights_only=True)
        assert state['architecture'] == 'WRNDualConvNet'
        model.load_state_dict(state['net'])
        assert (Path(output) / 'last.pt').exists()

        # Run the real selection/reload loop with controlled validation scores.
        # Four tiny epochs verify zero accuracy, regression, and tied best scores.
        selection_output = str(Path(output) / 'selection')
        tiny_model = torch.nn.Linear(1, 1)
        tiny_optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        scores = iter(enumerate((0.0, 30.0, 20.0, 30.0), start=1))

        def fake_train(loader, net, optimizer, criterion, epoch):
            with torch.no_grad():
                net.weight.fill_(epoch)
            return 1.0

        def fake_validate(*args):
            epoch, score = next(scores)
            if epoch == 2:
                first = torch.load(Path(selection_output) / 'best.pt', weights_only=True)
                assert first['epoch'] == 1 and first['val_acc'] == 0
            return 1.0, score

        loop_scope = dict(scope, net=tiny_model, optimizer=tiny_optimizer,
                          criterion=criterion, train=fake_train, validate=fake_validate)
        training_source = ''.join(notebook['cells'][7]['source'])
        loop = training_source[training_source.index('config = {'):]
        loop = loop.replace('"epochs": 30', '"epochs": 4')
        loop = loop.replace('"./result/wrn_dual_cross_attention"', repr(selection_output))
        with redirect_stdout(io.StringIO()):
            exec(compile(loop, 'checkpoint_selection', 'exec'), loop_scope)
        best = torch.load(Path(selection_output) / 'best.pt', weights_only=True)
        last = torch.load(Path(selection_output) / 'last.pt', weights_only=True)
        assert best['epoch'] == best['best_epoch'] == 2
        assert best['val_acc'] == best['best_val_acc'] == 30
        assert last['epoch'] == 4 and last['best_epoch'] == 2
        assert best['split'] == scope['split_info']
        assert 'optimizer' in last
        assert tiny_model.weight.item() == 2, 'Final model did not restore the best weights'
        assert last['net']['weight'].item() == 4
        assert not list(Path(selection_output).glob('*.tmp'))
    # Named channel constants must control both branches, attention, and the head.
    # Compile a changed-width variant in memory; leave the notebook untouched.
    model_source = ''.join(notebook['cells'][5]['source'])
    wrn = next(n for n in ast.parse(model_source).body
               if isinstance(n, ast.ClassDef) and n.name == 'WRNDualConvNet')
    changed_widths = ast.get_source_segment(model_source, wrn)
    for old_width, new_width in [('channel1 = 32', 'channel1 = 24'),
                                 ('channel2 = 64', 'channel2 = 48'),
                                 ('channel3 = 128', 'channel3 = 96')]:
        assert old_width in changed_widths
        changed_widths = changed_widths.replace(old_width, new_width)
    variant_scope = dict(scope)
    exec(compile(changed_widths, 'changed_channel_constants', 'exec'), variant_scope)
    variant = variant_scope['WRNDualConvNet'](100, blocks_per_stage=2).eval()
    assert all(len(getattr(variant, name)) == 2 for name in
               ('layer1', 'layer2', 'layer3', 'layerRGB1', 'layerRGB2', 'layerRGB3'))
    assert variant.cross_attention.rgb_attention.embed_dim == variant.head[-1].in_features == 96
    with torch.no_grad():
        variant_logits = variant(batch[0], batch[1])
    assert variant_logits.shape == (2, 100) and torch.isfinite(variant_logits).all()
    total = sum(p.numel() for p in model.parameters())
    attention = sum(p.numel() for p in model.cross_attention.parameters())
    print(f'PASS: device={device}, parameters={total:,}, cross_attention={attention:,}')
    if device.type == 'cuda':
        print(f'Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
