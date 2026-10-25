"""Check verified Steps 2/4/5/6 on CPU with real images and two tiny epochs.

Run: /mnt/nas/gezuhao/zhengyuqian/starvla/.venv/bin/python smoke_verified_training.py
"""
import ast
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import tempfile

import torch


def main():
    project = Path(__file__).resolve().parent
    os.chdir(project)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    notebook = json.loads((project / 'image-classifier_verified.ipynb').read_text())
    cells = [''.join(cell.get('source', [])) for cell in notebook['cells']]
    scope = {}
    exec(compile(cells[4], 'step2', 'exec'), scope)
    assert tuple(len(scope[name]) for name in (
        'train_dataset', 'val_dataset', 'test_dataset')) == (45000, 5000, 10000)
    train_ids, val_ids = scope['train_indices'], scope['val_indices']
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(50000))
    filenames = scope['train_source'].img_names
    counts = Counter(Path(filenames[i]).stem.rsplit('_', 1)[0] for i in val_ids)
    assert len(counts) == 100 and set(counts.values()) == {50}
    assert scope['val_source'].transform is scope['transform_test']
    assert not scope['val_loader'].drop_last
    assert isinstance(scope['val_loader'].sampler, torch.utils.data.SequentialSampler)
    repeated = {}
    exec(compile(cells[4], 'repeat_step2', 'exec'), repeated)
    assert repeated['split_info'] == scope['split_info']
    print('PASS: fixed class-stratified split, no overlap, deterministic validation.')

    # Replay the same augmentation to verify both branches use its actual output.
    source = scope['train_source']
    sample_index = train_ids[0]
    rng_state = torch.get_rng_state()
    rgb, gradient, label = source[sample_index]
    torch.set_rng_state(rng_state)
    with scope['Image'].open(source.img_names[sample_index]) as image:
        raw = source.transform(image.convert('RGB'))
    assert rgb.shape == gradient.shape == (3, 64, 64)
    assert torch.isfinite(rgb).all() and torch.isfinite(gradient).all()
    assert torch.allclose(rgb, source.normalize_rgb(raw))
    assert torch.allclose(gradient, source.RGB2Gradient(raw))
    samples = [source[sample_index][0] for _ in range(4)]
    assert any(not torch.equal(samples[0], image) for image in samples[1:])
    for dataset in (scope['val_dataset'], scope['test_dataset']):
        first, second = dataset[0], dataset[0]
        assert all(torch.equal(first[i], second[i]) for i in (0, 1))
        assert first[2] == second[2]
    print('PASS: training augmentation varies; RGB and gradient stay aligned; validation and test are deterministic.')

    tree = ast.parse(cells[6])
    definitions = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'models', 'exec'), scope)
    scope['device'] = torch.device('cpu')
    model = scope['DualConvNet'](classes_num=100, dropout_p=0.5)
    scope['net'] = model
    scope['criterion'] = torch.nn.CrossEntropyLoss()
    scope['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    first_weight = next(model.parameters()).detach().clone()
    loader = torch.utils.data.DataLoader
    subset = torch.utils.data.Subset
    scope['train_loader'] = loader(subset(scope['train_dataset'], [0, 1, 450, 451]), batch_size=2)
    scope['val_loader'] = loader(subset(scope['val_dataset'], [0, 1, 50]), batch_size=2)

    tree = ast.parse(cells[8])
    definitions = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'training_definitions', 'exec'), scope)
    validation = scope['validate']
    loss, score = validation(scope['val_loader'], model, scope['criterion'])
    rgb, gradient, labels = next(iter(loader(scope['val_loader'].dataset, batch_size=3)))
    with torch.no_grad():
        output = model(rgb, gradient)
        expected_loss = scope['criterion'](output, labels).item()
        expected_score = scope['accuracy'](output, labels).item()
    assert math.isclose(loss, expected_loss, rel_tol=1e-5)
    assert math.isclose(score, expected_score, abs_tol=1e-5)
    history = []

    def record_validation(*args):
        result = validation(*args)
        history.append(result[1])
        return result

    scope['validate'] = record_validation
    with tempfile.TemporaryDirectory(prefix='verified-training-', dir=project / '.ipynb_checkpoints') as temporary:
        # Execute the actual epoch loop and final best-model loading, on tiny loaders.
        statements = [node for node in tree.body if not isinstance(
            node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
        for node in statements:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'config' for target in node.targets):
                node.value = ast.parse(repr({'save_root': temporary, 'epochs': 2}), mode='eval').body
        code = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
        exec(compile(code, 'step4_tiny_epochs', 'exec'), scope)
        best_path = Path(temporary) / 'model_best.pth.tar'
        best = torch.load(best_path, map_location='cpu', weights_only=True)
        last = torch.load(Path(temporary) / 'epoch_2.pth.tar', map_location='cpu', weights_only=True)
        assert best['epoch'] == history.index(max(history)) + 1
        assert best['prec@1'] == max(history)
        assert last['epoch'] == 2 and last['optimizer']['state']
        assert not torch.equal(first_weight, next(iter(last['net'].values())))
        assert all(torch.equal(value.cpu(), best['net'][name]) for name, value in model.state_dict().items())
        assert not model.training
        assert best['split'] == scope['split_info']
        assert best['input_processing']['train_augmentation'] == str(scope['transform_train'])
        assert not list(Path(temporary).glob('*.tmp'))
        # A later non-best epoch must not replace the selected best model.
        saved_bytes = best_path.read_bytes()
        scope['save_checkpoint']({'epoch': 3}, False, temporary, 3)
        assert best_path.read_bytes() == saved_bytes
        print('PASS: real backward updates, weighted validation, checkpoint round-trip and best selection/loading.')

        # Step 5 must reload the checkpoint and use dataset-consistent preprocessing.
        with torch.no_grad():
            next(model.parameters()).add_(1)
        exec(compile(cells[10], 'step5', 'exec'), scope)
        assert all(torch.equal(value.cpu(), best['net'][name]) for name, value in model.state_dict().items())
        assert scope['out'].shape == (1, 100) and not scope['out'].requires_grad
        dataset = scope['test_dataset']
        image = scope['Image'].open(project / 'cifar100/test/apple_9904.jpg').convert('RGB')
        raw = scope['transform_test'](image)
        assert torch.allclose(scope['img'].squeeze(0), dataset.normalize_rgb(raw))
        assert torch.allclose(scope['imgGradient'].squeeze(0), dataset.RGB2Gradient(raw))
        assert not model.training
        print('PASS: Step 5 reloads best weights, matches dataset RGB/gradient processing, and predicts without gradients.')

        # Check all five test images, including the last batch with only one image.
        scope['test_loader'] = loader(subset(dataset, [0, 1, 100, 101, 200]), batch_size=2)
        rgb, gradient, labels = next(iter(loader(scope['test_loader'].dataset, batch_size=5)))
        with torch.no_grad():
            output = model(rgb, gradient)
            expected_loss = scope['criterion'](output, labels).item()
            expected_score = scope['accuracy'](output, labels).item()
            next(model.parameters()).add_(1)
        model.train()
        batches = []
        handle = model.register_forward_pre_hook(lambda module, inputs: batches.append(
            (inputs[0].shape[0], torch.is_grad_enabled(), module.training)))
        report = io.StringIO()
        try:
            with redirect_stdout(report):
                exec(compile(cells[12], 'step6', 'exec'), scope)
        finally:
            handle.remove()
        print(report.getvalue(), end='')
        assert batches == [(2, False, False), (2, False, False), (1, False, False)]
        assert math.isclose(scope['test_top1'], expected_score, abs_tol=1e-5)
        loss_text, score_text = report.getvalue().split('Loss: ', 1)[1].split(', Prec@1: ')
        assert math.isclose(float(loss_text), expected_loss, abs_tol=5.1e-5)
        assert math.isclose(float(score_text.split('%')[0]), expected_score, abs_tol=0.0051)
        assert all(torch.equal(value.cpu(), best['net'][name]) for name, value in model.state_dict().items())
        assert not model.training
        print('PASS: Step 6 reloads best weights, evaluates both inputs without gradients, includes the last batch, and weights metrics by sample count.')


if __name__ == '__main__':
    main()
