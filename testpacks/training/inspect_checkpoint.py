"""Inspect this run's own checkpoint on CPU; never downloads model weights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import sha256, write_json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipes import recipe, model


def inspect(path, epochs, seed, recipe_name='resnet50_imagenet'):
    import torch
    selected = recipe(recipe_name)

    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('epoch') != epochs or checkpoint.get('arch') != selected['architecture']:
        raise ValueError('Unexpected checkpoint epoch/architecture')
    state = checkpoint['state_dict']
    if not state or any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError('Missing or nonfinite model state')
    optimizer = checkpoint.get('optimizer', {}).get('state', {})
    if not optimizer:
        raise ValueError('No SGD momentum state: no completed optimizer updates evidenced')
    for entry in optimizer.values():
        for value in entry.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise ValueError('Nonfinite optimizer state')
    torch.manual_seed(seed)
    network = model(selected['architecture'])
    key = selected['changed_parameter']
    changed = not torch.equal(network.state_dict()[key], state[key].float())
    if not changed:
        raise ValueError('Checkpoint first convolution equals seeded initialization')
    network.load_state_dict(state, strict=True)
    return dict(complete=True, epoch=epochs, architecture=selected['architecture'], map_location='cpu',
                finite_model_state=True, finite_optimizer_state=True,
                parameter_changed_from_seeded_initialization=True,
                checkpoint_sha256=sha256(path), checkpoint_bytes=Path(path).stat().st_size,
                model_tensor_count=len(state), optimizer_state_count=len(optimizer),
                correctness='not_checked', scope='structural training/checkpoint checks, not independent numerical accuracy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recipe', default='resnet50_imagenet')
    args = parser.parse_args()
    write_json(args.output, inspect(args.checkpoint, args.epochs, args.seed, args.recipe))
