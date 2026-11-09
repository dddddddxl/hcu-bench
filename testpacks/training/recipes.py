"""Reviewed classification recipes; add models here without changing the runner lifecycle."""
RECIPES = {
    'resnet50_imagenet': {'architecture': 'resnet50', 'classes': 1000,
                          'changed_parameter': 'conv1.weight'},
}


def recipe(name):
    if name not in RECIPES:
        raise ValueError('Unknown training recipe: ' + name)
    return RECIPES[name]


def model(architecture):
    from torchvision import models
    if architecture not in {r['architecture'] for r in RECIPES.values()}:
        raise ValueError('Unregistered training architecture: ' + architecture)
    return getattr(models, architecture)(weights=None)
