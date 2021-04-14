# bad-minima

Repo mainly for preliminary experiments.

### Requisites

- PyTorch (pip `torch` or conda `pytorch`) `1.8.1`

### Setup (IMPORTANT)

```bash
python setup.py develop  # to allow absolute import of project files
```

## Examples

```python
import torch
from torchvision import datasets, models
from project import DATASET_DIR, get_weights_path

cifar = datasets.CIFAR10(root=DATASET_DIR, train=True, download=True)
model = models.vgg()
torch.save(model.state_dict(), get_weights_path("vgg_experiment_000"))
```
