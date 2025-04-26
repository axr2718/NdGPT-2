import torch
import torchvision
from torchvision.transforms import v2
from models.ndvit import NdVisionTransformer
from models.vit import VisionTransformer
import torch.nn as nn
from experiment.train import train
from experiment.test import test
from utils.parameter_count import parameter_count

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = torchvision.transforms.Compose([v2.Resize((224, 224)),
                                                v2.ToImage(),
                                                v2.ToDtype(torch.uint8, scale=True),
                                                v2.ToDtype(torch.float32, scale=True),
                                                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


    train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                      train=True,
                                                      transform=transform,
                                                      download=False)

    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                     train=False,
                                                     transform=transform,
                                                     download=False)

    num_classes = len(train_dataset.classes)

    model = VisionTransformer(num_classes=num_classes,
                              patch_size=16,
                              embed_dim=384,
                              depth=12,
                              mlp_ratio=4.0,
                              num_heads=6)
    
    num_params = parameter_count(model)

    print(f'Testing ViT-S Patch 16 with {num_params} parameters.')
    model.to(device)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,fused=True)

    trained_model = train(model=model,
                          train_dataset=train_dataset,
                          criterion=criterion,
                          optimizer=optimizer,
                          epochs=20,
                          batch_size=32)
    
    metrics = test(model=model,
                   test_dataset=test_dataset,
                   device=device)
    
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    
    model = NdVisionTransformer(num_classes=num_classes,
                              patch_size=16,
                              embed_dim=384,
                              depth=12,
                              mlp_ratio=0.5,
                              num_heads=6,)
                              #use_ndlinear=True)
    num_params = parameter_count(model)
    print(f'Testing NdViT-S Patch 16 with {num_params} parameters.')
    model.to(device)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,fused=True)

    trained_model = train(model=model,
                          train_dataset=train_dataset,
                          criterion=criterion,
                          optimizer=optimizer,
                          epochs=20,
                          batch_size=32)
    
    metrics = test(model=model,
                   test_dataset=test_dataset,
                   device=device)
    
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')