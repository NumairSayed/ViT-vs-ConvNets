from src.ViT import vit_tiny
from src.data_setup import get_dataloaders
from tqdm.auto import tqdm
from src.ViTTrainer import Trainer
from torch.optim import SGD, lr_scheduler

num_epochs = 10

train_loader, val_loader, num_classes = get_dataloaders(
            data_fraction= 1.0,   # Experiment 1: 0.05 / 0.1 / 0.25 / 0.5 / 1.0
            img_size= 32,
            batch_size= 256)
model = vit_tiny()
optimizer = SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
trainer = Trainer(model= model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler)

trainer.fit(epochs=num_epochs)
