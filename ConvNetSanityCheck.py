from src.resnet import resnet50_cifar, resnet18_cifar, resnet34_cifar
from torch.optim import SGD, lr_scheduler, Adam
from src.data_setup import get_dataloaders
from src.resnetTrainer import Trainer

class Exp1:
    def __init__(self,):
        pass
    def load_data(self, data_fraction: float = 1.0):
        self.train_loader, self.val_loader, self.num_classes = get_dataloaders(
            data_fraction= data_fraction,   # Experiment 1: 0.05 / 0.1 / 0.25 / 0.5 / 1.0
            img_size= 32,
            batch_size= 128)
        

    def train_with_different_data_fractions(self, num_epochs:int = 200, data_fraction: float = 1.0):
        self.load_data(data_fraction=data_fraction)
        model = resnet50_cifar(num_classes=self.num_classes,
            base_channels=64,
            dropout=0.1)
        optimizer = SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

        trainer = Trainer(model=model, train_loader=self.train_loader, val_loader=self.val_loader,
                          optimizer=optimizer, scheduler=scheduler)
        num_trainable_params = sum([p.numel() for p in model.parameters()])
        print('\n' + 'num_trainable_params = ' + str(num_trainable_params) + '\n')
        # trainer.fit(epochs=num_epochs)

if __name__ == "__main__":
    exp_obj = Exp1()
    exp_obj.train_with_different_data_fractions(data_fraction=0.05)

    
    