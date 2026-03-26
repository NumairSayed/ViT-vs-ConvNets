import torch
from torch import nn
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device="cuda",
        grad_clip=1.0,       # ViT needs this — transformers blow up without it
        label_smoothing=0.1, # helps ViT generalise on small datasets like CIFAR-10
        use_amp=True,        # mixed precision — free speedup on any modern GPU
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.grad_clip    = grad_clip

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # AMP scaler — handles float16 overflow automatically
        # set enabled=False on CPU or if use_amp=False so the code path is identical
        self.use_amp = use_amp and device != "cpu"
        self.scaler  = torch.amp.GradScaler(enabled=self.use_amp)

        # history for plotting loss / accuracy curves across experiments
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ---------------------------
    # Train one epoch
    # ---------------------------
    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                outputs = self.model(images)
                # model may return (logits, extras) when return_attention / return_all_layers is True
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            # Unscale before clipping so the threshold is in the real gradient space
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / total, correct / total

    # ---------------------------
    # Validate
    # ---------------------------
    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / total, correct / total

    # ---------------------------
    # Full training loop
    # ---------------------------
    def fit(self, epochs):
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_one_epoch()
            val_loss,   val_acc   = self.validate()

            # step scheduler after validation (works for CosineAnnealingLR etc.)
            if self.scheduler is not None:
                self.scheduler.step()

            # store for later plotting
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}]  "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
            )

        return self.history