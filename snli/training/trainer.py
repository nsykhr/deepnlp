import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from training.utils import get_adamw_optimizer

from typing import Callable, Tuple, Union


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 model_path: str,
                 optimizer_path: str,
                 lr: float = 1e-3,
                 device: Union[str, torch.device] = 'cpu'):
        model.to(device)
        self.model = model
        self.criterion = criterion
        self.optimizer = get_adamw_optimizer(model, lr=lr)
        self.device = device

        if not model_path.endswith('.pt'):
            model_path += '.pt'
        self.model_path = model_path
        if not optimizer_path.endswith('.pt'):
            optimizer_path += '.pt'
        self.optimizer_path = optimizer_path

        self.train_losses = []
        self.train_epoch_losses = []
        self.valid_epoch_losses = []
        self.train_epoch_metrics = []
        self.valid_epoch_metrics = []

    def parse_batch(self, batch: tuple) -> tuple:
        x = batch[:-1]
        y = batch[-1]

        x = tuple(map(lambda x: x.to(self.device), x))
        y = y.to(self.device)

        return x, y

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              last_n_losses: int = 500,
              num_epochs: int = 100,
              clip: float = 3.,
              verbose: bool = True) -> nn.Module:
        best_valid_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            self.model.train()

            correct, total = 0, 0
            train_epoch_losses = []

            progress_bar = tqdm(total=len(train_loader), disable=not verbose,
                                position=0, leave=True, desc='Current progress: ')

            for batch in train_loader:
                x, y = self.parse_batch(batch)
                logits = self.model(*x)

                loss = self.criterion(logits, y)
                self.train_losses.append(loss.item())
                train_epoch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()

                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                correct += (preds == y).long().sum().item()
                total += len(y)

                progress_bar.set_description(
                    f'Mean loss: {np.mean(self.train_losses[-last_n_losses:])}. Current progress: ')
                progress_bar.update()

            progress_bar.close()

            train_epoch_loss = np.mean(train_epoch_losses)
            self.train_epoch_losses.append(train_epoch_loss)
            self.train_epoch_metrics.append(correct / total)

            valid_epoch_loss, valid_epoch_accuracy = self.evaluate(valid_loader)
            self.valid_epoch_losses.append(valid_epoch_loss)
            self.valid_epoch_metrics.append(valid_epoch_accuracy)

            print(f'Mean training loss: {train_epoch_loss}. Mean validation loss: {valid_epoch_loss}.\n'
                  f'Training accuracy: {self.train_epoch_metrics[-1]}. '
                  f'Validation accuracy: {self.valid_epoch_metrics[-1]}.')

            if self.valid_epoch_metrics[-1] - max(self.valid_epoch_metrics) < -0.001 or \
                    valid_epoch_loss / best_valid_loss > 1.05:
                print('Validation performance has started degrading. Performing early stopping.')
                break

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                torch.save(self.model.state_dict(), self.model_path.replace('.pt', '_best.pt'))
                torch.save(self.optimizer.state_dict(), self.optimizer_path.replace('.pt', '_best.pt'))

        torch.save(self.model.state_dict(), self.model_path.replace('.pt', '_last.pt'))
        torch.save(self.optimizer.state_dict(), self.optimizer_path.replace('.pt', '_last.pt'))

        return self.model

    def evaluate(self, valid_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()

        losses = []
        correct, total = 0, 0

        for batch in valid_loader:
            with torch.no_grad():
                x, y = self.parse_batch(batch)
                logits = self.model(*x)
                losses.append(self.criterion(logits, y).item())

                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                correct += (preds == y).long().sum().item()
                total += len(y)

        accuracy = correct / total

        return np.mean(losses).item(), accuracy
