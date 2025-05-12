import torch
import sys
import numpy as np
import IPython.display
import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
# from pl_bolts.optimizers.lars import LARS
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
import torch.nn.functional as F
# from torch.optim.lr_scheduler import SequentialLR
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR


class AutoencoderTrainer:
    def __init__(self, autoencoder, optimizer, loss_fn, device, transform=None, normalize=None):
        self.autoencoder = autoencoder
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.transform = transform
        self.normalize = normalize
        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # ])
        # self.s = 0.5
        # self.transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
        #                                       transforms.RandomResizedCrop(32,(0.8,1.0)),
        #                                       transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s, 
        #                                                                                                          0.8*self.s, 
        #                                                                                                          0.8*self.s, 
        #                                                                                                          0.2*self.s)], p = 0.8),
        #                                                           transforms.RandomGrayscale(p=0.2)
        #                                                         ]),
        #                                         transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
        #                                       ]
        #                                      )
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
        #         #p=0.8
        #         p=0.2
        #     ),
        #     # transforms.RandomGrayscale(p=0.1),
        #     # transforms.GaussianBlur(kernel_size=3),
        #     transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
        # ])

    def train_batch(self, batch):
        self.autoencoder.train()
        self.optimizer.zero_grad()
        
        # Create augmentation of the batch
        if self.transform is not None:        
            with torch.no_grad():
                batch = torch.stack([self.transform(img) for img in batch])
                if self.normalize is not None:
                    batch = self.normalize(batch)
            batch = batch.to(self.device)
        
        # Compare with original batch maybe
        rec_x = self.autoencoder(batch)
        loss = self.loss_fn(rec_x, batch)
        
        loss.backward()
        self.optimizer.step()
        return loss
    
    def test_batch(self, batch):
        self.autoencoder.eval()
        with torch.no_grad():
            if self.normalize is not None:
                batch = self.normalize(batch)
            rec_x = self.autoencoder(batch)
            loss = self.loss_fn(rec_x, batch)
        return loss

    def run_batches(self, dl, dl_fn):
        losses = []
        with tqdm.tqdm(total=len(dl), file=sys.stdout) as pbar:
            for (x_data, _) in dl:
                x_data = x_data.to(self.device)
                loss = dl_fn(batch=x_data)
                losses.append(loss.cpu().item())
                pbar.update()
        return np.mean(losses)
        
    
    # def show_sample(self, dl, title):
    #     with torch.no_grad():
    #         batch_size = next(iter(dl))[0].size()[0]
    #         indices = np.random.choice(batch_size, size=5, replace=False)
    #         sample = next(iter(dl))[0][indices].to(self.device)
    #         reconstructed = self.decoder(self.encoder(sample)).cpu()
        
    #     fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    #     for i in range(5):
    #         axes[0, i].imshow(sample[i].cpu().squeeze(), cmap="gray")
    #         axes[0, i].axis("off")
    #         axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
    #         axes[1, i].axis("off")

    #     plt.suptitle(title)
    #     IPython.display.display(fig)
    #     plt.close(fig)
    
    def show_sample(self, dl, title): 
        self.autoencoder.eval()
        with torch.no_grad():
            batch_size = next(iter(dl))[0].size()[0]
            indices = np.random.choice(batch_size, size=5, replace=False)
            sample = next(iter(dl))[0][indices].to(self.device)
            reconstructed = self.autoencoder(sample).cpu()
        
        
        if sample[0].dim() == 2:  # Grayscale image
            cmap = 'gray'
        else:  # RGB image
            cmap = None
            
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axes[0, i].imshow(sample[i].cpu().permute(1, 2, 0), cmap=cmap)
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0), cmap=cmap)
            axes[1, i].axis("off")

        axes[0, 0].set_title('Original', fontsize=10, loc='left')
        axes[1, 0].set_title('Reconstructed', fontsize=10, loc='left')
        plt.suptitle(title)
        IPython.display.display(fig)
        plt.close(fig)
        
    
    def train(self, num_epochs, dl_train, dl_val):
        try:
            train_avg_losses = []
            val_avg_losses = []

            best_val_loss = float('inf')
            
            for epoch_idx in range(num_epochs):
                print(f'--- EPOCH {epoch_idx+1}/{num_epochs} ---')
                train_loss = self.run_batches(dl=dl_train, dl_fn=self.train_batch)
                train_avg_losses.append(train_loss)
                print(f'Train loss: {train_loss}')
                self.show_sample(dl=dl_train, title=f'Training Reconstruction - Epoch {epoch_idx+1}')
        
                val_loss = self.run_batches(dl=dl_val, dl_fn=self.test_batch)
                val_avg_losses.append(val_loss)
                print(f'Validation loss: {val_loss}')
                self.show_sample(dl=dl_val, title=f'Validation Reconstruction - Epoch {epoch_idx+1}')
                
                # Save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.autoencoder.state_dict()
                    print(f'Best model saved at epoch {epoch_idx+1} with validation loss: {val_loss}')
        
            # Load the best model
            self.autoencoder.load_state_dict(best_model_state)
            print('Best model loaded.')
                
            # Plot graph
            fig, ax = plt.subplots()
            ax.plot(range(num_epochs), train_avg_losses, 'g-', label='Train Loss')
            ax.plot(range(num_epochs), val_avg_losses, 'r-', label='Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title('Reconstruction Loss')
            fig.legend(loc="upper right")
            ax.set_ylim(bottom=0)
            plt.show()
        
        except KeyboardInterrupt:
            print('\n *** Training interrupted by user')
            
            
    def test(self, dl_test):
        try:
            loss = self.run_batches(dl=dl_test, dl_fn=self.test_batch)
            print(f'Test loss: {loss}')
            self.show_sample(dl=dl_test, title=f'Test Reconstruction')
        
        except KeyboardInterrupt:
            print('\n *** Testing interrupted by user')


















class ClassifierTrainer:
    def __init__(self, model, optimizer, loss_fn, device, transform=None, normalize=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.transform = transform
        self.normalize = normalize

    def train_batch(self, batch, labels):
        # Add augmentation if needed
        if self.transform is not None:
            with torch.no_grad():
                batch = torch.stack([self.transform(img) for img in batch])
                if self.normalize is not None:
                    batch = self.normalize(batch)
            batch = batch.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        y_prob = self.model(batch)
        loss = self.loss_fn(y_prob, labels)
        loss.backward()
        self.optimizer.step()
        y_pred = torch.argmax(y_prob, dim=-1)
        accuracy = torch.mean((y_pred == labels).to(torch.float32))
        return loss.item(), 100*accuracy.item()
    
    def test_batch(self, batch, labels):
        self.model.eval()
        with torch.no_grad():
            if self.normalize is not None:
                batch = self.normalize(batch)
            y_prob = self.model(batch)
            loss = self.loss_fn(y_prob, labels)
            y_pred = torch.argmax(y_prob, dim=-1)
            accuracy = torch.mean((y_pred == labels).to(torch.float32))
        return loss.item(), 100*accuracy.item()
    
    def run_batches(self, dl, dl_fn):
        losses = []
        accuracies = []
        with tqdm.tqdm(total=len(dl), file=sys.stdout) as pbar:
            for (x_data, x_labels) in dl:
                x_data = x_data.to(self.device)
                x_labels = x_labels.to(self.device)
                loss, accuracy = dl_fn(batch=x_data, labels=x_labels)
                losses.append(loss)
                accuracies.append(accuracy)
                pbar.update()
        return np.mean(losses), np.mean(accuracies)
    
    def train(self, num_epochs, dl_train, dl_val):
        try:
            train_avg_losses = []
            val_avg_losses = []
            train_avg_accuracies = []
            val_avg_accuracies = []
            
            best_val_loss = float('inf')
            
            for epoch_idx in range(num_epochs):
                print(f'--- EPOCH {epoch_idx+1}/{num_epochs} ---')
                train_loss, train_accuracy = self.run_batches(dl=dl_train, dl_fn=self.train_batch)
                train_avg_losses.append(train_loss)
                train_avg_accuracies.append(train_accuracy)
                print(f'Train loss: {train_loss}, accuracy: {train_accuracy}')
        
                val_loss, val_accuracy = self.run_batches(dl=dl_val, dl_fn=self.test_batch)
                val_avg_losses.append(val_loss)
                val_avg_accuracies.append(val_accuracy)
                print(f'Validation loss: {val_loss}, accuracy: {val_accuracy}')
                
                # Save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    print(f'Best model saved at epoch {epoch_idx+1} with validation loss: {val_loss}')
        
            # Load the best model
            self.model.load_state_dict(best_model_state)
            print('Best model loaded.')
                
            # Plot graphs
            fig, ax = plt.subplots()
            ax.plot(range(num_epochs), train_avg_losses, 'g-', label='Train Loss')
            ax.plot(range(num_epochs), val_avg_losses, 'r-', label='Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            fig.legend(loc="upper right")
            ax.set_ylim(bottom=0)
            ax.set_title('Classification Loss')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(range(num_epochs), train_avg_accuracies, 'g-', label='Train Accuracy')
            ax.plot(range(num_epochs), val_avg_accuracies, 'r-', label='Validation Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            fig.legend(loc="upper right")
            ax.set_ylim(bottom=0)
            ax.set_title('Classification Accuracy')
            plt.show()
        
        except KeyboardInterrupt:
            print('\n *** Training interrupted by user')
            
    def test(self, dl_test):
        try:
            loss, accuracy = self.run_batches(dl=dl_test, dl_fn=self.test_batch)
            print(f'Test loss: {loss}, accuracy: {accuracy}')
            if accuracy >= 60 and accuracy < 65:
                print('WOW!')
            elif accuracy >= 65 and accuracy < 70:
                print('WOW! :)')
            elif accuracy >= 70:
                print('Amazing!!!')
        
        except KeyboardInterrupt:
            print('\n *** Testing interrupted by user')
            
            
            
            
            
            
            


class ContrastiveTrainer:
    def __init__(self, model, device, num_epochs, optimizer, transform, normalize=None, scheduler=None, temperature=0.5):
        self.model = model
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=6e-2,
        #     momentum=0.9,
        #     weight_decay=5e-4
        # )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, num_epochs)
        
        #############################################################################################
        # warmup_epochs = 10
        # total_epochs = num_epochs

        # # 1. Warm-up scheduler: linearly scale LR from 10% to 100%
        # warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)

        # # 2. Cosine Annealing for the rest
        # cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5)

        # # 3. Combine them
        # self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        #############################################################################################
        
        self.loss_fn = nt_xent_loss
        self.temperature = temperature
        self.device = device
        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.1),
        #     transforms.RandomResizedCrop(size=input_size, scale=(0.7, 1.0)),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomGrayscale(p=0.1),
        #     transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
        # ])
        self.transform = transform
        self.normalize = normalize
        
            
        
    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create 2 augmentations of the batch
        with torch.no_grad():
            aug1 =  torch.stack([self.transform(img) for img in batch])
            aug2 =  torch.stack([self.transform(img) for img in batch])
            if self.normalize is not None:
                aug1 = self.normalize(aug1)
                aug2 = self.normalize(aug2)
        
        aug1 = aug1.to(self.device)
        aug2 = aug2.to(self.device)
        
        out1 = self.model(aug1)
        out2 = self.model(aug2)
        
        out1 = F.normalize(out1, dim=1)
        out2 = F.normalize(out2, dim=1)
        
        loss = self.loss_fn(out1, out2, temperature=self.temperature)
        loss.backward()

        self.optimizer.step()
        
        return loss.item()
    
    def test_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            # Create 2 augmentations of the batch
            aug1 =  torch.stack([self.transform(img) for img in batch])
            aug2 =  torch.stack([self.transform(img) for img in batch])
            if self.normalize is not None:
                aug1 = self.normalize(aug1)
                aug2 = self.normalize(aug2)
            
            aug1 = aug1.to(self.device)
            aug2 = aug2.to(self.device)
            
            out1 = self.model(aug1)
            out2 = self.model(aug2)
            
            out1 = F.normalize(out1, dim=1)
            out2 = F.normalize(out2, dim=1)
            
            loss = self.loss_fn(out1, out2, temperature=self.temperature)
        return loss.item()
    
    def run_batches(self, dl, dl_fn):
        losses = []
        with tqdm.tqdm(total=len(dl), file=sys.stdout) as pbar:
            for (x_data, _) in dl:
                # x_data = x_data.to(self.device)
                loss = dl_fn(batch=x_data)
                losses.append(loss)
                pbar.update()
        return np.mean(losses)
    
    def train(self, dl_train, dl_val):
        try:
            train_avg_losses = []
            val_avg_losses = []
            
            best_val_loss = float('inf')
            
            for epoch_idx in range(self.num_epochs):
                print(f'--- EPOCH {epoch_idx+1}/{self.num_epochs} ---')
                train_loss = self.run_batches(dl=dl_train, dl_fn=self.train_batch)
                train_avg_losses.append(train_loss)
                print(f'Train loss: {train_loss}')
        
                val_loss = self.run_batches(dl=dl_val, dl_fn=self.test_batch)
                val_avg_losses.append(val_loss)
                print(f'Validation loss: {val_loss}')
                
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}')
                
                # Save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    print(f'Best model saved at epoch {epoch_idx+1} with validation loss: {val_loss}')
        
            # Load the best model
            self.model.load_state_dict(best_model_state)
            print('Best model loaded.')
                
            # Plot graph
            fig, ax = plt.subplots()
            ax.plot(range(self.num_epochs), train_avg_losses, 'g-', label='Train Loss')
            ax.plot(range(self.num_epochs), val_avg_losses, 'r-', label='Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            # ax.set_title('Contrastive Loss')
            fig.legend(loc="upper right")
            ax.set_ylim(bottom=0)
            plt.show()
        
        except KeyboardInterrupt:
            print('\n *** Training interrupted by user')
            
