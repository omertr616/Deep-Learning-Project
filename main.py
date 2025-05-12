import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
import data_utils
import torch.nn as nn
import training

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    parser.add_argument('--load-models', action='store_true', default=False,help='Whether to load contrastive encdoder models.')
    return parser.parse_args()
    

if __name__ == "__main__":

    args = get_args()
    freeze_seeds(args.seed)
                
                                           
    if args.mnist:
        ds_train = datasets.MNIST(root=args.data_path, train=True, download=False, transform=None)
        ds_test = datasets.MNIST(root=args.data_path, train=False, download=False, transform=None)
    else:
        ds_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=None)
        ds_test = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=None)
        
    
    device = args.device
    
    if args.mnist:
        ###########################################################################################
        ########################################## MNIST ##########################################
        ###########################################################################################
        
        
        ########################################## 1.2.1 ##########################################
        if args.self_supervised:
            print('#'*50 + ' 1.2.1 ' + '#'*50)
        
            dl_train, dl_val, dl_test = data_utils.get_data_loaders(train_dataset=ds_train,
                                                                    test_dataset=ds_test,
                                                                    batch_size=64)
            from mnist import encoder, decoder

            encoder = encoder.encoder_mnist().to(device)
            decoder = decoder.decoder_mnist().to(device)
            autoencoder = nn.Sequential(encoder, decoder).to(device)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4)
            
                        
            
            encoder_decoder_trainer = training.AutoencoderTrainer(autoencoder=autoencoder,
                                                                                    optimizer=optimizer,
                                                                                    loss_fn=loss_fn,
                                                                                    device=device,
                                                                                    transform=None)

                
            encoder_decoder_trainer.train(num_epochs=15,
                                            dl_train=dl_train,
                                            dl_val=dl_val)
            
            encoder_decoder_trainer.test(dl_test=dl_test)


                
                
                
            from utils import plot_tsne
            plot_tsne(encoder, dl_test, device=device, name='reconstruction_mnist')
            
            
            
            from mnist import classifier

            classifier = classifier.ClassifierMnist()
            model = nn.Sequential(encoder, classifier).to(device)
            
            
            
            for param in list(encoder.parameters()):
                param.requires_grad = False
            for param in list(classifier.parameters()):
                param.requires_grad = True
            
            
            
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            loss_fn = nn.CrossEntropyLoss()
            trainer = training.ClassifierTrainer(model=model,
                                                optimizer=optimizer,
                                                loss_fn=loss_fn,
                                                device=device)

            
            trainer.train(num_epochs=20, dl_train=dl_train, dl_val=dl_val)

            trainer.test(dl_test)     
            
            
            print('#'*107)    
            print('\n'*10)
            
        ###########################################################################################
                
        
        
        
        
        
        
        
        
        ########################################## 1.2.2 ##########################################
        else:
            print('#'*50 + ' 1.2.2 ' + '#'*50)
            
            dl_train, dl_val, dl_test = data_utils.get_data_loaders(train_dataset=ds_train,
                                                                    test_dataset=ds_test,
                                                                    batch_size=64)
            
            
            
            from mnist import encoder, classifier

            encoder = encoder.encoder_mnist()
            classifier = classifier.ClassifierMnist()
            model = nn.Sequential(encoder, classifier).to(device)
            
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = nn.CrossEntropyLoss()

            trainer = training.ClassifierTrainer(model=model,
                                                optimizer=optimizer,
                                                loss_fn=loss_fn,
                                                device=device)

                
            trainer.train(num_epochs=10, dl_train=dl_train, dl_val=dl_val)

            trainer.test(dl_test)
                
            
            
            from utils import plot_tsne
            plot_tsne(model=encoder, dataloader=dl_test, device=device, name='supervised_mnist')
           
            
            print('#'*107)    
            print('\n'*10)
            
        ###########################################################################################
                
        
        
        
        
        
        
        
        
        ########################################## 1.2.3 ##########################################
        
        print('#'*50 + ' 1.2.3 ' + '#'*50)
        
        dl_train, dl_val, dl_test = data_utils.get_data_loaders(train_dataset=ds_train,
                                                                test_dataset=ds_test,
                                                                batch_size=48)
        
        
        
        from mnist import encoder

        encoder = encoder.encoder_mnist()

        projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        model = nn.Sequential(
            encoder,
            projection_head
        ).to(device)
        
        
        if args.load_models:
            encoder.load_state_dict(torch.load('saved_models/encoder_contrastive_mnist.pth'))
        else:
                transform = transforms.Compose([
                        transforms.RandomResizedCrop(28, scale=(0.5, 1.0)),
                        transforms.RandomApply([
                                transforms.GaussianBlur(kernel_size=3)
                                ], p=0.5),
                        ]) 
                
                num_epochs = 100
                # optimizer = torch.optim.SGD(
                #         model.parameters(),
                #         lr=6e-2,
                #         momentum=0.9,
                #         weight_decay=5e-4
                # )
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                contrastive_trainer = training.ContrastiveTrainer(model=model,
                                                                device=device,
                                                                num_epochs=num_epochs,
                                                                optimizer=optimizer,
                                                                transform=transform,
                                                                temperature=0.5)
                contrastive_trainer.train(dl_train=dl_train, dl_val=dl_val)
            
            
            
        from utils import plot_tsne
        plot_tsne(model=encoder, dataloader=dl_test, device=device, name='contrastive_mnist')
        
        
        
        from mnist import classifier

        classifier = classifier.ClassifierMnist().to(device)
        model = nn.Sequential(encoder, classifier).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        
        
        for param in list(encoder.parameters()):
            param.requires_grad = False
        for param in list(classifier.parameters()):
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        
        classifier_trainer = training.ClassifierTrainer(model=model,
                                                        optimizer=optimizer,
                                                        loss_fn=loss_fn,
                                                        device=device,
                                                        transform=None)
        
        
        classifier_trainer.train(num_epochs=20,
                                dl_train=dl_train,
                                dl_val=dl_val)
            
        classifier_trainer.test(dl_test=dl_test)
        
        
        
        print('#'*107)    
                
        ###########################################################################################
                
        
        
        
        
        
        
        
        
    else:
        ###########################################################################################
        ########################################## CIFAR ##########################################
        ###########################################################################################
        
        
        
        ########################################## 1.2.1 ##########################################
        if args.self_supervised:
            print('#'*50 + ' 1.2.1 ' + '#'*50)
        
            dl_train, dl_val, dl_test = data_utils.get_data_loaders(train_dataset=ds_train,
                                                                    test_dataset=ds_test,
                                                                    batch_size=64)
            
            
            from cifar10 import encoder, decoder

            encoder = encoder.encoder_cifar().to(device)
            decoder = decoder.decoder_cifar().to(device)
            autoencoder = nn.Sequential(encoder, decoder).to(device)
            
            
            
            loss_fn = nn.MSELoss()
            # loss_fn = nn.L1Loss()
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=5e-4)
            aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                    p=0.2
                )
            ])
            normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                            std=[0.24703223, 0.24348513, 0.26158784])

            encoder_decoder_trainer = training.AutoencoderTrainer(autoencoder=autoencoder,
                                                                    optimizer=optimizer,
                                                                    loss_fn=loss_fn,
                                                                    device=device,
                                                                    transform=aug_transform,
                                                                    normalize=normalize)

                
            encoder_decoder_trainer.train(num_epochs=20,
                                        dl_train=dl_train,
                                        dl_val=dl_val,)

            encoder_decoder_trainer.test(dl_test=dl_test)
            
            
            
            from utils import plot_tsne
            plot_tsne(model=encoder, dataloader=dl_test, device=device, name='reconstruction_cifar10')
            
            
            from cifar10 import classifier

            classifier = classifier.ClassifierCifar().to(device)
            model = nn.Sequential(encoder, classifier).to(device)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            
            
            for param in list(encoder.parameters()):
                param.requires_grad = False
            for param in list(classifier.parameters()):
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            
            
            aug_transform = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
            ])
            normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                            std=[0.24703223, 0.24348513, 0.26158784])
            
            classifier_trainer = training.ClassifierTrainer(model=model,
                                                        optimizer=optimizer,
                                                        loss_fn=loss_fn,
                                                        device=device)

            
            classifier_trainer.train(num_epochs=20,
                                    dl_train=dl_train,
                                    dl_val=dl_val)        

            classifier_trainer.test(dl_test=dl_test)
            
            
            
            
            
            print('#'*107)
            print('\n'*10)

        ###########################################################################################
                
        
        
        
        
        
        
        
               
        ########################################## 1.2.2 ##########################################
        else:
            print('#'*50 + ' 1.2.2 ' + '#'*50)
            
            
            dl_train, dl_val, dl_test = data_utils.get_data_loaders(train_dataset=ds_train,
                                                                    test_dataset=ds_test,
                                                                    batch_size=64)
            
            
            from cifar10 import encoder, classifier

            encoder = encoder.encoder_cifar()
            classifier = classifier.ClassifierCifar()
            model = nn.Sequential(encoder, classifier).to(device)
            
            
            
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            loss_fn = nn.CrossEntropyLoss()
            transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
            normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])

            trainer = training.ClassifierTrainer(model=model,
                                                optimizer=optimizer,
                                                loss_fn=loss_fn,
                                                device=device,
                                                transform=transform,
                                                normalize=normalize)

                
            trainer.train(num_epochs=50, dl_train=dl_train, dl_val=dl_val)

            trainer.test(dl_test)
                  
            
            from utils import plot_tsne
            plot_tsne(model=encoder, dataloader=dl_test, device=device, name='supervised_cifar10')
           
            
            print('#'*107)    
            print('\n'*10)
        
        ###########################################################################################
                
        
        
        
        
        
        
        
        
        ########################################## 1.2.3 ##########################################
        
        print('#'*50 + ' 1.2.3 ' + '#'*50)
            
        
        dl_train, dl_val, dl_test = data_utils.get_data_loaders(train_dataset=ds_train,
                                                                test_dataset=ds_test,
                                                                batch_size=128)
        
        
        from cifar10 import encoder

        encoder = encoder.encoder_cifar()

        projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        model = nn.Sequential(
            encoder,
            projection_head
        ).to(device)
        
        
        

        if args.load_models:
            encoder.load_state_dict(torch.load('saved_models/encoder_contrastive_cifar.pth'))
        else:
            transform = transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                            p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([
                                transforms.GaussianBlur(kernel_size=3)
                            ], p=0.5),
                    ])
            normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
            num_epochs = 200
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
            contrastive_trainer = training.ContrastiveTrainer(model=model,
                                                            device=device,
                                                            num_epochs=num_epochs,
                                                            optimizer=optimizer,
                                                            transform=transform,
                                                            normalize=normalize,
                                                            temperature=0.15)
            contrastive_trainer.train(dl_train=dl_train, dl_val=dl_val)
        
        
        
        from utils import plot_tsne
        plot_tsne(model=encoder, dataloader=dl_test, device=device, name='contrastive_cifar10')
        
        
        from cifar10 import classifier

        classifier = classifier.ClassifierCifar().to(device)
        model = nn.Sequential(encoder, classifier).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for param in list(encoder.parameters()):
            param.requires_grad = False
        for param in list(classifier.parameters()):
            param.requires_grad = True
            
            
        

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], 
                                        std=[0.24703223, 0.24348513, 0.26158784])

        trainer = training.ClassifierTrainer(model=model,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            device=device,
                                            transform=transform,
                                            normalize=normalize)
        

        
        trainer.train(num_epochs=30, dl_train=dl_train, dl_val=dl_val)
        
        trainer.test(dl_test)
                
            
        print('#'*107)    
        
        ###########################################################################################