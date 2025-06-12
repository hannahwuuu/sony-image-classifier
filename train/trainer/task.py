import os
import argparse
import json
import torch
import shutil
import concurrent.futures
from torch import nn
from glob import glob
from PIL import Image
import concurrent.futures
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from trainer import game_net_v2
from trainer import game_net_v1

TRAIN_PATH = os.environ['AIP_TRAINING_DATA_URI']
VAL_PATH = os.environ['AIP_VALIDATION_DATA_URI']
TEST_PATH = os.environ['AIP_TEST_DATA_URI']
CACHE_DIR = '/tmp/aip_cache'

def gs_to_local_cache_path(gcs_uri):
    local_path = os.path.join(CACHE_DIR, gcs_uri)
    return local_path

def download_to_local_cache(gcs_uri):
    local_path = gs_to_local_cache_path(gcs_uri)

    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy(gcs_uri, local_path)  # Simulating download
    return local_path

def bulk_cache_images(image_paths, max_workers=128):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        cached_paths = list(executor.map(download_to_local_cache, image_paths))
    return cached_paths

def process_jsonl_file(jsonl_path, class_indices):
    image_paths = []
    labels = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                gcs_uri = data.get('imageGcsUri')
                if not gcs_uri:
                    continue
                image_paths.append(gs_to_gcs_path(gcs_uri))
                labels.append(class_indices[data['classificationAnnotation']['displayName']])
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
    return image_paths, labels

class AIPImageDataset(Dataset):
    def __init__(self, image_paths, labels, num_classes):
        self.labels = labels
        self.num_classes = num_classes

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.images = self._preload_images(image_paths)

        assert len(self.images) == len(image_paths), "Mismatch between loaded images and image paths"

    def _load_single_image(self, path):
        try:
            with open(path, 'rb') as f:
                image = Image.open(f).convert('RGB')
                return image.copy()
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def _preload_images(self, image_paths, max_workers=128):
        cached_image_paths = bulk_cache_images(image_paths, max_workers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            imgs = list(executor.map(self._load_single_image, cached_image_paths))
        return imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return self.transform(image), torch.tensor(label, dtype=torch.float)
    

def load_aip_image_dataset(aip_data_uri_pattern, batch_size, class_names, train=True):
    class_indices = {name: i for i, name in enumerate(class_names)}
    jsonl_files = glob(gs_to_gcs_path(aip_data_uri_pattern))

    all_image_paths = []
    all_labels = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(process_jsonl_file, path, class_indices) for path in jsonl_files]

        for future in concurrent.futures.as_completed(futures):
            image_paths, labels = future.result()
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)

    dataset = AIPImageDataset(all_image_paths, all_labels, len(class_names))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=16, pin_memory=True)
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir', default=os.getenv('AIP_MODEL_DIR'), type=str,
        help='a Cloud Storage URI of a directory intended for saving model artifacts')
    parser.add_argument(
        '--checkpoint-dir', default=os.getenv('AIP_CHECKPOINT_DIR'), type=str,
        help='a Cloud Storage URI of a directory intended for saving checkpoints')
    parser.add_argument(
        '--epochs', type=int, default=2)
    parser.add_argument(
        '--lr', type=float, default=1e-3)
    parser.add_argument(
        '--batch-size', type=int, default=32)
    parser.add_argument('--load-data-only', action='store_true', help='load data only')

    parser.add_argument('--optim-type', type=str, default='adam',
                        choices=['sgd', 'sgd_momentum', 'adam'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='Learning rate decay factor')
    parser.add_argument('--l2-reg', type=float, default=0.0001,
                        help='L2 regularization factor')
    parser.add_argument('--model-type', type=str, default='v1')
    
    args = parser.parse_args()

    print(vars(args))

    return args

def makedirs(model_dir):
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return   

def get_optimizer(params, optim_type, lr, momentum, lr_decay, l2_reg):
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=l2_reg)
    elif optim_type == "sgd_momentum":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=l2_reg)
    elif optim_type == "adam":
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=l2_reg)
    else:
        raise ValueError(optim_type)
    return optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

def gs_to_gcs_path(gs_uri):
    return gs_uri.replace("gs://", "/gcs/")

class Trainer(object):

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 scheduler,
                 train_loader,
                 val_loader,
                 test_loader,
                 device,
                 model_name,
                 checkpoint_path):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
    
    def save(self, model_dir):
        model_path = os.path.join(model_dir.replace("gs://", "/gcs/"), 'model.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def fit(self, epochs, is_chief):

        for epoch in range(epochs):

            print(f"Epoch {epoch + 1}/{epochs}")
            self.train()

            if is_chief and epoch % 5 == 0:
                self.evaluate(self.val_loader)
                torch.save(self.model.state_dict(), self.checkpoint_path)
            elif is_chief and epoch == epochs - 1:
                print("Training complete. Testing on test set...")
                self.evaluate(self.test_loader)
                torch.save(self.model.state_dict(), self.checkpoint_path)

    def train(self):
    
        self.model.train()
        size = len(self.train_loader.dataset)

        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(X).squeeze()
            loss = self.loss_fn(out, y)

            loss.backward()
            self.optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        total_loss, total_correct = 0.0, 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            out = self.model(X).squeeze()
            total_loss += self.loss_fn(out, y).item() * X.size(0)
            total_correct += ((torch.sigmoid(out) > 0.5) == y).sum().item()

        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = total_correct / len(dataloader.dataset)
        print(f"Test Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}")

def main():

    args = parse_args()

    local_model_dir = './tmp/model'
    local_checkpoint_dir = './tmp/checkpoint'

    model_dir = args.model_dir or local_model_dir
    checkpoint_dir = args.checkpoint_dir or local_checkpoint_dir

    gs_prefix = "gs://"

    if model_dir and model_dir.startswith(gs_prefix):
        model_dir = gs_to_gcs_path(model_dir)
    if checkpoint_dir and checkpoint_dir.startswith(gs_prefix):
        checkpoint_dir = gs_to_gcs_path(checkpoint_dir)
    
    makedirs(checkpoint_dir)
    print(f'Checkpoints will be saved to {checkpoint_dir}')
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    print(f'checkpoint_path is {checkpoint_path}')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.model_type == 'v1':
        model = game_net_v1.GameNetV1().to(device)
    elif args.model_type == 'v2':
        model = game_net_v2.GameNetV2().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Initial chief checkpoint is saved to {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path))
    print(f'Initial chief checkpoint is loaded from {checkpoint_path}')


    loss_fn = nn.BCEWithLogitsLoss()
    optimizer, scheduler = get_optimizer(model.parameters(), 
                            args.optim_type, 
                            args.lr, 
                            args.momentum, 
                            args.lr_decay, 
                            args.l2_reg)
    
    class_names = {'negative': 0, 'positive': 1}

    print(f"Loading training data from {TRAIN_PATH}")
    train_loader = load_aip_image_dataset(TRAIN_PATH, args.batch_size, class_names, train=True)
    print(f"Loading training data from {VAL_PATH}")
    val_loader = load_aip_image_dataset(VAL_PATH, args.batch_size, class_names, train=True)
    print(f"Loading training data from {TEST_PATH}")
    test_loader = load_aip_image_dataset(TEST_PATH, args.batch_size, class_names, train=False)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        model_name='game_net_v1.pt',
        checkpoint_path=checkpoint_path
    )

    trainer.fit(args.epochs, is_chief=True)

    
    makedirs(model_dir)
    trainer.save(model_dir)
    print(f"Model saved to {model_dir}")
    

    return 

if __name__ == "__main__":
    main()