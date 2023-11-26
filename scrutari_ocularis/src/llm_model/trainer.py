from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import os
import torch
import h5py
from llm_model.scrutari_ocularis_model import ScrutariOcularisModel

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
   
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)           
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Adaptar la extracción de data y target para tu dataset
            data, target = data.to(device), target.to(device) 
            
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_images_from_hdf5(filename, transform=None):
    data_list = []
    
    with h5py.File(filename, 'r') as hf:
        for group_name in hf.keys():
            group = hf[group_name]
            images_np = group["image"][()]  # Esto es un array de imágenes
            
            # Extraer el número de los atributos del grupo
            num = group.attrs["number"]
            
            images_np = torch.tensor(images_np).permute(2, 0, 1)

            # Iterar sobre cada imagen en images_np y agregarla a data_list
            for image_np in images_np:
                # Convertir el ndarray a una imagen PIL
                #pil_image = Image.fromarray(image_np.astype(np.uint8).squeeze())
                
                # Aplicar la transformación si se proporciona
                if transform:
                    image_np = transform(image_np)
                
                image_tensor = image_np.unsqueeze(0)

                data_object = {
                    "label": num,
                    "image": image_tensor
                }
                data_list.append(data_object)

    return data_list

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_filename, transform=None):
        self.data = load_images_from_hdf5(hdf5_filename, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["image"], self.data[idx]["label"]
    
def train_model(args):
    print("start CustomModel")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    elif use_mps:
        device = torch.device("mps")
        print("USING MPS")
    else:
        device = torch.device("cpu")
        print("USING CPU")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.Pad((70, 70)),
        #transforms.Resize((224, 224)),
        #transforms.Grayscale(num_output_channels=3),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5,))
    ])

    """ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) """
    print("start CustomModel")

    train_dataset = CustomDataset('train_data_s.h5', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    test_dataset = CustomDataset('test_data_s.h5', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print("End Load Dataset")

    model = ScrutariOcularisModel().to(device) #NumericClassifier(num_classes=999).to(device)

    name_file_model = "scrutari_ocularis_model_v_1.pt"

    if os.path.exists(name_file_model):
        model.load_state_dict(torch.load(name_file_model))
        print("Save model")
    else:
        print("No se encontró un modelo previo. Entrenando desde cero.")
        
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), name_file_model)
        print("Save model")

def setup_args():
    parser = argparse.ArgumentParser(description='PyTorch Custom Model OCR')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    return parser.parse_args()

if __name__ == '__main__':
    args = setup_args()
    train_model(args)