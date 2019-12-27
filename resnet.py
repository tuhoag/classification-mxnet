import argparse
import tqdm
import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.model_zoo.vision as mvision
import mxnet.gluon.data as mdata

num_samples = 10
width = 67
height = 67
SAMPLES = mx.nd.uniform(low=0, high=1, shape=(num_samples, 1, width, height))
LABELS = mx.nd.uniform(low=0, high=1, shape=(num_samples, 1))

def get_model(num_classes=1):
    model = mvision.resnet18_v2(classes=num_classes)
    return model

def load_train_data():
    validation_ratio = 0.3
    num_validation_samples = int(num_samples * validation_ratio)
    num_train_samples = num_samples - num_validation_samples

    samples = SAMPLES[0:num_train_samples]
    labels = LABELS[0:num_train_samples]

    return samples, labels

def load_validation_data():
    validation_ratio = 0.3
    num_validation_samples = int(num_samples * validation_ratio)
    num_train_samples = num_samples - num_validation_samples

    samples = SAMPLES[num_train_samples:num_samples]
    labels = LABELS[num_train_samples:num_samples]
    
    return samples, labels

class MyDataset(mdata.Dataset):
    def __init__(self, is_train=True):
        if is_train:
            samples, labels = load_train_data()
        else:
            samples, labels = load_validation_data()
        
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.samples[index, :, :, :]
        label = self.labels[index]

        return sample, label


def get_data():
    train_data = MyDataset()
    val_data = MyDataset(is_train=False)

    return train_data, val_data

def copy_batch_to_device(batch_samples, batch_labels, device):
    return batch_samples.copyto(device), batch_labels.copyto(device)

def train(model, train_data, val_data, num_epochs, batch_size, num_workers, learning_rate, device):
    train_loader = mdata.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = mdata.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.initialize(ctx=device)

    optim = mx.optimizer.Adam(learning_rate=learning_rate)
    trainer = gluon.Trainer(model.collect_params(), optimizer=optim)
    loss_fn = gluon.loss.L2Loss()
    metric_fn = mx.metric.MSE()

    # train step
    for epoch in range(num_epochs):
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            pbar.set_description('[{}/{}] train'.format(epoch, num_epochs))

            for batch_train_samples, batch_train_labels in train_loader:
                batch_train_samples, batch_train_labels = copy_batch_to_device(batch_train_samples, batch_train_labels, device)

                with mx.autograd.record():
                    outputs = model(batch_train_samples)
                    losses = loss_fn(outputs, batch_train_labels)
                losses.backward()

                trainer.step(batch_size)

                mean_losses = losses.sum().asscalar()
                pbar.set_postfix(loss=mean_losses)
                pbar.update(1)

        metric_fn.reset()
        with tqdm.tqdm(total=len(val_loader)) as pbar:
            pbar.set_description('[{}/{}] eval'.format(epoch, num_epochs))

            for batch_val_samples, batch_val_labels in val_loader:
                batch_val_samples, batch_val_labels = copy_batch_to_device(batch_val_samples, batch_val_labels, device)
                outputs = model(batch_val_samples)

                metric_fn.update(outputs, batch_val_labels)
                pbar.set_postfix(mse=metric_fn.get())
                pbar.update(1)

def test_train_data(train_data):
    data_loader = mdata.DataLoader(train_data, batch_size=2)
    for idx, (data, label) in enumerate(data_loader):
        print(idx, data.shape, label.shape)

def main(args):
    learning_rate = args['lr']
    num_workers = args['workers']
    batch_size = args['batch']
    num_epochs = args['epochs']
    device_id = args['device']

    device = get_device(device_id)

    model = get_model(1)
    print('loaded model: {}'.format(model))

    train_data, val_data = get_data()
    
    train(model=model, train_data=train_data, val_data=val_data, num_epochs=num_epochs, batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate, device=device)

def get_device(device_id):
    if device_id >= 0:
        device = mx.gpu(device_id)
    else:
        device = mx.cpu(abs(device_id) - 1)

    return device

def get_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--epochs", type=int)

    args, _ = parser.parse_known_args()

    params = {}
    for arg in vars(args):
        params[arg] = getattr(args, arg)

    return params


if __name__ == "__main__":
    args = get_arguments()
    main(args)