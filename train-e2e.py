import argparse
import glob
import itertools
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.autograd import Variable

import data_utils
import model
import plot_utils

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def parse_args():
    desc = "Pytorch implementation of Learning Invariant Representation for CL (IRCL) on the Split MNIST benchmark"
    parser = argparse.ArgumentParser(description=desc)

    # data
    parser.add_argument("--img_size",
                        type=int,
                        default=28,
                        help="dimensionality of the input image")
    parser.add_argument("--channels",
                        type=int,
                        default=1,
                        help="dimensionality of the input channels")
    parser.add_argument("--n_classes",
                        type=int,
                        default=10,
                        help="total number of classes")

    # architecture
    parser.add_argument("--latent_dim",
                        type=int,
                        default=32,
                        help="dimensionality of the latent code")
    parser.add_argument(
        '--n_hidden_cvae',
        type=int,
        default=300,
        help='number of hidden units in conditional variational autoencoder')
    parser.add_argument('--n_hidden_specific',
                        type=int,
                        default=20,
                        help='number of hidden units in the specific module')
    parser.add_argument(
        '--n_hidden_classifier',
        type=int,
        default=40,
        help='number of hidden units in the classification module')

    # training parameters
    parser.add_argument('--learn_rate',
                        type=float,
                        default=1e-2,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='the number of epochs to run')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=1000,
                        help='test Batch size')
    parser.add_argument("--log_interval",
                        type=int,
                        default=50,
                        help="interval between logging")
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')

    parser.add_argument("--seed", type=int, default=1, help="seed")

    # visualization
    parser.add_argument(
        '--results_path',
        type=str,
        default='results',
        help='the path of output images (generated and reconstructed)')
    parser.add_argument('--n_img_x',
                        type=int,
                        default=8,
                        help='number of images along x-axis')
    parser.add_argument('--n_img_y',
                        type=int,
                        default=8,
                        help='number of images along y-axis')

    return check_args(parser.parse_args())


def check_args(args):
    # results_path
    try:
        os.mkdir(args.results_path)
    except (FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path + '/*')

    for f in files:
        os.remove(f)

    return args


def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)

    return Variable(cat)


def visualize(args, test_loader, encoder, decoder, epoch, n_classes, device):
    plotter = plot_utils.plot_samples(args.results_path, args.n_img_x,
                                      args.n_img_y, args.img_size,
                                      args.img_size)
    # plot samples of the reconstructed images from the first batch of the test set of the current task

    for test_batch_idx, (test_data, test_target) in enumerate(test_loader):
        test_data, test_target = test_data.to(device), test_target.to(device)
        x = test_data[0:plotter.n_total_imgs, :]
        x_id = test_target[0:plotter.n_total_imgs]
        x_id_onehot = get_categorical(x_id, n_classes).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z, _, _ = encoder(x)
            reconstructed_x = decoder(torch.cat([z, x_id_onehot], dim=1))
            reconstructed_x = reconstructed_x.reshape(plotter.n_total_imgs,
                                                      args.img_size,
                                                      args.img_size)
            plotter.save_images(x.cpu().data,
                                name="/x_epoch_%02d" % (epoch) + ".jpg")
            plotter.save_images(reconstructed_x.cpu().data,
                                name="/reconstructed_x_epoch_%02d" % (epoch) +
                                ".jpg")

        break

    # plot pseudo random samples from the previous learned tasks
    z = Variable(
        Tensor(np.random.normal(0, 1,
                                (plotter.n_total_imgs, args.latent_dim))))
    z_id = np.random.randint(0, 10, size=[plotter.n_total_imgs])
    z_id_one_hot = get_categorical(z_id, n_classes).to(device)
    decoder.eval()
    with torch.no_grad():
        pseudo_samples = decoder(
            torch.cat([z, Variable(Tensor(z_id_one_hot))], 1))
        pseudo_samples = pseudo_samples.reshape(plotter.n_total_imgs,
                                                args.img_size, args.img_size)
        plotter.save_images(pseudo_samples.cpu().data,
                            name="/pseudo_sample_epoch_%02d" % (epoch) +
                            ".jpg")


def evaluate(encoder, specific, classifier, device, task_test_loader):
    correct_class = 0
    n = 0
    specific.eval()
    classifier.eval()
    encoder.eval()
    with torch.no_grad():
        for data, target in task_test_loader:
            data, target = data.to(device), target.to(device,
                                                      dtype=torch.int64)
            n += target.shape[0]
            z_representation, _, _ = encoder(data)
            specific_representation = specific(data.view(data.shape[0], -1))
            model_output = classifier(specific_representation,
                                      z_representation)
            pred_class = model_output.argmax(dim=1, keepdim=True)
            correct_class += pred_class.eq(
                target.view_as(pred_class)).sum().item()

    print('Test evaluation: ACC: {}/{} ({:.3f}%)'.format(
        correct_class, n, 100 * correct_class / float(n)))

    return 100. * correct_class / float(n)


def train(args, optimizer_cvae, optimizer_C, encoder, decoder, specific,
          classifier, train_loader, test_loader, device):

    ## loss ##
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    classification_loss = nn.CrossEntropyLoss()

    encoder.train()
    decoder.train()
    specific.train()
    classifier.train()

    for epoch in range(args.num_epochs):
        representations = []
        labels = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #---------------------------#
            ## train encoder-decoder ##
            #---------------------------#
            encoder.zero_grad()
            decoder.zero_grad()
            classifier.zero_grad()
            y_onehot = get_categorical(target, args.n_classes).to(device)
            encoded_imgs, z_mu, z_var = encoder(data)
            decoded_imgs = decoder(torch.cat([encoded_imgs, y_onehot], dim=1))
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. -
                                      z_var) / args.batch_size
            rec_loss = pixelwise_loss(decoded_imgs, data) / args.batch_size
            cvae_loss = rec_loss + kl_loss
            cvae_loss.backward()
            optimizer_cvae.step()

            #---------------------------#
            ## train Classifer ##
            #---------------------------#
            encoder.zero_grad()
            decoder.zero_grad()
            classifier.zero_grad()

            z_representation, _, _ = encoder(data)
            specific_representation = specific(data.view(data.shape[0], -1))
            representations.extend(specific_representation.tolist())
            labels.extend(target.tolist())
            outputs = classifier(specific_representation,
                                 z_representation.detach())
            c_loss = classification_loss(outputs, target)
            c_loss.backward()
            optimizer_C.step()

            total_loss = cvae_loss.item() + c_loss.item()

        print(
            f'Train Epoch: {epoch} - AutoEncoder Loss: {cvae_loss:.03f} - Classifer Loss: {c_loss:.03f}'
        )

        if epoch % 5 == 0 or epoch + 1 == args.num_epochs:
            test_acc = evaluate(encoder, specific, classifier, device,
                                test_loader)
            visualize(args, test_loader, encoder, decoder, epoch,
                      args.n_classes, device)

        print('Plotting TSNE')

        if True or epoch == args.num_epochs - 1:
            tsne = TSNE(n_components=2,
                        verbose=0,
                        perplexity=30,
                        n_iter=500,
                        n_iter_without_progress=100,
                        init='pca',
                        learning_rate=100.0,
                        n_jobs=2)
            tsne_results = tsne.fit_transform(representations[:1000])
            title = f'Specific Representation'
            plot_utils.tsne_plot(tsne_results, labels[:1000],
                                 f'results/representation_epoch{epoch}.png',
                                 title)

    return test_acc


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    train_set, test_set = data_utils.load_data()
    img_shape = (1, 28, 28)

    encoder = model.Encoder(img_shape, 300, 32).to(device)
    decoder = model.Decoder(img_shape, 300, 32, 10, use_label=True).to(device)
    specific = model.Specific(img_shape, 20).to(device)
    classifier = model.Classifier(32, 20, 10, 40).to(device)

    optimizer_cvae = torch.optim.Adam(itertools.chain(encoder.parameters(),
                                                      decoder.parameters()),
                                      lr=args.learn_rate)
    optimizer_C = torch.optim.Adam(itertools.chain(classifier.parameters(),
                                                   specific.parameters()),
                                   lr=args.learn_rate / 50)

    train_loader = data_utils.get_train_loader(train_set, args.batch_size)
    test_loader = data_utils.get_train_loader(test_set, args.test_batch_size)

    test_acc = train(args, optimizer_cvae, optimizer_C, encoder, decoder,
                     specific, classifier, train_loader, test_loader, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
