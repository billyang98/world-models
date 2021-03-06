""" Recurrent model training """
import argparse
from functools import partial
from os import makedirs, mkdir
from os.path import exists, join

import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.loaders import RolloutSequenceDataset
from models.mdrnn import MDRNN, gmm_loss
from models.vae import VAE

## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import EarlyStopping, ReduceLROnPlateau
from utils.misc import ASIZE, LSIZE, RED_SIZE, RSIZE, SIZE, save_checkpoint

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument(
    "--logdir", type=str, help="Where things are logged and models are loaded from."
)
parser.add_argument(
    "--noreload", action="store_true", help="Do not reload if specified."
)
parser.add_argument(
    "--include_reward",
    action="store_true",
    help="Add a reward modelisation term to the loss.",
)
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
parser.add_argument(
    "--iteration_num",
    type=int,
    help="Iteration number of full traning of the world model, " "VAE, MDNRNN, C",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Directory where the rollouts exist",
    default="datasets/walker",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constants
BSIZE = 16
SEQ_LEN = 32
epochs = args.epochs

# Loading VAE
vae_file = join(args.logdir, "vae", "best.tar")
if args.iteration_num is not None:
    vae_file = join(
        args.logdir, "vae", "iter_{}".format(args.iteration_num), "best.tar"
    )
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print(
    "Loading VAE at epoch {} "
    "with test error {}".format(state["epoch"], state["precision"])
)
print("Loaded VAE from {}".format(vae_file))

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state["state_dict"])

# Loading model
rnn_dir = join(args.logdir, "mdrnn")
# We should load the model from the previous iteration if this is iterative
# training
prev_rnn_dir = rnn_dir
if args.iteration_num is not None:
    rnn_dir = join(args.logdir, "mdrnn", "iter_{}".format(args.iteration_num))
    prev_rnn_dir = join(args.logdir, "mdrnn", "iter_{}".format(args.iteration_num - 1))
if not exists(rnn_dir):
    makedirs(rnn_dir)
rnn_file = join(rnn_dir, "best.tar")
prev_rnn_file = join(prev_rnn_dir, "best.tar")


mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=0.9)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
earlystopping = EarlyStopping("min", patience=30)

if exists(prev_rnn_file) and not args.noreload:
    rnn_state = torch.load(prev_rnn_file)
    print(
        "Loading MDRNN at epoch {} "
        "with test error {}".format(rnn_state["epoch"], rnn_state["precision"])
    )
    print("MDRNN loaded from {}".format(prev_rnn_file))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    earlystopping.load_state_dict(state["earlystopping"])


# Data Loading
dataset_dir = args.dataset_dir
if args.iteration_num is not None:
    dataset_dir = join(dataset_dir, "iter_{}".format(args.iteration_num))

transform = transforms.Lambda(lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset(dataset_dir, SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE,
    num_workers=8,
    shuffle=True,
)
test_loader = DataLoader(
    RolloutSequenceDataset(
        dataset_dir, SEQ_LEN, transform, train=False, buffer_size=10
    ),
    batch_size=BSIZE,
    num_workers=8,
)


def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(
                x.view(-1, 3, SIZE, SIZE),
                size=RED_SIZE,
                mode="bilinear",
                align_corners=True,
            )
            for x in (obs, next_obs)
        ]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)
        ]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(
                BSIZE, SEQ_LEN, LSIZE
            )
            for x_mu, x_logsigma in [
                (obs_mu, obs_logsigma),
                (next_obs_mu, next_obs_logsigma),
            ]
        ]
    return latent_obs, latent_next_obs


def get_loss(
    latent_obs, action, reward, terminal, latent_next_obs, include_reward: bool
):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action, reward, terminal, latent_next_obs = [
        arr.transpose(1, 0)
        for arr in [latent_obs, action, reward, terminal, latent_next_obs]
    ]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epoch, train, include_reward):  # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        if (i+1) * BSIZE > len(loader.dataset):
            break
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(
                latent_obs, action, reward, terminal, latent_next_obs, include_reward
            )

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(
                    latent_obs,
                    action,
                    reward,
                    terminal,
                    latent_next_obs,
                    include_reward,
                )

        cum_loss += losses["loss"].item()
        cum_gmm += losses["gmm"].item()
        cum_bce += losses["bce"].item()
        cum_mse += (
            losses["mse"].item() if hasattr(losses["mse"], "item") else losses["mse"]
        )

        pbar.set_postfix_str(
            "loss={loss:10.6f} bce={bce:10.6f} "
            "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                loss=cum_loss / (i + 1),
                bce=cum_bce / (i + 1),
                gmm=cum_gmm / LSIZE / (i + 1),
                mse=cum_mse / (i + 1),
            )
        )
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, "checkpoint.tar")
    save_checkpoint(
        {
            "state_dict": mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "earlystopping": earlystopping.state_dict(),
            "precision": test_loss,
            "epoch": e,
        },
        is_best,
        checkpoint_fname,
        rnn_file,
    )

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break
