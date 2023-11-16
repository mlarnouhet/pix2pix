import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import cityscape
from model import Generator, Discriminator
from utils import initialize_weights
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
from fastai.vision.all import show_image
matplotlib.use('Qt5Agg')

def train_net(args, train_dataloader, gen, disc, opt_gen, opt_disc, criterion, L1_Loss):
    for batch_id, (ground_truth, mask) in enumerate(tqdm(train_dataloader)):
        mask = mask.unsqueeze(1).float().to(args.device)
        ground_truth = ground_truth.float().to(args.device)

        ### Train Discriminator
        fake = gen(mask)
        disc_real = disc(mask, ground_truth)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(mask, fake.detach())
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator
        disc_fake = disc(mask, fake)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
        l1_loss = args.lambda_param * L1_Loss(fake, ground_truth)
        loss_gen_tot = loss_gen + l1_loss
        gen.zero_grad()
        loss_gen_tot.backward()
        opt_gen.step()

    print(f"Train loss disc is {loss_disc}")
    print(f"Train loss gen is {loss_gen_tot}")


def evaluate(args, val_dataloader, gen, disc, criterion, L1_Loss):
    gen.eval()
    disc.eval()
    for batch_id, (ground_truth, mask) in enumerate(tqdm(val_dataloader)):
        mask = mask.unsqueeze(1).float().to(args.device)
        ground_truth = ground_truth.float().to(args.device)

        ### Train Discriminator
        fake = gen(mask)
        disc_real = disc(mask, ground_truth)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(mask, fake.detach())
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2


        ### Train Generator
        disc_fake = disc(mask, fake)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
        l1_loss = args.lambda_param * L1_Loss(fake, ground_truth)
        loss_gen_tot = loss_gen + l1_loss

    gen.train()
    disc.train()




    print(f"Val loss disc is {loss_disc}")
    print(f"Val loss gen is {loss_gen_tot}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=2e-4)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--img_size', default=256)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--num_channels', default=3)
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--keep_checkpoints', default=True)
    parser.add_argument('--data_dir', default="data\\")
    parser.add_argument('--lambda_param', default=100)
    parser.add_argument('--use_pretrained', default=False)
    parser.add_argument('--save_pretrained', default=True)

    args = parser.parse_args()

    train(args)


def train(args):

    #Create checkpoint dir
    if args.keep_checkpoints:
        if not os.path.isdir("checkpoint\\"):
            os.makedirs("checkpoint\\")

    print(f"Using device {args.device}")
    transforms = A.Compose(
        [
            A.Resize(height=args.img_size, width=args.img_size),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_dataset = cityscape("data", "train", transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = cityscape("data", "val", transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    gen = Generator(1, 64, 3).to(args.device)
    disc = Discriminator(1, 3, 64).to(args.device)
    initialize_weights(gen)
    initialize_weights(disc)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.999))
    criterion = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    gen.train()
    disc.train()

    if args.use_pretrained:
        print("Loading checkpoint")
        gen_checkpoint = torch.load("checkpoint\\gen.pth.tar")
        disc_checkpoint = torch.load("checkpoint\\disc.pth.tar")
        gen.load_state_dict(gen_checkpoint["state_dict"])
        opt_gen.load_state_dict(gen_checkpoint["optimizer"])
        disc.load_state_dict(disc_checkpoint["state_dict"])
        opt_disc.load_state_dict(disc_checkpoint["optimizer"])

    for epoch in range(args.num_epochs):
        train_net(args, train_dataloader, gen, disc, opt_gen, opt_disc, criterion, L1_Loss)

        if epoch % 10 == 0:
            print("Saving checkpoint")

            torch.save({
                "state_dict": gen.state_dict(),
                "optimizer": opt_gen.state_dict(),
            },
                "checkpoint\\gen.pt")

            torch.save({
                "state_dict": disc.state_dict(),
                "optimizer": opt_disc.state_dict(),
            },
                "checkpoint\\disc.pt")

            evaluate(args, val_dataloader, gen, disc, criterion, L1_Loss)


if __name__ == '__main__':
    main()


