import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cgan_dataset import CGANDataset
from loss.loss import VGGLoss, StyleLoss
from model.cgan import Generator, Discriminator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, train_h5_path, test_h5_path,
                 input_dim, output_channels, lr_g, lr_d, batch_size, num_epochs,
                 device, tf_log_dir, checkpoint_dir="checkpoints",
                 seed=42, accumulate_steps=1, validate_interval=100,
                 val_subset_size=256, test_subset_size=256,
                 g_steps=1, d_steps=1):

        set_seed(seed)

        self.train_h5_path = train_h5_path
        self.test_h5_path = test_h5_path
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.tf_log_dir = tf_log_dir
        self.accumulate_steps = accumulate_steps
        self.validate_interval = validate_interval
        self.val_subset_size = val_subset_size
        self.test_subset_size = test_subset_size
        self.g_steps = g_steps
        self.d_steps = d_steps

        self.generator = Generator(input_dim=input_dim, output_channels=output_channels).to(device)
        self.discriminator = Discriminator(input_channels=output_channels).to(device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.criterion_gan = nn.BCEWithLogitsLoss().to(device)
        self.criterion_vgg = VGGLoss().to(device)
        self.criterion_style = StyleLoss().to(device)
        self.criterion_reconstruction = nn.L1Loss().to(device)

        full_dataset = CGANDataset(train_h5_path, apply_noise=True)
        val_size = 1200
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        self.dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_dataset = CGANDataset(test_h5_path, apply_noise=False)

        self.writer = SummaryWriter(log_dir=tf_log_dir)
        self.best_reconstruction_loss = float('inf')
        self.start_epoch = 0

     
        total_iters = len(self.dataloader) * num_epochs
        warmup_iters = int(0.1 * total_iters)

        def get_iter_scheduler(optimizer):
            def lr_lambda(current_iter):
                if current_iter < warmup_iters:
                    return (current_iter + 1) / warmup_iters
                progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
                return 0.5 * (1 + np.cos(np.pi * progress))
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.g_scheduler = get_iter_scheduler(self.g_optimizer)
        self.d_scheduler = get_iter_scheduler(self.d_optimizer)
        self.total_iters = total_iters

    def evaluate_l1_loss_subset(self, dataset, subset_size=256):
        self.generator.eval()
        with torch.no_grad():
            indices = torch.randperm(len(dataset))[:subset_size]
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            total_loss = 0
            for conditions, real_images, _ in loader:
                conditions = conditions.to(self.device)
                real_images = real_images.to(self.device)
                fake_images = self.generator(conditions)
                loss = self.criterion_reconstruction(fake_images, real_images)
                total_loss += loss.item()

        self.generator.train()
        return total_loss / len(loader)

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({'state_dict': self.generator.state_dict()}, os.path.join(self.checkpoint_dir, f"{epoch}_generator.pth"))
        torch.save({'state_dict': self.discriminator.state_dict()}, os.path.join(self.checkpoint_dir, f"{epoch}_discriminator.pth"))

    def train(self):
        iteration = 0
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), desc="Epochs"):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)
            self.generator.train()
            self.discriminator.train()

            g_accum_counter = 0
            d_accum_counter = 0
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            for i, (conditions, real_images, _) in enumerate(pbar):
                conditions = conditions.to(self.device)
                real_images = real_images.to(self.device)
                batch_size = conditions.size(0)

                real_labels = torch.full((batch_size, 1, 16, 16), 0.9).to(self.device)
                fake_labels = torch.full((batch_size, 1, 16, 16), 0.1).to(self.device)

                for _ in range(self.d_steps):
                    fake_images = self.generator(conditions)
                    d_real = self.discriminator(real_images, conditions)
                    d_fake = self.discriminator(fake_images.detach(), conditions)
                    d_loss_real = self.criterion_gan(d_real, real_labels)
                    d_loss_fake = self.criterion_gan(d_fake, fake_labels)
                    d_loss = (d_loss_real + d_loss_fake) / self.accumulate_steps

                    d_loss.backward()
                    d_accum_counter += 1

                    if d_accum_counter % self.accumulate_steps == 0:
                        self.d_optimizer.step()
                        self.d_optimizer.zero_grad()
                        d_accum_counter = 0

                for _ in range(self.g_steps):
                    fake_images = self.generator(conditions)
                    fake_outputs = self.discriminator(fake_images, conditions)
                    g_loss_gan = self.criterion_gan(fake_outputs, real_labels)
                    g_loss_vgg = self.criterion_vgg(fake_images, real_images)
                    g_loss_style = self.criterion_style(fake_images, real_images)
                    g_loss_reconstruction = self.criterion_reconstruction(fake_images, real_images)
                    g_loss = (2 * g_loss_vgg + g_loss_style + 2 * g_loss_reconstruction + g_loss_gan) / self.accumulate_steps

                    g_loss.backward()
                    g_accum_counter += 1

                    if g_accum_counter % self.accumulate_steps == 0:
                        self.g_optimizer.step()
                        self.g_optimizer.zero_grad()
                        g_accum_counter = 0

                if (iteration + 1) % self.validate_interval == 0:
                    val_l1 = self.evaluate_l1_loss_subset(self.val_dataset, self.val_subset_size)
                    test_l1 = self.evaluate_l1_loss_subset(self.test_dataset, self.test_subset_size)

                    self.writer.add_scalar('Loss/Val_L1', val_l1, iteration)
                    self.writer.add_scalar('Loss/Test_L1', test_l1, iteration)

                    if g_loss_reconstruction.item() < self.best_reconstruction_loss:
                        self.best_reconstruction_loss = g_loss_reconstruction.item()
                        torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, "best_generator.pth"))
                        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, "best_discriminator.pth"))

                self.writer.add_scalar('Loss/D', d_loss.item(), iteration)
                self.writer.add_scalar('Loss/G', g_loss.item(), iteration)
                self.writer.add_scalar('Loss/Rec_L1', g_loss_reconstruction.item(), iteration)
                self.writer.add_scalar('Loss/VGG', g_loss_vgg.item(), iteration)
                self.writer.add_scalar('Loss/Style', g_loss_style.item(), iteration)
                self.writer.add_scalar('Loss/GAN', g_loss_gan.item(), iteration)

                if (iteration + 1) % 100 == 0:
                    print(f"[Iter {iteration + 1}] D={d_loss.item():.4f} | G={g_loss.item():.4f} | Rec={g_loss_reconstruction.item():.4f}")
                    self.save_checkpoint(iteration + 1)

                self.g_scheduler.step()
                self.d_scheduler.step()
                iteration += 1

        self.writer.close()


if __name__ == "__main__":
    train_h5_path = r"train_dataset.h5"
    test_h5_path = r"test_dataset.h5"

    input_dim = 200
    output_channels = 1
    batch_size = 12
    lr_g = 1e-4
    lr_d = 1e-4
    num_epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = "weight/t1"
    tf_log_dir = "log/t1"

    trainer = Trainer(train_h5_path, test_h5_path,
                      input_dim, output_channels,
                      lr_g, lr_d, batch_size, num_epochs,
                      device, tf_log_dir, checkpoint_dir,
                      seed=42,
                      accumulate_steps=4,
                      validate_interval=100,
                      val_subset_size=252,
                      test_subset_size=252)
    trainer.train()
