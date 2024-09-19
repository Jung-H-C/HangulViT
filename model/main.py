import os, sys, time
import logging

import torch
from torch import nn, optim
from metrics.char_error_rate import char_level_accuracy
from model.config import *

from build_model import build_model

def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (image, input_label, output_label) in enumerate(data_loader):
        image = image.to(model.device)
        input_label = input_label.to(model.device)
        output_label = output_label.to(model.device)

        optimizer.zero_grad()
        output = model(image, input_label)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = output_label.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{epoch:04d}.pt") # epoch 번호를 4자리 숫자 형식으로 파일명 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, checkpoint_path)

    return epoch_loss / num_samples

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_cer = []
    with torch.no_grad():
        for idx, (image, input_label, output_label) in enumerate(data_loader):
            image = image.to(model.device)
            input_label = input_label.to(model.device)
            output_label = output_label.to(model.device)

            output = model(image, input_label)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = output_label.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            CER = char_level_accuracy(y_hat, y_gt, PAD_TOKEN)
            total_cer.append(CER)
        num_samples = idx+1

    loss_avr = epoch_loss / num_samples
    cer = sum(total_cer) / len(total_cer)
    return loss_avr, cer


def main():
    model = build_model()

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

        model.apply(initialize_weights)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, verbose = True, factor = SCHEDULER_FACTOR, patience = SCHEDULER_PATIENCE)
