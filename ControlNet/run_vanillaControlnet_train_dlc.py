from share import *
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger, LossLogger
from cldm.model import create_model, load_state_dict
import random, os
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime


# Configs
resume_path = '/scratch/tuttare/DeepShade_repo/ControlNet/models/control_sd21_ini.ckpt'

batch_size = 16
logger_freq = 300
learning_rate = 1e-4
sd_locked = True
only_mid_control = False

# Define your dataset class
import json
import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, seed=42):
        self.data = []
        with open('/scratch/tuttare/DeepShade_repo/dataset/Tempe/train_ok.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
                random.seed(seed)
                random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == '__main__':
    model = create_model('/scratch/tuttare/DeepShade_repo/ControlNet/models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    dataset = MyDataset()
    print('There are data size of:', len(dataset))
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    training_dir = f"/scratch/tuttare/DeepShade_repo/logs/ControlNet_vanilla_Tempe/{time_str}/"

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    loss_logger = LossLogger(log_dir= training_dir+'loss_curves/')

    best_checkpoint_callback = ModelCheckpoint(
        dirpath= training_dir + 'best/',
        filename='best-{epoch:02d}-{train_loss_simple_step:.4f}',
        monitor='train/loss_simple_step',
        mode='min',
        save_top_k=1
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath= training_dir + 'periodic/',
        filename='epoch-{epoch:02d}',
        every_n_epochs=10,
        save_top_k=-1
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp',
        precision=32,
        callbacks=[loss_logger, best_checkpoint_callback],
        max_epochs=50
    )

    print('Running batch size:', batch_size, 'Learning rate:', learning_rate, 'Resume from:', resume_path)

    # Train
    trainer.fit(model, train_dataloaders=dataloader)

    # Save final checkpoint
    final_ckpt_path = os.path.join(training_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_ckpt_path)

    print(f"\nTraining completed. Final model saved at:\n{final_ckpt_path}\n")

