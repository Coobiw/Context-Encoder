import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import MyDataset
from netG import Net_G
from netD import Net_D
import cfg
# from torch.utils.tensorboard import SummaryWriter
import time

#超参数定义
D_lr = 0.01
G_lr = 0.001
weightR = 0.995
weightA = 0.005
if __name__ == '__main__':
    device = t.device("cuda")#t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    train_set = MyDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)

    train_data = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

    netG = Net_G()

    netG = netG.to(device)

    netD = Net_D()


    netD = netD.to(device)

    D_criterion = nn.BCELoss().to(device)
    D_optimizer = optim.Adam(netD.parameters(), lr=D_lr,betas = (0.9,0.99),weight_decay=0.0005)

    G_criterion = nn.MSELoss().to(device)
    G_optimizer = optim.Adam(netG.parameters(), lr=G_lr, betas=(0.9, 0.99),weight_decay=0.0005)

    # writer = SummaryWriter(log_dir='train_log', comment='train')
    #print('loading')
    netD.train()
    netG.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        t1 = time.time()
        if epoch == cfg.EPOCH_NUMBER//2 and epoch != 0:
            for group in D_optimizer.param_groups:
                group["lr"] *= 0.5
            for group in G_optimizer.param_groups:
                group["lr"] *= 0.5
        if epoch == (3*cfg.EPOCH_NUMBER)//4 and epoch != 0:
            for group in D_optimizer.param_groups:
                group["lr"] *= 0.1
            for group in G_optimizer.param_groups:
                group["lr"] *= 0.1


        for i, sample in enumerate(train_data):
            t3 = time.time()
            img_data = sample["img"].to(device)
            img_label = sample["label"].to(device)
            img_mask = sample['mask'].to(device)
            img_weight = sample['weight'].to(device)


            num = img_data.shape[0]
            G_out = netG(img_data).to(device)
            D_out_real = netD(img_label*img_mask)#.to(device)
            D_out_fake = netD(G_out*img_mask)#.to(device)
            fake_labels = t.zeros((num,1)).to(device)
            real_labels = t.ones((num,1)).to(device)

            D_loss = D_criterion(D_out_fake,fake_labels)+D_criterion(D_out_real,real_labels)
            #print(D_criterion(D_out_fake,fake_labels).item())
            #netD的训练
            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()
            #netG的训练
            G_out = netG(img_data).to(device)
            D_out = netD(G_out*img_mask)
            G_D = D_criterion(D_out,real_labels)
            G_weightedMSE = (G_out*img_mask - img_label*img_mask).pow(2)
            G_weightedMSE = G_weightedMSE * img_weight
            G_weightedMSE = G_weightedMSE.mean()
            G_loss = weightR*G_weightedMSE+weightA*G_D
            #print(G_D)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            t4=time.time()
            print("Epoch %d/%d| Step %d/%d| D_Loss: %.5f G_loss : %.5f Lasting time: %.5f" % (epoch, cfg.EPOCH_NUMBER, i, (len(train_set) // cfg.BATCH_SIZE), D_loss,G_loss,t4-t3))

            print(".........")
            print("finished No.%d batch computing ! " % (i + 1))
        if (epoch + 1)==cfg.EPOCH_NUMBER :
            t.save(netG, "./new_NN/netG/5_%.3f_netG_epoch"% weightR + str(epoch + 1) + ".pth")
            t.save(netD, "./new_NN/netD/5_%.3f_netD_epoch"% weightA + str(epoch + 1) + ".pth")
        t2 = time.time()
        print('the training time per epoch:  ', t2 - t1)
        # writer.add_scalar('netD_loss_line', D_loss, epoch)
        # writer.add_scalar('netG_loss_line', G_loss, epoch)

    print("finish training")

