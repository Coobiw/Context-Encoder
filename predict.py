import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import time

def predict_uneven(img_root,img_name,mask_path,result_root):  # mask里element值为True的地方为非物体，为False为物体
    img = Image.open(img_root+'/'+img_name).convert('L').resize((128, 128))
    img_save = np.array(img).astype('uint8')
    img_no_mask = img_save.copy()
    mask = np.array(Image.open(mask_path).convert('L').resize((128, 128)))
    mask = mask == 0
    # print(mask)
    img_save = img_save * mask
    for ih in range(0, 128):
        for iw in range(0, 128):
            if img_save[ih, iw] == 0:
                img_save[ih, iw] = 255
    img = Image.fromarray(img_save)
    start = time.time()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    im = transform(img).cuda()
    net = torch.load('./model/half_compressed.pth').float()
    net = net.cuda()
    # print(net)
    net.eval()
    with torch.no_grad():
        im = Variable(im[None, :, :, :])
        out = net(im)
    out = (out * 0.5 + 0.5) * 255
    end = time.time()
    print("lasting time is %f seconds"%(end-start))
    out_np = out.squeeze().data.cpu().numpy().astype('uint8')
    out_final = img_save * mask + out_np * (~mask)
    out_final = Image.fromarray(out_final).resize((2448, 2048))
    out_final.save(result_root+'/'+'result_'+img_name)


if __name__ == '__main__':
    img_root = "./test_img"
    img_name = 'image90.jpg'
    mask_path = './test_mask/mask_90.jpg'
    result_root = './test_result'
    predict_uneven(img_root,img_name,mask_path,result_root)