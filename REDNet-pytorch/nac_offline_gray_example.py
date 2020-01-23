import argparse
import os
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
from model_gray import REDNet10, REDNet20, REDNet30
import numpy as np

from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    dtype = torch.cuda.FloatTensor
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    aug_torch = [np_to_torch(np_image.copy()).type(dtype), np_to_torch(np.rot90(np_image, 1, (1, 2)).copy()).type(dtype),
                 np_to_torch(np.rot90(np_image, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(np_image, 3, (1, 2)).copy()).type(dtype)]
    aug_torch += [np_to_torch(flipped.copy()).type(dtype), np_to_torch(np.rot90(flipped, 1, (1, 2)).copy()).type(dtype),
                  np_to_torch(np.rot90(flipped, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(flipped, 3, (1, 2)).copy()).type(dtype)]

    return aug, aug_torch


np.random.seed(30)
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet20', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--noise_level', type=int, default=5)
    opt = parser.parse_args()
    os.makedirs(opt.outputs_dir, exist_ok=True)
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()
    average_psnr = 0
    average_ssim = 0
    aug_mean = True
    dtype = torch.cuda.FloatTensor
    files = os.listdir(opt.image_path)
    for idx, image_name in enumerate(files):
        filename = os.path.basename(os.path.join(opt.image_path, image_name)).split('.')[0]

        label = pil_image.open(os.path.join(opt.image_path, image_name)).convert('L')
        label = np.array(label).astype(np.float32)/255.
        label = np.expand_dims(label, 2)
        [w, h, c] = label.shape
        if w%2==1:
            label = label[:w-1, :, :]
        if h%2==1:
            label = label[:, :h-1, :]
        noise = np.random.normal(0.0, 1.0, size=label.shape)*(opt.noise_level/255.)
        noise2 = np.rot90(noise, 1, (0, 1))
        if aug_mean:
            input_aug, _ = create_augmentations(label.transpose(2, 0, 1))
            test_out = []
            with torch.no_grad():
                for idx, test_img_aug_ in enumerate(input_aug):
                    if idx % 2 == 0:
                        test_noisy_img_torch = np_to_torch(test_img_aug_ + noise.transpose(2, 0, 1)).type(dtype)
                    if idx % 2 == 1:
                        test_noisy_img_torch = np_to_torch(test_img_aug_ + noise2.transpose(2, 0, 1)).type(dtype)
                    out_effect_np_ = torch_to_np(model(test_noisy_img_torch))
                    test_out.append(out_effect_np_)
            test_out[0] = test_out[0].transpose(1, 2, 0)
            for aug in range(1, 8):
                if aug < 4:
                    test_out[aug] = np.rot90(test_out[aug].transpose(1, 2, 0), 4 - aug)
                else:
                    test_out[aug] = np.flipud(np.rot90(test_out[aug].transpose(1, 2, 0), 8 - aug))
            final_reuslt = np.mean(test_out, 0)
            pred = (np.clip(final_reuslt, 0, 1)*255).astype(np.uint8)
        else:
            input = label + noise

            input = transforms.ToTensor()(input).unsqueeze(0).to(device, dtype=torch.float)

            with torch.no_grad():
                pred = model(input)

            pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        psnr = compare_psnr(pred.astype(np.float)/255., label, data_range=1)
        ssim = compare_ssim(pred.astype(np.float)/255., label, data_range=1, multichannel=True)
        average_psnr += psnr
        average_ssim += ssim
        output = pil_image.fromarray(pred[:,:,0], mode='L')
        output.save(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, opt.arch)))
        print('image %s, sigma : %.f, psnr: %.2f, ssim: %.4f'%(image_name, opt.noise_level, psnr, ssim))

    average_psnr = average_psnr/len(files)
    average_ssim = average_ssim/len(files)
    print('Average psnr: %.2f, average ssim: %.4f'%(average_psnr, average_ssim))