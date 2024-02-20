import argparse
import matplotlib.pyplot as plt
from codes import mvtecad
from tqdm import tqdm
from codes.utils import resize, makedirpath
from skimage.segmentation import mark_boundaries
from codes.inspection import eval_encoder_NN_multiK
from codes.networks import EncoderHier

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='repeat100')
args = parser.parse_args()


def save_maps(name, maps):

    N = maps.shape[0]
    images = mvtecad.get_x(mode='test')
    masks = mvtecad.get_mask()
    print(images.shape)

    for n in tqdm(range(N)):
        fig, axes = plt.subplots(ncols=2)
        fig.set_size_inches(6, 3)

        image = resize(images[n], (128, 128))
        mask = resize(masks[n], (128, 128))
        image = mark_boundaries(image, mask, color=(1, 0, 0), mode='thick')

        axes[0].imshow(image)
        axes[0].set_axis_off()

        axes[1].imshow(maps[n], vmax=maps[n].max(), cmap='Reds')
        axes[1].set_axis_off()

        plt.tight_layout()
        fpath = f'anomaly_maps/{name}/n{n:03d}.png'
        makedirpath(fpath)
        plt.savefig(fpath)
        plt.close()

#########################

def main():
    name = args.name
    enc = EncoderHier(K=64, D=64).cuda()
    enc.load(name)
    enc.eval()
    results = eval_encoder_NN_multiK(enc)
    maps = results['maps_mult']

    save_maps(name, maps)


if __name__ == '__main__':
    main()