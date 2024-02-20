import numpy as np
from torch.utils.data import Dataset
from .utils import *


__all__ = ['SVDD_Dataset', 'PositionDataset']


# 패치를 생성하기 위한 좌표점 생성하는 함수
# H와 W는 모두 182 (MRI 이미지 사이즈)
# 182*182 이미지 내에서 K*K 크기의 패치를 생성하기 위해 기준 점을 임의로 선택
def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w

# PositionClaissifirer를 위한 패치 1과 패치 2 생성하는 함수
def generate_coords_position(H, W, K):

    # P1 패치(사이즈 : K*K)를 생성하기 위한 좌표
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    # p1을 기준으로 8개 방향 중 임의로 하나의 방향을 선택
    # 8개 방향에 대한 정보는 딕셔너리 자료형 pos_to_diff를 참고
    pos = np.random.randint(8)

    # P2 패치(사이즈 : K*K)를 생성하기 위한 좌표
    # 8개 방향 중 임의로 선택된 방향으로 P2 패치 생성을 위한 좌표를 생성
    with task('P2'):
        J = K // 4

        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + h_diff
        w2 = w1 + w_diff

        #np.clip (arr, min, max) : arr에서 min보다 작은 값은 min으로 대체, max보다 큰 값은 max로 대체
        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2, pos

# SVDD를 위한 패치 1과 패치 2 생성 포지션 클래스 신경망에서 생성하는 기법과 다름
# 방향을 맞추는 것이 아니라 P1 패치와 인접한 패치를 임의로 생성
def generate_coords_svdd(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    with task('P2'):
        J = K // 32

        h_jit, w_jit = 0, 0

        while h_jit == 0 and w_jit == 0:
            h_jit = np.random.randint(-J, J + 1)
            w_jit = np.random.randint(-J, J + 1)

        h2 = h1 + h_jit
        w2 = w1 + w_jit

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2


pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}

# SVDD를 위한 데이터세트 클래스
class SVDD_Dataset(Dataset):
    def __init__(self, memmap, K=64, repeat=1):
        super().__init__()
        self.arr = np.asarray(memmap)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        K = self.K
        n = idx % N

        # (256,256,K) --> (182,182,K)
        p1, p2 = generate_coords_svdd(182,182, K)

        image = self.arr[n]

        # crop_image_CHW 패치 (K*K) 생성하는 함수
        # utils.py에 있음
        patch1 = crop_image_CHW(image, p1, K)
        patch2 = crop_image_CHW(image, p2, K)

        return patch1, patch2

    @staticmethod
    def infer(enc, batch):
        x1s, x2s, = batch
        h1s = enc(x1s)
        h2s = enc(x2s)
        diff = h1s - h2s
        l2 = diff.norm(dim=1)
        loss = l2.mean()

        return loss

# SSL을 위한 데이터세트 클래스
class PositionDataset(Dataset):
    def __init__(self, x, K=64, repeat=1):
        super(PositionDataset, self).__init__()
        self.x = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.x.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.x.shape[0]
        K = self.K
        n = idx % N

        image = self.x[n]
        # generate_coords_position(256, 256, K) --> generate_coords_position(182, 182, K)
        p1, p2, pos = generate_coords_position(182, 182, K)

        patch1 = crop_image_CHW(image, p1, K).copy()
        patch2 = crop_image_CHW(image, p2, K).copy()

        # perturb RGB
        # MRI는 흑백이미지라 RGB perturb가 의미 없다고 생각하여 주석 처리
        # rgbshift1 = np.random.normal(scale=0.02, size=(3, 1, 1))
        # rgbshift2 = np.random.normal(scale=0.02, size=(3, 1, 1))

        # patch1 += rgbshift1
        # patch2 += rgbshift2

        # additive noise
        # np.random.normal(scale=0.02, size=(3, K, K)) --> np.random.normal(scale=0.02, size=(1, K, K))
        noise1 = np.random.normal(scale=0.02, size=(1, K, K))
        noise2 = np.random.normal(scale=0.02, size=(1, K, K))

        patch1 += noise1
        patch2 += noise2

        return patch1, patch2, pos