from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores
from .nearest_neighbor import search_NN

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


# 흑백이미지라서 NHWC2NCHW(x)가 필요없음
def infer(x, enc, K, S):
    x = NHWC2NCHW(x)
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(anomaly_scores)
    return auroc_det, auroc_seg


#########################


def eval_encoder_NN_multiK(enc):
    # train 폴더에서 normalized된 이미지
    x_tr = mvtecad.get_x_standardized(mode='train')
    # test 폴더에서 normalized된 이미지
    x_te = mvtecad.get_x_standardized(mode='test')

    # 182*182이미지를 64*64 크기의 필터와 stride는 16으로 설정하여
    # type((182-64 + 0 ) / 16 + 1) != integer 이거 크기가 귀찮게 18
    # 182*182말고 192*192 안 될까요???? 아니면 padding = 5로 설정
    # 이미지 패치를 학습된 encoder로 representation 추출
    embs64_tr = infer(x_tr, enc, K=64, S=16)
    embs64_te = infer(x_te, enc, K=64, S=16)

    # x_tr = mvtecad.get_x_standardized(mode='train')
    # x_te = mvtecad.get_x_standardized(mode='test')

    # 182*182이미지를 32*32 크기의 필터와 stride는 16으로 설정하여
    # type((182-32+0) / 4 + 1) != integer
    # padding을 5로 설정하는 것이 좋겠다.
    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    embs32_te = infer(x_te, enc.enc, K=32, S=4)

    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(embs64, embs32)

# 이 함수가 실질적으로 detection, segmentation 성능을 평가
def eval_embeddings_NN_multiK(embs64, embs32, NN=1):
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    # (256, 256) --> (182, 182)로 바꿈
    # map score 계산
    maps_64 = distribute_scores(maps_64, (182, 182), K=64, S=16)
    det_64, seg_64 = assess_anomaly_maps(maps_64)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (182, 182), K=32, S=4)
    det_32, seg_32 = assess_anomaly_maps(maps_32)

    maps_sum = maps_64 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(maps_sum)

    maps_mult = maps_64 * maps_32
    det_mult, seg_mult = assess_anomaly_maps(maps_mult)

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps