import argparse
import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *

parser = argparse.ArgumentParser()

# SVDD Loss를 lambda_value만큼 반영 e.g. 0.8 or 1 ...
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--rep', default=100, type=float)
parser.add_argument('--name', default='repeat100', type=str)

args = parser.parse_args()

def train():
    print('start!')
    D = args.D
    # D가 뭐지? 무슨 차원으로 줄일 건지를 의미하나???
    lr = args.lr
    rep = args.rep
    name = args.name

    # with task() 컨택스트 매니저 자원관리 효용성을 높임
    # utils.py에 있다.
    with task('Networks'):
        print('Newwork Loading')
        # 신경망들은 networks.py에서 관리됨
        # SVDD를 위한 인코더 신경망의 파라미터를 GPU에 load
        enc = EncoderHier(64, D).cuda()
        # 64*64 패치를 가지고 SSL을 위한 포지션 분류기 신경망의 파라미터 GPU에 load
        cls_64 = PositionClassifier(64, D).cuda()
        # 32*32 패치를 가지고 SSL을 위한 포지션 분류기 신경망의 파라미터 GPU에 load
        cls_32 = PositionClassifier(32, D).cuda()

        # 위 모델들을 modules라는 리스트로 관리
        modules = [enc, cls_64, cls_32]
        # 세 개의 신경망의 학습할 파라미터를 params라는 리스트로 관리
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    # 데이터 세트 만듦
    with task('Datasets'):
        print('Make Datasets')
        train_x = mvtecad.get_x_standardized(mode='train')
        train_x = NHWC2NCHW(train_x)

        # rep = 100
        datasets = dict()

        # 64*64 패치 생성 for SSL을 위한 데이터 세트
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        # 32*32 패치 생성 for SSL을 위한 데이터 세트
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        # 64*64 패치 생성 for SVDD를 위한 데이터 세트
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        # 32*32 패치 생성 for SVDD를 위한 데이터 세트
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        # DictionaryConcatDataset는 utils.py에 존재
        # SSL과 SVDD 작업을 위한 데이터세트를 딕셔너리 형태로 관리?
        
        
        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training')
    for i_epoch in range(args.epochs):
        print(f'Epoch : {i_epoch+1} 시작')
        if i_epoch != 0:
            # 3개의 신경망을 각각 학습 모드로 변경
            for module in modules:
                module.train()
            # loader
            for d in loader:
                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                # 64*64 SSl loss함수 계산
                # 학습된 cls_64,  EncoderHier(64, D) 신경망을 가지고, datasets[f'pos_64'] 데이터를 가지고 SSL의 loss 계산
                loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                # 32*32 SSl loss함수 계산 enc.enc
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                # 64*64 패치 loss함수 계산
                # EncoderHier(64, D)로 datasets[f'svdd_64'] 데이터를 가지고 SVDD의 loss 계산
                loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                # 32*32 패치 loss함수 계산
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])
                # Patch SVDD의 Totla loss
                loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                loss.backward()
                opt.step()

        aurocs = eval_encoder_NN_multiK(enc)
        print(f'Epoch : {i_epoch+1} 성능 평가')
        print(f'loss_pos_64 : {loss_pos_64}, loss_pos_32 : {loss_pos_32}, loss_svdd_64 : {loss_svdd_64}, loss_svdd_32 : {loss_svdd_32}')
        print(f'Total loss : {loss}\n')
        log_result(name, aurocs)
        enc.save(name)
        cls_64.save(name)
        cls_32.save(name)

def log_result(name, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |sum| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({name})')

if __name__ == '__main__':
    train()