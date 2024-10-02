import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import time

# checkpoint -> result 불러오기
import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
from utils.metrics import *
from utils.tools import EarlyStopping

from utils.metrics import *
from utils.tools import linear_regression_direct, linear_predict
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom

# 모델 훈련셋 결과 확인하기
from torch.utils.data import DataLoader

# fix random seed
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
parser = argparse.ArgumentParser(description='TimesNet')

# config 지정 -> 오류 방지를 위해 

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# added option 
parser.add_argument('--train_ratio', type=float, default=0.7, help='train data ratio')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test data ratio')
parser.add_argument('--train_step', type=float, default=1.0, help='train data with certain stes. for example train_step=2 means only train even number of data')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period length')

# PITS
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--shared_embedding', type=int, default=1, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Piformer
parser.add_argument('--joint_var', type=int, default=0, help='use attention for each patching; True 1 False 0')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False, 
                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

#piformer
parser.add_argument('--shuffle', type=int, default=1, help="Shuffle data when training")
parser.add_argument('--base_model', type=str, default="iTransformer", help="Base Model Type")

# 스크립트 4개 정리 (./scripts/long_term_forecast/Multi_script/iTransformer_exchange_weather.sh)
scripts_list = ["""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_96 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --d_model 64\
  --d_ff 128\
  --des 'Exp' \
  --itr 1 """,
                """--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_192 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --d_model 64\
  --d_ff 128\
  --des 'Exp' \
  --itr 1 """,
                """--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id iTransformer_weather_96_96 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 64\
  --d_ff 128\
  --itr 1""",
                """--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id iTransformer_weather_96_192 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 64\
  --d_ff 128\
  --itr 1 """]

args0 = parser.parse_args(scripts_list[0].split())
args0.use_gpu = True if torch.cuda.is_available() and args0.use_gpu else False

if args0.use_gpu and args0.use_multi_gpu:
    args0.devices = args0.devices.replace(' ', '')
    device_ids0 = args0.devices.split(',')
    args0.device_ids = [int(id_) for id_ in device_ids0]
    args0.gpu = args0.device_ids[0]

print('Args in experiment:')
print(args0)

args1 = parser.parse_args(scripts_list[1].split())
args1.use_gpu = True if torch.cuda.is_available() and args1.use_gpu else False

if args1.use_gpu and args1.use_multi_gpu:
    args1.devices = args1.devices.replace(' ', '')
    device_ids1 = args1.devices.split(',')
    args1.device_ids = [int(id_) for id_ in device_ids1]
    args1.gpu = args1.device_ids[0]

print('Args in experiment:')
print(args1)

args2 = parser.parse_args(scripts_list[2].split())
args2.use_gpu = True if torch.cuda.is_available() and args2.use_gpu else False

if args2.use_gpu and args2.use_multi_gpu:
    args2.devices = args2.devices.replace(' ', '')
    device_ids2 = args2.devices.split(',')
    args2.device_ids = [int(id_) for id_ in device_ids2]
    args2.gpu = args2.device_ids[0]

print('Args in experiment:')
print(args2)

args3 = parser.parse_args(scripts_list[3].split())
args3.use_gpu = True if torch.cuda.is_available() and args1.use_gpu else False

if args3.use_gpu and args3.use_multi_gpu:
    args1.devices = args3.devices.replace(' ', '')
    device_ids3 = args3.devices.split(',')
    args3.device_ids = [int(id_) for id_ in device_ids3]
    args3.gpu = args3.device_ids[0]

print('Args in experiment:')
print(args3)

# 스크립트 4개 정리 (./scripts/long_term_forecast/Multi_script/iTransformer_exchange_weather.sh)
exchange_96_96_result = "long_term_forecast_iTransformer_Exchange_96_96_Mod-iTransformer_data-exchange_rate.csv_(96to96)_0(1727247520)"
exchange_96_192_result = "long_term_forecast_iTransformer_Exchange_96_192_Mod-iTransformer_data-exchange_rate.csv_(96to192)_0(1727247761)"
weather_96_96_result = "long_term_forecast_iTransformer_weather_96_96_Mod-iTransformer_data-weather.csv_(96to96)_0(1727247965)"
weather_96_192_result = "long_term_forecast_iTransformer_weather_96_192_Mod-iTransformer_data-weather.csv_(96to192)_0(1727250638)"

# 변경해야 할 부분
setting_pairs = [
    (exchange_96_96_result, args0),
    (exchange_96_192_result, args1),
    (weather_96_96_result, args2),
    (weather_96_192_result, args3)
]

idx = 0 # 순서
setting_path = setting_pairs[idx][0]
args = setting_pairs[idx][1]

# 모델 호출 - Exp_Long_Term_Forecast - exchange_96_96
exp_model = Exp_Long_Term_Forecast(args)
exp_model._build_model()
device = exp_model.device

# 위의 argument와 맞는 모델 호출
checkpoint_path = './checkpoints/'
model_path = f"{checkpoint_path}{setting_path}/checkpoint.pth"
exp_model.model.load_state_dict(torch.load(model_path), strict=False)

# data_provider -> Exchange_rate
dataset_input = Dataset_Custom(args, args.root_path,
                                    flag='train', size=(args.seq_len, args.label_len, args.pred_len),
                                    features='M', data_path = args.data_path,
                                    target='OT', scale=True, freq='h', timeenc=0,
                                    seasonal_patterns=None, train_ratio=args.train_ratio, test_ratio=args.test_ratio)
dataset_input_test = Dataset_Custom(args, args.root_path,
                                    flag='test', size=(args.seq_len, args.label_len, args.pred_len),
                                    features='M', data_path = args.data_path,
                                    target='OT', scale=True, freq='h', timeenc=0,
                                    seasonal_patterns=None, train_ratio=args.train_ratio, test_ratio=args.test_ratio)

exp_model.model.eval()


dataset_input_loader = DataLoader(
            dataset_input,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False)
dataset_input_test_loader = DataLoader(
            dataset_input_test,
            batch_size=1, # 모든 데이터셋을 확인해야 해서 batch_size를 강제로 1로 조정.
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)

X = np.array([[t] for t in range(-args.seq_len, 0)])  # X는 입력 feature, shape: [seq_len, 1]
X_new = np.array([[t] for t in range(args.pred_len)])  # 예측을 위한 새로운 시간 변수
X_concat = np.concatenate([X, X_new], axis=0).reshape(-1)

# Combination 모델 제작
class CombinedModel(nn.Module):
    # 모델 정의 - 
    def __init__(self, res_A, res_B, res_C):
        super(CombinedModel, self).__init__()
        self.res_A = res_A  # iTransformer train_result
        self.res_B = res_B  # lin_reg_96 train_result
        self.res_C = res_C
        self.a = nn.Parameter(torch.ones(1, device=device)*0.98, requires_grad=True)
        if res_C != None:
            self.b = nn.Parameter(torch.ones(1, device=device)*0.02, requires_grad=True)
        # self.c = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        # self.d = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    
    def set_a(self, val):
        # nn.Parameter를 다시 생성하지 않고 값을 설정
        with torch.no_grad():
            self.a.copy_(torch.tensor([val], device=device))

    def set_b(self, val):
        # nn.Parameter를 다시 생성하지 않고 값을 설정
        with torch.no_grad():
            self.b.copy_(torch.tensor([val], device=device))
    
    def forward(self, x):
        output_A = self.res_A(x)
        output_B = self.res_B(x)
        if self.res_C == None:
            self.b = nn.Parameter(torch.ones(1, device=device), requires_grad=True) - self.a
            self.c = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
            combined_output = self.a * output_A + self.b * output_B
        else:
            output_C = self.res_C(x)
            self.c = nn.Parameter(torch.ones(1, device=device), requires_grad=True) - self.a - self.b
            combined_output = self.a * output_A + self.b * output_B + output_C * self.c # + self.d
        return combined_output
    
    # model_output_function

def res_iTransformer(batch_x): # S 
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    return exp_model.model(batch_x, None, torch.zeros(B, len(X_new), N), None)

"""
def res_lin_reg(batch_x, reg_size=96):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    # 각 배치와 변수에 대해 선형 회귀 해를 계산
    vals = [[linear_regression_direct(X[-reg_size:], batch_x.permute(0,2,1)[idx, var , -reg_size:], device) for var in range(N)] for idx in range(B)]
    lin_result = [[linear_predict(X_new, vals[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
    return lin_result
"""

def res_lin_reg(batch_x, reg_size=96):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)

    # 각 배치와 변수에 대해 선형 회귀 해를 PyTorch 연산으로 계산
    vals = []
    for idx in range(B):
        for var in range(N):
            # 회귀 계수를 nn.Parameter로 설정해서 학습 가능하게 만듭니다.
            X_torch = torch.tensor(X[-reg_size:], dtype=torch.float32, device=batch_x.device)
            y_torch = batch_x.permute(0, 2, 1)[idx, var, -reg_size:]

            # 선형 회귀 계수 계산
            w = torch.linalg.lstsq(X_torch, y_torch.unsqueeze(1)).solution.squeeze()
            
            # w가 스칼라인 경우, 이를 1D 텐서로 변환
            if w.dim() == 0:
                w = w.unsqueeze(0).to(device)
            
            vals.append(w)

    # 예측값 계산
    lin_result = []
    for idx in range(B):
        batch_result = []
        for var in range(N):
            # print(idx, var, vals[idx*N + var])
            pred = torch.Tensor(X_new).to(device) @ vals[idx * N + var]
            batch_result.append(pred)
        lin_result.append(torch.stack(batch_result, dim=0))

    # 결과를 3D 텐서로 변환
    lin_result = torch.stack(lin_result, dim=0).to(device).permute(0, 2, 1)
    return lin_result


def zero_model(batch_x): # S 길이
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    return torch.zeros(B, len(X_new), N)

# 함수 도출
def get_res_lin(reg_size):
    def fn(reg_size):
        return res_lin_reg(reg_size)
    
    return fn

def res_lin_reg_24(batch_x):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    # 24 조각에 대해서도 계산
    vals_24 = [[linear_regression_direct(X[-24:], batch_x.permute(0,2,1)[idx, var , -24:], device) for var in range(N)] for idx in range(B)]
    lin_result_24 = [[linear_predict(X_new, vals_24[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result_24 = torch.stack([torch.stack(lin_result_24[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
    return lin_result_24


def res_lin_reg_48(batch_x):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    # 48 조각에 대해서도 계산
    vals_48 = [[linear_regression_direct(X[-48:], batch_x.permute(0,2,1)[idx, var , -48:], device) for var in range(N)] for idx in range(B)]
    lin_result_48 = [[linear_predict(X_new, vals_48[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result_48 = torch.stack([torch.stack(lin_result_48[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
    return lin_result_48

def res_lin_reg_12(batch_x):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    # 12 조각에 대해서도 계산
    vals_12 = [[linear_regression_direct(X[-12:], batch_x.permute(0,2,1)[idx, var , -12:], device) for var in range(N)] for idx in range(B)]
    lin_result_12 = [[linear_predict(X_new, vals_12[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result_12 = torch.stack([torch.stack(lin_result_12[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
    return lin_result_12

# 우선 train_set의 data_exchange를 바탕으로 측정값 참값 가져기
# 트레인 데이터셋을 테스트해서 결과 받기, test 함수에서 가져옴
preds_te_tr = [] # 예측값
trues_te_tr = [] # 참값
preds_te_lin = [] # 96_lin
preds_te_lin_48 = [] # 48_lin
preds_te_lin_24 = [] # 24_lin
preds_te_lin_12 = [] # 24_lin

start_time = time.time() # 시작타
print(f"iteration start, time : {time.localtime()}")
    

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    # encoder - decoder

    # use_amp도 사용하지 않음, 
    outputs = exp_model.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    lin_result = res_lin_reg(batch_x, 96).to(device)

    lin_result_24 = res_lin_reg(batch_x, 24).to(device)
    
    # 48 테스트
    lin_result_48 = res_lin_reg(batch_x, 48).to(device)
    
    # 12 테스트
    lin_result_12 = res_lin_reg(batch_x, 12).to(device)
    
    outputs = outputs[:, -args.pred_len:, :]
    batch_y = batch_y[:, -args.pred_len:, :].to(device)
    outputs = outputs.detach().cpu().numpy()
    batch_y = batch_y.detach().cpu().numpy()

    pred = outputs
    true = batch_y

    preds_te_tr.append(pred)
    trues_te_tr.append(true)
    preds_te_lin.append(lin_result)
    preds_te_lin_24.append(lin_result_24)
    preds_te_lin_48.append(lin_result_48)
    preds_te_lin_12.append(lin_result_12)

    if (i+1)%100==0:
        print(f"step {i+1} completed")
    
preds_te_tr = np.concatenate(preds_te_tr, axis=0)
trues_te_tr = np.concatenate(trues_te_tr, axis=0)
preds_te_lin = np.transpose(torch.concat(preds_te_lin, axis=0).detach().cpu().numpy(), (0,2,1))
preds_te_lin_24 = np.transpose(torch.concat(preds_te_lin_24, axis=0).detach().cpu().numpy(), (0,2,1))
preds_te_lin_48 = np.transpose(torch.concat(preds_te_lin_48, axis=0).detach().cpu().numpy(), (0,2,1))
preds_te_lin_12 = np.transpose(torch.concat(preds_te_lin_12, axis=0).detach().cpu().numpy(), (0,2,1))

# operation 끝
data_col_time = time.time()
print(f"경과 시간: {data_col_time-start_time}")

# 모델 실험
num_epochs = 3
combine_model_test = CombinedModel(res_iTransformer, res_lin_reg, None)
combine_model_test_2 = CombinedModel(res_iTransformer, res_lin_reg_24, None)
# combine_model_test training
combine_model_test.train()
combine_model_test_2.train()

criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(combine_model_test.parameters(), lr=0.001)
# optimizer_2 = torch.optim.Adam(combine_model_test_2.parameters(), lr=0.001)

optimizer = torch.optim.Adam([combine_model_test.a], lr=0.001)
optimizer_2 = torch.optim.Adam([combine_model_test_2.a], lr=0.001)

# optimizer = torch.optim.Adam([combine_model_test.a, combine_model_test.b, combine_model_test.c], lr=0.001)
# optimizer_2 = torch.optim.Adam([combine_model_test_2.a, combine_model_test_2.b, combine_model_test_2.c], lr=0.001)

# 검증 데이터셋 결과 확인

def vali(vali_data, vali_loader, criterion):
    total_loss = []
    total_loss_2 = []
    combine_model_test.eval()
    combine_model_test_2.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            targets = batch_y[:, -args.pred_len:, :].to(device)
            outputs = combine_model_test(batch_x)
            outputs_2 = combine_model_test_2(batch_x)
            loss = criterion(outputs, targets)
            loss_2 = criterion(outputs_2, targets)
            total_loss.append(loss)
            total_loss_2.append(loss_2)

    total_loss = [v.item() for v in total_loss]
    total_loss = np.average(total_loss)
    combine_model_test.train()
    total_loss_2 = [v.item() for v in total_loss_2]
    total_loss_2 = np.average(total_loss_2)
    combine_model_test_2.train()
    return total_loss, total_loss_2

# 모델 훈련

early_stopping = EarlyStopping(patience=2, verbose=True)
loss_points = [] # (a, b)
loss_points_2 = [] # (a, b)
input_len = int(np.ceil(len(dataset_input) / args.batch_size) )
input_len_div = int(np.ceil(input_len / 5))
print("SIZE OF OUTPUT_TORCH", input_len, input_len_div)

for epoch in range(num_epochs):
    cnt = 0
    train_loss = []
    train_loss_2 = []
    # exp_model.train()

    path = os.path.join(args.checkpoints, setting_path)
    if not os.path.exists(path):
        os.makedirs(path)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):
        cnt += 1
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        targets = batch_y[:, -args.pred_len:, :].to(device)
        optimizer.zero_grad()
        optimizer_2.zero_grad()
        outputs = combine_model_test(batch_x)
        outputs_2 = combine_model_test_2(batch_x)
        loss = criterion(outputs, targets)
        loss_2 = criterion(outputs_2, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss)
        loss_2.backward()
        optimizer_2.step()
        train_loss_2.append(loss_2)
        if (cnt+1) % 50 == 0:
            print(f"{cnt+1}th batch done, loss {loss}") 
        if i == input_len -1 or i % input_len_div == 0:
            a, b, = combine_model_test.a[0].item(), combine_model_test.b[0].item(),
            loss_points.append((a,b))
            a1, b1, = combine_model_test_2.a[0].item(), combine_model_test_2.b[0].item(),
            loss_points_2.append((a1,b1))

    print("="*50)
    print(f"Epoch {epoch+1} DONE")
    print()
    train_loss = [v.item() for v in train_loss]
    train_loss = np.average(train_loss)
    vali_loss = vali(dataset_input_test, dataset_input_test_loader, criterion)[0]
    train_loss_2 = [v.item() for v in train_loss_2]
    train_loss_2 = np.average(train_loss_2)
    vali_loss_2 = vali(dataset_input_test, dataset_input_test_loader, criterion)[1]
    print("vali_loss:", vali_loss, vali_loss_2)
    print()
    print(combine_model_test.a, combine_model_test.b, combine_model_test.c)
    print(combine_model_test_2.a, combine_model_test_2.b, combine_model_test_2.c)
    # early_stopping(vali_loss, combine_model_test, path)
    # if early_stopping.early_stop:
    #    print("Early stopping")
    #    break
    model_path = path + '/' + 'checkpoint_ensenble.pth'
    # torch.save(combine_model_test.state_dict(), model_path)
    # combine_model_test.load_state_dict(torch.load(model_path, map_location=device))

# 훈련 결과 도출
combine_model_test.eval()

# 계수 수집 끝
end_time = time.time()
print(f"Find Coefficients process done. Spent :{end_time - start_time}")

# 실제 데이터 셋 호출
result_list = ['pred.npy', 'true.npy']
result_path = './results/'
np_pred = np.load(f"{result_path}{setting_path}/{result_list[0]}")
np_true = np.load(f"{result_path}{setting_path}/{result_list[1]}")


# 선형회귀 계수 결과값으로 계산
def get_np_pred_lin(np_pred, reg_size=96):
    # 이제 계산도 한다
    # 각 배치와 변수에 대해 선형 회귀 해를 계산
    B, L, N = np_pred.shape  # L은 시퀀스 길이(seq_len)
    vals = [[linear_regression_direct(X[-reg_size:], dataset_input_test[idx][0][-reg_size:, var]) for var in range(N)] for idx in range(B)]
    lin_result = [[linear_predict(X_new, vals[idx][var]) for var in range(N)] for idx in range(B)]
    # 결과를 numpy 모듈로 변경
    np_pred_lin = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1).detach().cpu().numpy()
    return np_pred_lin

# 이제 계산도 한다
# 각 배치와 변수에 대해 선형 회귀 해를 계산
np_pred_lin = get_np_pred_lin(np_pred, 96)

np_pred_lin_24 = get_np_pred_lin(np_pred, 24)

# loss_points에서 수집한 도트들을 비교 -> 최소 MSE, 최소 MAE 검색, 최소 SMAE 검색
a_res, b_res = (1,0)
a_res_mae, b_res_mae = (1,0)
a_res_smae, b_res_smae = (1,0)
mse_temp = 10 # 임시값
mae_temp = 10 # 임시값
smae_temp = 10 # 임시값
for j, (a,b) in enumerate(loss_points):
    res_temp = a*np_pred + b*np_pred_lin + (1-a-b)*np_pred_lin_24
    mse_step = MSE(res_temp, np_true)
    mae_step = MAE(res_temp, np_true)
    smae_step = SMAE(res_temp, np_true)
    
    if mse_step < mse_temp:
        mse_temp, a_res, b_res = mse_step, a, b
    if mae_step < mae_temp:
        mae_temp, a_res_mae, b_res_mae = mae_step, a, b
    if abs(smae_step) < abs(smae_temp):
        smae_temp, a_res_smae, b_res_smae = smae_step, a, b
        
# 우선 mae 기준으로 할당.
a, b = a_res, b_res


# loss_points에서 수집한 도트들을 비교 -> 최소 MSE, 최소 MAE 검색, 최소 SMAE 검색
a_res_2, b_res_2 = (1,0)
a_res_mae_2, b_res_mae_2 = (1,0)
a_res_smae_2, b_res_smae_2 = (1,0)
mse_temp_2 = 10 # 임시값
mae_temp_2 = 10 # 임시값
smae_temp_2 = 10 # 임시값
for j, (a_2,b_2) in enumerate(loss_points_2):
    res_temp_2 = a_2*np_pred + b_2*np_pred_lin + (1-a_2-b_2)*np_pred_lin_24
    mse_step_2 = MSE(res_temp_2, np_true)
    mae_step_2 = MAE(res_temp_2, np_true)
    smae_step_2 = SMAE(res_temp_2, np_true)
    
    if mse_step_2 < mse_temp_2:
        mse_temp_2, a_res_2, b_res_2 = mse_step_2, a_2, b_2
    if mae_step_2 < mae_temp_2:
        mae_temp_2, a_res_mae_2, b_res_mae_2 = mae_step_2, a_2, b_2
    if abs(smae_step_2) < abs(smae_temp_2):
        smae_temp_2, a_res_smae_2, b_res_smae_2 = smae_step_2, a_2, b_2
        
# 우선 mae 기준으로 할당.
a, b = a_res, b_res

a_2, b_2 = a_res_2, b_res_2

# 마지막으로 비교
final_res = a*np_pred + b* np_pred_lin + (1-a-b)*np_pred_lin_24

final_res_2 = a_2*np_pred + b_2* np_pred_lin_24 + (1-a_2-b_2)*np_pred_lin

timestep = time.time()

# 메트릭 비교하기 (원본 iTransformer)
with open(f'run_ensenble_txt_{setting_path}.txt', 'w', encoding='utf8') as A:
    wr = "it - 96 - 24"
    wr += "TRAIN_PRED\n"
    wr += f"{MSE(np_pred, np_true), MAE(np_pred, np_true), SMAE(np_pred, np_true), REC_CORR(np_pred, np_true), STD_RATIO(np_pred, np_true), SLOPE_RATIO(np_pred, np_true)} \n"
    wr += "TRAIN_ENSEMBLE_PRED\n"
    wr += f"{MSE(final_res, np_true), MAE(final_res, np_true), SMAE(final_res, np_true), REC_CORR(final_res, np_true), STD_RATIO(final_res, np_true), SLOPE_RATIO(final_res, np_true)}\n"
    wr += "LIN_PRED\n"
    wr += "TRAIN_PRED_LINEAR\n"
    wr += f"{MSE(np_pred_lin, np_true), MAE(np_pred_lin, np_true), SMAE(np_pred_lin, np_true), REC_CORR(np_pred_lin, np_true), STD_RATIO(np_pred_lin, np_true), SLOPE_RATIO(np_pred_lin, np_true)}\n"
    wr += "TRAIN_PRED_LINEAR_24\n"
    wr += f"{MSE(np_pred_lin_24, np_true), MAE(np_pred_lin_24, np_true), SMAE(np_pred_lin_24, np_true), REC_CORR(np_pred_lin_24, np_true), STD_RATIO(np_pred_lin_24, np_true), SLOPE_RATIO(np_pred_lin_24, np_true)}\n"
    wr += "\n"
    wr += "it - 24 - 96"
    wr += "TRAIN_ENSEMBLE_PRED\n"
    wr += f"{MSE(final_res_2, np_true), MAE(final_res_2, np_true), SMAE(final_res_2, np_true), REC_CORR(final_res_2, np_true), STD_RATIO(final_res_2, np_true), SLOPE_RATIO(final_res_2, np_true)}\n"
    wr += "\n"
    wr += f"loss1 : {loss_points}"
    wr += "\n"
    wr += f"loss2 : {loss_points_2}"
    A.write(wr)

print("setting_path : ", setting_path)
# 메트릭 저장
metric_path = f"./results/{setting_path}/"
np.save(metric_path + f"pred_ensemble({timestep}).npy", final_res)
np.save(metric_path + f"pred_ensemble_inv({timestep}).npy", final_res)
np.save(metric_path + f"pred_lin({timestep}).npy", np_pred_lin)
np.save(metric_path + f"pred_lin24({timestep}).npy", np_pred_lin_24)

print("JOB FINISHED")