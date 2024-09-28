import numpy as np
from utils.metrics import *
import os 

# test result path

result_path = "./results"

obj_path = "long_term_forecast_iTransformer_weather_96_96_Mod-iTransformer_data-weather.csv_(96to96)_0(1727247965)" # 테스트 결과

res_name = ['pred.npy','true.npy']

np_pred_name = f"{result_path}/{obj_path}/{res_name[0]}"
np_true_name = f"{result_path}/{obj_path}/{res_name[1]}"

np_pred = np.load(np_pred_name) # 예측값
np_true = np.load(np_true_name) # 실제값

metric_path = './results_metric'
if not os.path.exists(metric_path):
    os.makedirs(metric_path)

res_filename = f"{metric_path}/res_{obj_path}_write.txt" # 파일이름

with open(res_filename, 'w', encoding='utf8') as A:
    mse = MSE(np_pred, np_true)
    mae = MAE(np_pred, np_true)
    smae = SMAE(np_pred, np_true)
    std_ratio = STD_RATIO(np_pred, np_true)
    slope_ratio = SLOPE_RATIO(np_pred, np_true)

    wr = obj_path + "\n"
    wr += f"MSE : {mse}, MAE: {mae}, SMAE : {smae}, STD RATIO : {std_ratio}, SLOPE RATIO : {slope_ratio}"

    A.write(wr)

print(wr)


