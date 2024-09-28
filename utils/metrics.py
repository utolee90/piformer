import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def SMAE(pred, true): # signed mae, 실제값 - 예측값
    return np.mean(true-pred)

# 상관계수 추측
def REC_CORR(pred, true, flag='mean'): 
    
    # 2차원 - 개별변수
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            return np.mean(np.array([np.corrcoef(pred[:,j], true[:,j])[0,1] for j in range(pred.shape[1]) if not np.isnan(np.corrcoef(pred[:,j], true[:,j])[0,1])]))
        elif flag in ['median', 'med']:
            return np.median(np.array([np.corrcoef(pred[:,j], true[:,j])[0,1] for j in range(pred.shape[1]) if not np.isnan(np.corrcoef(pred[:,j], true[:,j])[0,1])]))
        else:
            return 
    elif pred.ndim == 3:
        if flag == 'mean_total':
            return np.mean(np.array(
                    [
                        [np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for k in range(pred.shape[2]) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k])[0,1])] 
                        for j in range(len(pred))
                    ])
                )
        elif flag == 'median_total':
            return np.median(np.array(
                    [
                        [np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for k in range(pred.shape[2]) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k])[0,1])] 
                        for j in range(len(pred))
                    ])
                )
        elif flag in  ['mean', 'average', 'me', 'avg']:
            return np.mean([
                np.mean([np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k]))[0,1]])
                for k in range(pred.shape[2])
            ])
        elif flag in ['median', 'med']:
            return np.median([
                np.median([np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k]))[0,1]])
                for k in range(pred.shape[2])
            ])
        else:
            return 
 
def RATIO_IRR(pred, true, coef=2): # 오차값 분석. 기본값 표준편차 2
    
    tot_size = np.size(pred)
    mae = MAE(pred, true)

    # Calculate absolute errors
    err = np.abs(pred - true)
    
    # Determine large errors (errors that are k times larger than MAE)
    large_errors = err > coef * mae
    
    # Calculate the ratio of large errors
    large_error_ratio = np.sum(large_errors) / np.size(true)
    
    return large_error_ratio

# std_ratio - 참값과 예측값 표준편차 비교 - 100 이하만 필터링
def STD_RATIO(pred, true, flag='mean'):
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            std_preds = np.array([np.std(pred[:, j]) for j in range(pred.shape[1])])
            std_trues = np.array([np.std(true[:, j]) for j in range(pred.shape[1])])
            ratios = 1/2 * (std_trues / std_preds + std_preds / std_trues)
            ratios[ratios>100] = np.nan # 100 이상의 값을 모두 nan처리
            return np.nanmean(ratios)
        elif flag in ['median', 'med']:
            std_preds = np.array([np.std(pred[:, j]) for j in range(pred.shape[1])])
            std_trues = np.array([np.std(true[:, j]) for j in range(pred.shape[1])])
            ratios = 1/2 * (std_trues / std_preds + std_preds / std_trues)
            ratios[ratios>100] = np.nan # 100 이상의 값을 모두 nan처리
            return np.nanmedian(ratios)
    elif pred.ndim == 3:
        if flag in  ['mean', 'average', 'me', 'avg']:
            std_preds = np.array([[np.std(pred[l, :, j]) for j in range(pred.shape[2])] for l in range(pred.shape[0])])
            std_trues = np.array([[np.std(true[l, :, j]) for j in range(pred.shape[2])] for l in range(pred.shape[0])])
            ratios = 1/2 * (std_trues / std_preds + std_preds / std_trues)
            ratios[ratios>100] = np.nan # 100 이상의 값을 모두 nan처리
            return np.nanmean(ratios)
        elif flag in  ['median', 'med']:
            std_preds = np.array([[np.std(pred[l, :, j]) for j in range(pred.shape[2])] for l in range(pred.shape[0])])
            std_trues = np.array([[np.std(true[l, :, j]) for j in range(pred.shape[2])] for l in range(pred.shape[0])])
            ratios = 1/2 * (std_trues / std_preds + std_preds / std_trues)
            ratios[ratios>100] = np.nan # 100 이상의 값을 모두 nan처리
            return np.nanmedian(ratios)
    
    return None


# 선형회귀 계수 구하기
def get_slope(X, y):
    # X에 bias term 추가 (1로 채워진 열을 맨 앞에 추가)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # [n_samples, n_features+1]
    
    # 선형 회귀 공식: θ = (X^T * X)^(-1) * X^T * y
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best[1]

# slope_ratio - 기울기 차이 절대값 평균
def SLOPE_RATIO(pred, true, flag='mean'):
    X = np.array(range(pred.shape[-2])) # 기울기

    if pred.ndim == 2:
        rate_pred = np.array([get_slope(X, pred[:, j]) for j in range(pred.shape[1])])
        rate_true = np.array([get_slope(X, true[:, j])  for j in range(pred.shape[1])])
        if flag in  ['mean', 'average', 'me', 'avg']:
            return (len(X)-1) * np.mean(np.abs(rate_true - rate_pred))# 참값 대비 예측값 기울기 평균
        elif flag in ['median', 'med']:
            return (len(X)-1)  * np.median(np.abs(rate_true - rate_pred)) # 참값 대비 예측값 기울기 중간값
    elif pred.ndim == 3:
        rate_pred = np.array([[get_slope(X, pred[l, :, j]) for j in range(pred.shape[2])] for l in range(pred.shape[0])])
        rate_true = np.array([[get_slope(X, true[l, :, j]) for j in range(pred.shape[2])] for l in range(pred.shape[0])])
        if flag in  ['mean', 'average', 'me', 'avg']:
            return (len(X)-1) *np.mean(np.abs(rate_true - rate_pred))
        elif flag in  ['median', 'med']:
            return (len(X)-1) *np.median(np.abs(rate_true - rate_pred))
     
    return None