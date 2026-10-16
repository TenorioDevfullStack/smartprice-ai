"""Métricas em reais, segmentos e faixa observacional de validação."""
import numpy as np
from sklearn.metrics import r2_score

from src.config import PRICE_BINS, PRICE_LABELS


def regression_metrics(real, predicted):
    y,p = np.asarray(real,dtype=float),np.asarray(predicted,dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Métricas exigem arrays alinhados e finitos.")
    if not len(y):
        return {"n":0}
    error = p-y
    absolute = np.abs(error)
    return {"n":len(y), "MAE":float(absolute.mean()), "RMSE":float(np.sqrt(np.mean(error**2))),
        "R2":float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None,
        "MedAE":float(np.median(absolute)), "MdAPE_pct":float(np.median(absolute/np.maximum(np.abs(y),1))*100),
        "mean_error":float(error.mean()), "underestimate_pct":float(np.mean(error<0)*100),
        "overestimate_pct":float(np.mean(error>0)*100), "negative_predictions":int(np.sum(p<0))}


def price_segments(real, predicted):
    y,p = np.asarray(real),np.asarray(predicted)
    return [{"segment":label, **regression_metrics(y[(y>low)&(y<=high)],p[(y>low)&(y<=high)])}
        for low,high,label in zip(PRICE_BINS[:-1],PRICE_BINS[1:],PRICE_LABELS)]


def calibrate_error(real, predicted, min_group=50):
    y,p = np.asarray(real), np.maximum(0,np.asarray(predicted))
    errors = np.abs(y-p)
    global_radius = float(np.quantile(errors,.8))
    groups=[]
    for low, high,label in zip(PRICE_BINS[:-1],PRICE_BINS[1:],PRICE_LABELS):
        selected = errors[(p>=low if low==0 else p>low)&(p<=high)]
        groups.append({"segment":label, "n":len(selected), "radius":float(np.quantile(selected,.8)) if len(selected)>=min_group else global_radius,
            "fallback_global":len(selected)<min_group})
    return {"quantile":.8, "global_radius":global_radius, "groups":groups,
        "source":"Previsões fora da amostra nas três janelas de validação; agrupamento pelo preço previsto, nunca pelo preço real na consulta.",
        "label":"Faixa baseada no erro histórico do modelo", "statistical_confidence_interval":False}


def observation_band(prediction, calibration):
    value = max(0,float(prediction))
    index = int(np.searchsorted(PRICE_BINS[1:-1],value,side="left"))
    radius = calibration["groups"][index]["radius"]
    return max(0,value-radius), value+radius
