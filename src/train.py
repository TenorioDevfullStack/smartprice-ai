"""python -m src.train — seleção temporal, depois avaliação no holdout."""
import json
import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.config import ROOT, FEATURES, AREA, CEP, STANDARD, TARGET, DATE, HOLDOUT_START, MODEL_PATH, METRICS_PATH, REPORTS
from src.data_processing import freeze_dataset, temporal_windows, sha256
from src.modeling import CANDIDATES, make_candidate, NonNegativeRegressor
from src.evaluation import regression_metrics, price_segments, calibrate_error, observation_band


def interval(frame):
    return {"n":len(frame),"start":str(frame[DATE].min().date()),"end":str(frame[DATE].max().date())}


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def train():
    started=time.perf_counter()
    frame,quality = freeze_dataset()
    development=frame.loc[frame[DATE].lt(HOLDOUT_START)].copy()
    windows=temporal_windows(development)
    validation=[]
    oof={}
    for name in CANDIDATES:
        predictions=[]
        for fold,(training,valid) in enumerate(windows,1):
            begin=time.perf_counter()
            candidate=make_candidate(name)
            with threadpool_limits(limits=2):
                candidate.fit(training[FEATURES],training[TARGET])
                p=candidate.predict(valid[FEATURES])
            metrics=regression_metrics(valid[TARGET],p)
            validation.append({"model":name,"fold":fold,"train":interval(training),"validation":interval(valid),
                "metrics":metrics,"segments":price_segments(valid[TARGET],p),"seconds":round(time.perf_counter()-begin,2)})
            predictions.append(pd.DataFrame({"real":valid[TARGET],"predicted":p,DATE:valid[DATE]}))
            print(f"{name} | janela {fold} | MAE {metrics['MAE']:,.2f} | {time.perf_counter()-begin:.1f}s",flush=True)
        oof[name]=pd.concat(predictions)
    ranking=[]
    for name in CANDIDATES:
        rows=[r["metrics"]["MAE"] for r in validation if r["model"]==name]
        pred=oof[name]
        ranking.append({"model":name,"metrics":regression_metrics(pred["real"],pred["predicted"]),
            "fold_mae_std":float(np.std(rows)),"fold_mae":rows})
    ranking.sort(key=lambda row:row["metrics"]["MAE"])
    selected=ranking[0]["model"]
    # Seleção concluída e persistida ANTES de qualquer avaliação dos candidatos no holdout.
    selection={"selected":selected,"criterion":"MAE ponderado pelo número de observações nas três janelas temporais; desempate pela ordem fixa dos candidatos.",
        "ranking":ranking,"windows":validation,"holdout_used_for_selection":False,
        "caveat":"O holdout V1 foi examinado anteriormente no diagnóstico histórico. Não é uma avaliação prospectiva completamente cega.",
        "target_policy":"Métricas em reais após expm1; sem correção de viés lognormal. Comparação sem clipping na seleção.",
        "representations":{"ridge":"Quatro features V1, CEP completo com categorias raras agrupadas; log altera somente target.",
            "trees":"Área, log1p(área), ACC, prefixo CEP de 3 dígitos e padrão. HGB usa categorias nativas; RF usa one-hot. Prefixo é agrupamento postal, não bairro."}}
    write_json(REPORTS/"selection.json",selection)
    print(f"Selecionado por validação: {selected}. Avaliando holdout agora.",flush=True)
    final=NonNegativeRegressor(make_candidate(selected))
    with threadpool_limits(limits=2):
        final.fit(development[FEATURES],development[TARGET])
    holdout=frame.loc[frame[DATE].ge(HOLDOUT_START)].copy()
    holdout_metrics={}
    diagnostics={}
    final_prediction=final.predict(holdout[FEATURES])
    original_prediction=joblib.load(ROOT/"models/baseline_v1_recuperado.joblib").predict(holdout[FEATURES])
    # Somente relato: nenhuma decisão muda depois de ver estes resultados.
    for name in CANDIDATES:
        if name==selected:
            p=final.predict_unbounded(holdout[FEATURES])
        elif name=="ridge_v1":
            p=original_prediction
        else:
            model=make_candidate(name)
            with threadpool_limits(limits=2):
                model.fit(development[FEATURES],development[TARGET])
                p=model.predict(holdout[FEATURES])
        holdout_metrics[name]=regression_metrics(holdout[TARGET],p)
        diagnostics[name]=price_segments(holdout[TARGET],p)
    calibration=calibrate_error(oof[selected]["real"],oof[selected]["predicted"])
    lower,upper=np.array([observation_band(p,calibration) for p in final_prediction]).T
    unseen={c:sorted(set(holdout[c])-set(development[c])) for c in [CEP,STANDARD]}
    negative_before=int(np.sum(final.predict_unbounded(holdout[FEATURES])<0))
    metadata={"dataset":quality,"features":FEATURES,"target":TARGET,"selected_model":selected,
        "split":{"development":interval(development),"holdout":interval(holdout),"windows":[{"train":interval(t),"validation":interval(v)} for t,v in windows]},
        "selection":selection,"holdout_comparison":holdout_metrics,"holdout_segments":diagnostics,
        "final_holdout_metrics":regression_metrics(holdout[TARGET],final_prediction),
        "final_holdout_segments":price_segments(holdout[TARGET],final_prediction),
        "prediction_policy":{"nonnegative_clip":True,"minimum":0,"selected_raw_negative_holdout":negative_before,"selected_clipped_holdout":negative_before,
            "ridge_original_negative_holdout":int(np.sum(original_prediction<0)),"explanation":"Limite físico explícito >=0 somente na inferência final. Candidatos e Ridge histórico reportados sem clipping."},
        "unseen_categories":{c:{"unique":len(values),"rows":int(holdout[c].isin(values).sum())} for c,values in unseen.items()},
        "training_categories":{c:sorted(development[c].unique().tolist()) for c in [CEP,STANDARD]},
        "training_area_range":[float(development[AREA].min()),float(development[AREA].max())],
        "calibration":calibration,"observational_band_holdout_coverage_pct":float(np.mean((holdout[TARGET]>=lower)&(holdout[TARGET]<=upper))*100),
        "environment":{"python":platform.python_version(),"libraries":{n:version(n) for n in ["numpy","pandas","scikit-learn","scipy","joblib","streamlit","plotly"]}},
        "seconds":round(time.perf_counter()-started,2)}
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(final,MODEL_PATH,compress=3)
    np.testing.assert_array_equal(joblib.load(MODEL_PATH).predict(holdout[FEATURES]),final_prediction)
    metadata["model_sha256"]=sha256(MODEL_PATH)
    write_json(METRICS_PATH,metadata)
    # Amostra exclusivamente para gráficos; métricas acima usam o holdout completo.
    diagnostics_frame=pd.DataFrame({DATE:holdout[DATE],"real":holdout[TARGET],"predicted":final_prediction,"ridge_v1":original_prediction})
    monthly=[{"month":str(month),**regression_metrics(group["real"],group["predicted"])} for month,group in diagnostics_frame.groupby(diagnostics_frame[DATE].dt.to_period("M"))]
    write_json(REPORTS/"holdout_monthly.json",monthly)
    diagnostics_frame.sample(n=min(2500,len(holdout)),random_state=42).sort_index().to_csv(REPORTS/"holdout_plot_sample.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    write_json(REPORTS/"dataset_quality.json",quality)
    print(json.dumps({"selected":selected,"holdout":metadata["final_holdout_metrics"],"tail":metadata["final_holdout_segments"][-1],"model_bytes":MODEL_PATH.stat().st_size,"seconds":metadata["seconds"]},ensure_ascii=True,indent=2),flush=True)
    return metadata


if __name__=="__main__":
    train()
