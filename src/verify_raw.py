"""Reconstrução independente da V3 a partir do XLSX; não executa notebooks.

python -m src.verify_raw — somente leitura; confere cada índice/feature/target/data.
Mantém os filtros e o critério de duplicatas do notebook original.
"""
import json

import numpy as np
import pandas as pd

from src.config import ROOT, AREA, CEP, STANDARD, YEAR, TARGET, DATE, FEATURES
from src.data_processing import sha256, load_recovered_data
from src.feature_engineering import normalize_cep


def verify_raw():
    manifest=json.loads((ROOT/"models/baseline_v1_manifesto.json").read_text(encoding="utf-8"))
    raw=ROOT/manifest["raw_workbook"]
    if sha256(raw)!=manifest["source_sha256"]["raw_workbook"]:
        raise ValueError("XLSX divergente da fonte recuperada.")
    monthly=[]
    for sheet in manifest["sheet_order"]:
        monthly.append(pd.read_excel(raw,sheet_name=sheet).assign(Mes_Arquivo=sheet))
        print(f"Leitura concluída: {sheet}",flush=True)
    frame=pd.concat(monthly,ignore_index=True)
    counts={"raw_2025_sheets":len(frame)}
    frame[DATE]=pd.to_datetime(frame[DATE],errors="coerce",dayfirst=True)
    frame=frame.loc[frame["Descrição do uso (IPTU)"].eq("APARTAMENTO EM CONDOMÍNIO (EXIGE FRAÇÃO IDEAL)")].copy()
    counts["apartments"]=len(frame)
    proportion="Proporção Transmitida (%)"
    for column in [TARGET,AREA,proportion]:
        frame[column]=pd.to_numeric(frame[column],errors="coerce")
    frame=frame.loc[frame["Natureza de Transação"].eq("1.Compra e venda") & frame[proportion].eq(100) & frame[TARGET].gt(0) & frame[AREA].gt(0)].copy()
    frame["Preco_m2"]=frame[TARGET]/frame[AREA]
    counts["v1_positive_whole_sales"]=len(frame)
    frame=frame.loc[frame[TARGET].ge(10000)].copy()
    counts["v2_min_price"]=len(frame)
    frame=frame.loc[frame[DATE].dt.year.eq(2025)].copy()
    counts["transaction_year_2025"]=len(frame)
    counts["exact_duplicates_removed"]=int(frame.duplicated().sum())
    frame=frame.drop_duplicates(keep="first")
    counts["v3"]=len(frame)
    projection=frame[[AREA,STANDARD,YEAR,TARGET,DATE]].copy()
    projection[CEP]=frame["CEP"].map(normalize_cep).astype("string")
    projection[STANDARD]=projection[STANDARD].astype("string")
    projection[YEAR]=pd.to_numeric(projection[YEAR],errors="coerce")
    projection=projection[FEATURES+[TARGET,DATE]].rename_axis("indice_original")
    pd.testing.assert_frame_equal(projection.sort_index(),load_recovered_data(),check_dtype=True)
    if len(frame)!=63164:
        raise ValueError("Contagem V3 divergente.")
    counts["all_rows_match_recovered_partitions"]=True
    counts["raw_sha256"]=sha256(raw)
    counts["duplicate_scope"]="Todas as colunas V2, incluindo Mes_Arquivo; não deduplica operações semelhantes entre meses."
    print(json.dumps(counts,ensure_ascii=True,indent=2))
    return counts


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",action="store_true",help="Persistir somente relatório agregado, sem microdados")
    args=parser.parse_args()
    result=verify_raw()
    if args.report:
        path=ROOT/"reports/raw_verification.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
