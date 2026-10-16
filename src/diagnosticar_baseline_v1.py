"""Diagnóstico somente por leitura dos artefatos recuperados do Baseline V1.

Não executa células, recuperação, fit, ajustes nem exportação de microdados.
Uso: python src/diagnosticar_baseline_v1.py
Inspeção sem gerar relatório: python src/diagnosticar_baseline_v1.py --inspect
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "pipeline": ROOT / "models/baseline_v1_recuperado.joblib",
    "partitions": ROOT / "data/processed/baseline_v1_particoes.joblib",
    "predictions": ROOT / "data/processed/baseline_v1_previsoes_teste.csv",
    "manifest": ROOT / "models/baseline_v1_manifesto.json",
}
OUT = ROOT / "docs/diagnostico_baseline_v1_2026-10-16"
AREA, CEP, PADRAO, ACC = "Área Construída (m2)", "CEP_8", "Padrão (IPTU)", "ACC (IPTU)"
FEATURES = [AREA, CEP, PADRAO, ACC]


def require(condition, message):
    if not condition:
        raise ValueError(f"VALIDAÇÃO FALHOU; diagnóstico interrompido: {message}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(frame):
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist() if isinstance(frame, pd.DataFrame) else [frame.name],
        "sha256_values_and_index": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(),
        "sha256_ordered_index": hashlib.sha256(frame.index.to_numpy(dtype="int64").tobytes()).hexdigest(),
    }


def metrics(frame):
    if frame.empty:
        return {"n": 0}
    error = frame["previsto"] - frame["real"]
    return {
        "n": len(frame), "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(error.to_numpy() ** 2))),
        "R2": float(r2_score(frame["real"], frame["previsto"])) if len(frame) > 1 and frame["real"].nunique() > 1 else None,
        "vies_medio": float(error.mean()), "vies_mediano": float(error.median()),
        "erro_absoluto_mediano": float(error.abs().median()),
        "erro_percentual_absoluto_mediano": float((error.abs() / frame["real"].abs()).median() * 100),
        "subestimados": int(error.lt(0).sum()), "negativos": int(frame["previsto"].lt(0).sum()),
        "soma_erro_absoluto": float(error.abs().sum()),
        "soma_erro_quadratico": float(np.sum(error.to_numpy() ** 2)),
    }


def validate():
    for path in PATHS.values():
        require(path.is_file(), f"arquivo ausente: {path}")
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in PATHS.values()}
    manifest = json.loads(PATHS["manifest"].read_text(encoding="utf-8"))
    expected_files = {str(PATHS[k].relative_to(ROOT)).replace("\\", "/") for k in ("pipeline", "partitions", "predictions")}
    require(set(manifest["artifact_sha256"]) == expected_files, "lista de hashes de artefatos inesperada")
    for rel, expected in manifest["artifact_sha256"].items():
        require(before[rel] == expected, f"SHA-256 divergente: {rel}")
    pipeline = joblib.load(PATHS["pipeline"])
    parts = joblib.load(PATHS["partitions"])
    require(set(parts) == set(manifest["partition_signatures"]), "partições inesperadas")
    for name, value in parts.items():
        require(signature(value) == manifest["partition_signatures"][name], f"assinatura divergente: {name}")
        require(value.index.is_unique, f"índices duplicados: {name}")
    for split in ("train", "test"):
        x, y, dates = [parts[f"{prefix}_{split}"] for prefix in ("X", "y", "datas")]
        require(x.columns.tolist() == manifest["features"] == FEATURES, "features/ordem divergentes")
        require(x.index.equals(y.index) and x.index.equals(dates.index), f"índices desalinhados: {split}")
        require(len(x) == manifest["split"][f"{split}_rows"], f"contagem: {split}")
        for edge, function in (("start", "min"), ("end", "max")):
            require(str(getattr(dates, function)().date()) == manifest["split"][f"{split}_{edge}"], f"data: {split}/{edge}")
    require(parts["X_train"].index.intersection(parts["X_test"].index).empty, "sobreposição treino/teste")
    require(parts["datas_train"].max() < parts["datas_test"].min(), "sobreposição temporal")
    pred = pd.read_csv(PATHS["predictions"], index_col="indice_original", parse_dates=["Data"], float_precision="round_trip")
    require(pred.columns.tolist() == ["Data", "Valor real (R$)", "Valor previsto (R$)"], "schema CSV")
    require(pred.index.is_unique and pred.index.equals(parts["X_test"].index), "índices CSV")
    require(np.array_equal(pred["Data"].to_numpy(), parts["datas_test"].to_numpy()), "datas CSV")
    require(np.array_equal(pred["Valor real (R$)"].to_numpy(), parts["y_test"].to_numpy()), "target CSV")
    require(np.isfinite(pred[["Valor real (R$)", "Valor previsto (R$)"]].to_numpy()).all(), "valores não finitos")
    require(pred["Valor real (R$)"].gt(0).all(), "target não positivo")
    require(np.array_equal(pipeline.predict(parts["X_test"]), pred["Valor previsto (R$)"].to_numpy()), "previsões não reproduzidas exatamente")
    params = pipeline.get_params(deep=True)
    params = {k: v if v is None or isinstance(v, (str, int, bool)) or isinstance(v, float) and np.isfinite(v) else repr(v) for k, v in params.items()}
    require(params == manifest["pipeline_parameters"], "parâmetros do pipeline divergentes")
    return manifest, before, pipeline, parts, pred


def describe(frame):
    result = {}
    for col in ["real", "previsto", AREA, ACC, "erro", "erro_abs"]:
        values = frame[col]
        result[col] = {"ausentes": int(values.isna().sum()), "media": float(values.mean()), **{key: float(value) for key, value in values.quantile([0, .25, .5, .75, 1]).items()}}
    return result


def groups(frame, column):
    return [{"grupo": str(name), **metrics(group)} for name, group in frame.groupby(column, observed=True, dropna=False)]


def analyze(manifest, pipeline, parts, pred):
    train, test = parts["X_train"], parts["X_test"]
    frame = test.copy(deep=True)
    frame["real"] = pred["Valor real (R$)"]
    frame["previsto"] = pred["Valor previsto (R$)"]
    frame["erro"] = frame["previsto"] - frame["real"]
    frame["erro_abs"] = frame["erro"].abs()
    frame["alto_valor"] = frame["real"].gt(5_000_000)
    frame["negativo"] = frame["previsto"].lt(0)
    known = train[CEP].dropna().unique()
    unseen = test[CEP].notna() & ~test[CEP].isin(known)
    frame["cobertura"] = np.select([test[CEP].isna(), unseen], ["Ausente", "Inédito"], default="Conhecido")
    frame["faixa_area"] = pd.cut(frame[AREA], [-np.inf, 50, 100, 200, 500, np.inf], labels=["Até 50 m²", ">50–100 m²", ">100–200 m²", ">200–500 m²", ">500 m²"])
    frame["faixa_acc"] = pd.cut(frame[ACC], [-np.inf, 1980, 2000, 2020, np.inf], labels=["ACC até 1980", "ACC 1981–2000", "ACC 2001–2020", "ACC após 2020"])

    pre = pipeline.named_steps["preprocessamento"]
    cep_pipe = pre.named_transformers_["cep"]
    encoder = cep_pipe.named_steps["one_hot"]
    train_codes = cep_pipe.named_steps["imputacao"].transform(train[[CEP]])[:, 0]
    test_codes = cep_pipe.named_steps["imputacao"].transform(test[[CEP]])[:, 0]
    learned = encoder.categories_[0]
    infrequent = encoder.infrequent_categories_[0]
    infrequent = np.array([]) if infrequent is None else infrequent
    encoded = cep_pipe.transform(test[[CEP]])
    names = encoder.get_feature_names_out([CEP]).tolist()
    effective_unseen = ~np.isin(test_codes, learned)
    effective_rare = np.isin(test_codes, infrequent)
    frame["grupo_encoder"] = np.select([effective_unseen, effective_rare], ["Inédito", "Conhecido infrequente"], default="Conhecido frequente")
    rare_columns = [i for i, name in enumerate(names) if name.endswith("_infrequent_sklearn")]
    require(len(rare_columns) <= 1, "mais de uma coluna infrequente")
    row_sums = np.asarray(encoded.sum(axis=1)).ravel()
    if len(infrequent):
        require(len(rare_columns) == 1, "coluna infrequente ausente")
        bucket = rare_columns[0]
        for mask in [effective_unseen, effective_rare]:
            require(np.all(np.asarray(encoded[mask, bucket].toarray()).ravel() == 1), "codificação infrequente inesperada")
            require(np.all(row_sums[mask] == 1), "mais de uma categoria ativa")
    else:
        require(np.all(row_sums[effective_unseen] == 0), "CEP inédito deveria gerar zeros sem bucket")
    counts = pd.Series(train_codes).value_counts()
    require(set(infrequent.tolist()) == set(counts[counts.lt(encoder.min_frequency)].index), "vocabulário infrequente incompatível com frequências")

    # Decomposição aditiva das previsões do pipeline já ajustado, sem fit.
    matrix = pre.transform(test)
    ridge = pipeline.named_steps["regressao"]
    coef = ridge.coef_
    contributions = {}
    for name in ["numericas", "cep", "padrao"]:
        section = pre.output_indices_[name]
        contributions[name] = np.asarray(matrix[:, section] @ coef[section]).ravel()
    rebuilt = sum(contributions.values()) + float(ridge.intercept_)
    require(np.allclose(rebuilt, frame["previsto"], rtol=1e-12, atol=1e-7), "decomposição não reproduz previsões")
    numeric_columns = next(columns for name, _, columns in pre.transformers_ if name == "numericas")
    require(list(numeric_columns) == [AREA, ACC], "ordem das numéricas inesperada")
    numeric_details = {}
    numeric_start = pre.output_indices_["numericas"].start
    for offset, label in enumerate(["area", "acc"]):
        column = numeric_start + offset
        numeric_details[label] = np.asarray(matrix[:, column].toarray()).ravel() * coef[column]
    padrao_pipe = pre.named_transformers_["padrao"]
    padrao_codes = padrao_pipe.named_steps["imputacao"].transform(test[[PADRAO]])[:, 0]
    padrao_encoder = padrao_pipe.named_steps["one_hot"]
    frame["padrao_inedito"] = ~np.isin(padrao_codes, padrao_encoder.categories_[0])

    high, low = frame[frame["alto_valor"]], frame[~frame["alto_valor"]]
    neg, nonneg = frame[frame["negativo"]], frame[~frame["negativo"]]
    result = {
        "metodologia": {"erro_assinado": "previsto - real", "unidade": "R$", "alto_valor": "real > 5.000.000", "negativo": "previsto < 0", "grupos_exploratorios": "faixas de área e categorias originais; sem ajuste, inferência causal ou testes de significância"},
        "global": metrics(frame), "alto_valor": metrics(high), "demais_valores": metrics(low),
        "negativos": metrics(neg), "nao_negativos": metrics(nonneg),
        "perfis": {"alto_valor": describe(high), "demais_valores": describe(low), "negativos": describe(neg), "nao_negativos": describe(nonneg)},
        "cobertura": {
            "ceps_distintos_treino": int(train[CEP].nunique()), "ceps_distintos_teste": int(test[CEP].nunique()),
            "ceps_distintos_intersecao": len(set(train[CEP].dropna()) & set(test[CEP].dropna())),
            "ceps_distintos_ineditos": int(test.loc[unseen, CEP].nunique()), "registros_ineditos": int(unseen.sum()),
            "ausentes_treino": int(train[CEP].isna().sum()), "ausentes_teste": int(test[CEP].isna().sum()),
            "metricas": groups(frame, "cobertura"),
            "metricas_sem_alto_valor": groups(low, "cobertura"),
            "metricas_alto_valor": groups(high, "cobertura"),
        },
        "encoder": {
            "min_frequency": encoder.min_frequency, "handle_unknown": encoder.handle_unknown,
            "categorias_aprendidas": len(learned), "categorias_infrequentes": len(infrequent),
            "registros_treino_infrequentes": int(np.isin(train_codes, infrequent).sum()),
            "categorias_frequentes": len(learned) - len(infrequent), "colunas_saida": encoded.shape[1],
            "bucket_infrequente": bool(rare_columns), "ineditos_mapeados_bucket": int(effective_unseen.sum()) if rare_columns else 0,
            "teste_conhecidos_infrequentes": int(effective_rare.sum()),
            "teste_infrequentes_ou_ineditos": int((effective_unseen | effective_rare).sum()),
            "metricas": groups(frame, "grupo_encoder"),
            "padrao_handle_unknown": padrao_encoder.handle_unknown,
            "padrao_inedito_teste": int(frame["padrao_inedito"].sum()),
            "padrao_inedito_negativos": int(neg["padrao_inedito"].sum()),
        },
        "associacoes": {
            "alto_valor_area": groups(high, "faixa_area"), "alto_valor_padrao": groups(high, PADRAO),
            "alto_valor_acc": groups(high, "faixa_acc"), "alto_valor_encoder": groups(high, "grupo_encoder"),
            "alto_valor_spearman_area_erro": float(high[[AREA, "erro"]].corr(method="spearman").iloc[0, 1]),
            "alto_valor_spearman_area_erro_abs": float(high[[AREA, "erro_abs"]].corr(method="spearman").iloc[0, 1]),
            "negativos_cobertura": groups(neg, "cobertura"), "negativos_encoder": groups(neg, "grupo_encoder"),
            "negativos_area": groups(neg, "faixa_area"), "teste_area": groups(frame, "faixa_area"),
            "negativos_padrao": groups(neg, PADRAO), "teste_padrao": groups(frame, PADRAO),
            "negativos_acc": groups(neg, "faixa_acc"), "teste_acc": groups(frame, "faixa_acc"),
            "negativos_ceps_distintos": int(neg[CEP].nunique()),
            "negativos_ceps_top": [{"cep": str(c), "n_negativos": int(n), "n_teste": int(frame[CEP].eq(c).sum())} for c, n in neg[CEP].value_counts().head(10).items()],
        },
        "decomposicao": {
            "intercepto": float(ridge.intercept_),
            "diferenca_maxima_recomposicao": float(np.max(np.abs(rebuilt - frame["previsto"].to_numpy()))),
            "medias_negativos": {name: float(values[frame["negativo"].to_numpy()].mean()) for name, values in contributions.items()},
            "medias_nao_negativos": {name: float(values[~frame["negativo"].to_numpy()].mean()) for name, values in contributions.items()},
            "numericas_negativos": {name: float(values[frame["negativo"].to_numpy()].mean()) for name, values in numeric_details.items()},
            "numericas_nao_negativos": {name: float(values[~frame["negativo"].to_numpy()].mean()) for name, values in numeric_details.items()},
        },
        "participacao_erros_alto_valor": {
            "absoluto_pct": float(high["erro_abs"].sum() / frame["erro_abs"].sum() * 100),
            "quadratico_pct": float(np.sum(high["erro"].to_numpy() ** 2) / np.sum(frame["erro"].to_numpy() ** 2) * 100),
        },
    }
    keys = {"MAE": "MAE (R$)", "RMSE": "RMSE (R$)", "R2": "R²", "erro_absoluto_mediano": "Mediana do erro absoluto (R$)", "erro_percentual_absoluto_mediano": "Erro percentual absoluto mediano (%)"}
    for local, saved in keys.items():
        require(np.isclose(result["global"][local], manifest["metrics_recovered"][saved], rtol=1e-13, atol=1e-9), f"métrica global divergente: {saved}")
    return result, frame


def br(value, decimals=2):
    if value is None:
        return "indefinido"
    return f"{value:,.{decimals}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def metric_table(rows):
    return table(["Grupo", "n", "MAE (R$)", "RMSE (R$)", "R²", "Viés médio (R$)", "Viés mediano (R$)", "Mediana |erro| (R$)", "Mediana |erro|/real"], [
        [r["grupo"], br(r["n"], 0), br(r["MAE"]), br(r["RMSE"]), br(r["R2"], 4), br(r["vies_medio"]), br(r["vies_mediano"]), br(r["erro_absoluto_mediano"]), br(r["erro_percentual_absoluto_mediano"]) + "%"] for r in rows])


def rate_table(rows):
    return table(["Grupo", "Registros no teste", "Previsões negativas", "Taxa no grupo"], [
        [r["grupo"], br(r["n"], 0), br(r["negativos"], 0), br(r["negativos"] / r["n"] * 100) + "%"] for r in rows])


def make_plots(frame, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ordinary = ~frame["alto_valor"] & ~frame["negativo"]
    for mask, color, label, size, alpha in [
        (ordinary, "#6c8397", "Demais previsões", 8, .28),
        (frame["alto_valor"], "#bb541e", "Valor real > R$ 5 milhões (141)", 20, .8),
        (frame["negativo"], "#7f3c8d", "Previsão negativa (326)", 18, .7),
    ]:
        ax.scatter(frame.loc[mask, "real"], frame.loc[mask, "previsto"], c=color, label=label, s=size, alpha=alpha, linewidths=0)
    ax.plot([8_000, 80_000_000], [8_000, 80_000_000], linestyle="--", color="#303030", linewidth=1, label="Previsão = valor real")
    ax.axhline(0, color="#333333", linewidth=.8)
    ax.axvline(5_000_000, color="#bb541e", linewidth=.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=100_000)
    ax.set(xlim=(8_000, 80_000_000), ylim=(-2_300_000, 80_000_000), xlabel="Valor real (R$; escala logarítmica)", ylabel="Valor previsto (R$; escala logarítmica simétrica)", title="Baseline V1 · todas as 12.566 observações de teste")
    def money_tick(value, position):
        if value == 0:
            return "0"
        return br(value / 1_000_000, 0) + " mi" if abs(value) >= 1_000_000 else br(value / 1000, 0) + " mil"
    ax.xaxis.set_major_locator(FixedLocator([10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 70_000_000]))
    ax.yaxis.set_major_locator(FixedLocator([-2_000_000, -100_000, 0, 100_000, 1_000_000, 10_000_000, 70_000_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(money_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(money_tick))
    ax.grid(alpha=.15)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(output / "01_real_vs_previsto.png", dpi=180)
    plt.close(fig)

    rows = result["encoder"]["metricas"]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    pos = np.arange(len(rows))
    for shift, key, color in [(-.18, "MAE", "#216e83"), (.18, "RMSE", "#bb541e")]:
        bars = ax.bar(pos + shift, [r[key] / 1000 for r in rows], width=.36, label=key, color=color)
        ax.bar_label(bars, labels=[br(r[key] / 1000, 1) for r in rows], padding=4)
    ax.set_xticks(pos, [r["grupo"].replace(" ", "\n", 1) + "\nn=" + br(r["n"], 0) for r in rows])
    ax.set(ylabel="Erro (R$ mil)", title="Cobertura de CEP · categorias aprendidas no treino", ylim=(0, max(r["RMSE"] / 1000 for r in rows) * 1.17))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: br(value, 0)))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.savefig(output / "02_erros_por_cobertura_cep.png", dpi=180)
    plt.close(fig)


def report_text(r):
    g, h, n, c, e, a, d = [r[key] for key in ["global", "alto_valor", "negativos", "cobertura", "encoder", "associacoes", "decomposicao"]]
    profile = r["perfis"]
    known, unseen = c["metricas"]
    blocks = [
        "# RESULTADO DA ETAPA — DIAGNÓSTICO BASELINE V1",
        "**Data:** 16/10/2026 · SmartPrice AI. Diagnóstico executado exclusivamente sobre os quatro artefatos recuperados; sem treinamento, ajuste, remoção de observações ou implementação de V2.",
        "## Validação e metodologia",
        "Os quatro caminhos foram confirmados antes do carregamento:",
        "\n".join("- `" + str(p.resolve()) + "`" for p in PATHS.values()),
        "**Validação: OK.** Os três hashes de arquivos disponíveis no manifesto conferem. As seis assinaturas de partição conferem em contagem, colunas, conteúdo e índice ordenado. X, y e datas têm índices únicos e alinhados; treino e teste são disjuntos. CSV lido com `float_precision='round_trip'`, índice e datas preservados. Targets e datas coincidem exatamente com as partições. `pipeline.predict(X_test)` reproduz bit a bit as previsões salvas. Todos os parâmetros do pipeline conferem com o manifesto e as cinco métricas globais foram reproduzidas.",
        "Treino: **50.598 registros**, 02/01/2025–21/10/2025. Teste: **12.566 registros**, 22/10/2025–31/12/2025. A análise usa uma cópia de X_test; as seis partições mantêm suas assinaturas ao final. Pipeline: Ridge `alpha=1.0`, `solver='lsqr'`.",
        "Erro assinado = **previsto − real**: viés negativo indica subestimação. MAE é a média de |erro| e não a mediana. RMSE é a raiz da média dos erros quadráticos; R² compara com a média do valor real **dentro de cada grupo**, portanto não mede sozinho o efeito da cobertura. A mediana percentual usa |previsto − real| / real. Grupos são descritivos, sem testes de significância ou controle de fatores. Nenhum valor foi truncado.",
        "Limite de proveniência: o manifesto não possui hash de si próprio; seu SHA-256 foi registrado antes/depois desta etapa. Os hashes históricos de fontes brutas e células são referências da recuperação e não foram usados como dados nem revalidados nesta análise. Não havia hashes históricos das partições anteriores à recuperação para comparação independente. A validação atual se refere aos artefatos recuperados.",
        metric_table([{"grupo": "Teste completo", **g}]),
        "## 1. Imóveis com valor real acima de R$ 5 milhões",
        f"**{h['n']} imóveis ({br(h['n']/g['n']*100)}% do teste)**. {h['subestimados']} subestimados ({br(h['subestimados']/h['n']*100)}%); nenhum com previsão negativa. Concentram **{br(r['participacao_erros_alto_valor']['absoluto_pct'])}% da soma dos erros absolutos** e **{br(r['participacao_erros_alto_valor']['quadratico_pct'])}% da soma dos erros quadráticos** do teste.",
        metric_table([{"grupo": "Real > R$ 5 milhões", **h}, {"grupo": "Real ≤ R$ 5 milhões", **r['demais_valores']}]),
        f"Valor real entre R$ {br(profile['alto_valor']['real'][0])} e R$ {br(profile['alto_valor']['real'][1])}; mediana R$ {br(profile['alto_valor']['real'][.5])}. Área mediana **{br(profile['alto_valor'][AREA][.5],0)} m²**, contra {br(profile['demais_valores'][AREA][.5],0)} m² nos demais imóveis; intervalo de {br(profile['alto_valor'][AREA][0],0)}–{br(profile['alto_valor'][AREA][1],0)} m². ACC mediano {br(profile['alto_valor'][ACC][.5],0)}, contra {br(profile['demais_valores'][ACC][.5],0)} nos demais. Não foram atribuídos significados adicionais aos códigos de padrão IPTU.",
        "### Características associadas aos erros de alto valor",
        metric_table(a["alto_valor_area"]),
        f"Acima de 500 m², o MAE em reais é maior, mas a mediana percentual é menor que no grupo de 200–500 m². Dentro dos 141 imóveis, Spearman entre área e erro assinado = **{br(a['alto_valor_spearman_area_erro'],3)}** e entre área e |erro| = **{br(a['alto_valor_spearman_area_erro_abs'],3)}**: não há associação monotônica forte que permita atribuir a subestimação apenas à área.",
        metric_table(a["alto_valor_padrao"]),
        "O padrão 25 tem o maior MAE dentre os grupos com várias observações; 24 concentra a maior quantidade de casos. O padrão 22 tem apenas um caso: sem generalização.",
        metric_table(a["alto_valor_encoder"]),
        "Os conhecidos infrequentes apresentam maior MAE neste segmento; os seis inéditos são insuficientes para inferência estável. Os grupos diferem em preço, área e padrão; a tabela não isola o efeito do CEP.",
        metric_table(a["alto_valor_acc"]),
        "ACC é apresentado com seus valores originais, agrupados em faixas para evitar conclusões a partir de anos com pouquíssimas observações. Essas comparações não estabelecem causalidade.",
        "## 2. Previsões negativas",
        f"**{n['n']} previsões negativas ({br(n['n']/g['n']*100)}% do teste)**, confirmando a quantidade anteriormente reportada. Todas subestimam valores reais positivos. MAE **R$ {br(n['MAE'])}**, viés médio **−R$ {br(abs(n['vies_medio']))}**, viés mediano **−R$ {br(abs(n['vies_mediano']))}**; RMSE R$ {br(n['RMSE'])}.",
        table(["Característica", "Previsões negativas", "Demais previsões"], [
            ["Valor real mediano (R$)", br(profile['negativos']['real'][.5]), br(profile['nao_negativos']['real'][.5])],
            ["Área mediana (m²)", br(profile['negativos'][AREA][.5],0), br(profile['nao_negativos'][AREA][.5],0)],
            ["ACC mediano", br(profile['negativos'][ACC][.5],0), br(profile['nao_negativos'][ACC][.5],0)],
        ]),
        f"Previsões entre **−R$ {br(abs(profile['negativos']['previsto'][0]))}** e **−R$ {br(abs(profile['negativos']['previsto'][1]))}**; mediana −R$ {br(abs(profile['negativos']['previsto'][.5]))}. Valores reais entre R$ {br(profile['negativos']['real'][0])} e R$ {br(profile['negativos']['real'][1])}. Área de {br(profile['negativos'][AREA][0],0)} a {br(profile['negativos'][AREA][1],0)} m²; 50% entre {br(profile['negativos'][AREA][.25],0)} e {br(profile['negativos'][AREA][.75],0)} m².",
        rate_table(a["teste_area"]),
        "**297 de 326 (91,10%)** têm até 100 m². A taxa por faixa confirma a concentração em áreas pequenas; contagem e taxa têm denominadores distintos.",
        rate_table(a["teste_padrao"]),
        "O padrão 22 concentra **251/326 (76,99%)** negativos, mas também domina o teste. A taxa do padrão 21 (**59/836; 7,06%**) supera a do 22 (**251/9.112; 2,75%**); a concentração absoluta em 22 não equivale à maior taxa de ocorrência. Não há padrões inéditos no teste.",
        rate_table(a["teste_acc"]),
        "**197/326 (60,43%)** negativos têm ACC até 1980, contra **1.846/12.240 (15,08%)** nas demais previsões. Esta associação descritiva não separa área, localização e padrão.",
        rate_table(c["metricas"]),
        rate_table(e["metricas"]),
        f"Os negativos envolvem **{a['negativos_ceps_distintos']} CEPs distintos**; **304/326** são de CEPs conhecidos e **227/326** de conhecidos frequentes. Portanto, eles também ocorrem amplamente em categorias aprendidas no treino.",
        table(["CEP", "Negativos", "Registros do CEP no teste", "Taxa"], [[row['cep'], row['n_negativos'], row['n_teste'], br(row['n_negativos']/row['n_teste']*100)+'%'] for row in a['negativos_ceps_top']]),
        "CEPs acima ordenados pela contagem de negativos (dez primeiros); amostras pequenas e seleção pelo resultado impedem tratar suas taxas como estimativas estáveis.",
        "### Conferência algébrica do pipeline ajustado",
        "A saída de `preprocessamento.transform(X_test)` foi multiplicada pelos coeficientes Ridge já aprendidos. Intercepto e blocos reproduzem as previsões com diferença máxima **R$ " + br(d['diferenca_maxima_recomposicao'],8) + "**. Nenhum ajuste foi feito.",
        table(["Componente médio da previsão (R$)", "Negativos", "Demais previsões"], [
            ["Intercepto", br(d['intercepto']), br(d['intercepto'])],
            ["Área padronizada × coeficiente", br(d['numericas_negativos']['area']), br(d['numericas_nao_negativos']['area'])],
            ["ACC padronizado × coeficiente", br(d['numericas_negativos']['acc']), br(d['numericas_nao_negativos']['acc'])],
            ["Bloco CEP", br(d['medias_negativos']['cep']), br(d['medias_nao_negativos']['cep'])],
            ["Bloco padrão IPTU", br(d['medias_negativos']['padrao']), br(d['medias_nao_negativos']['padrao'])],
            ["Soma = previsão média", br(profile['negativos']['previsto']['media']), br(profile['nao_negativos']['previsto']['media'])],
        ]),
        "No grupo negativo, as contribuições médias dos blocos somam um valor abaixo de zero com o intercepto. A Ridge em reais não impõe restrição de positividade. A decomposição explica a conta deste pipeline, sem interpretar coeficientes como efeitos causais; seus valores dependem da codificação e da centralização utilizadas.",
        "## 3. Cobertura de CEPs entre treino e teste",
        table(["Medida", "Quantidade"], [
            ["CEPs distintos no treino", br(c['ceps_distintos_treino'],0)], ["CEPs distintos no teste", br(c['ceps_distintos_teste'],0)],
            ["CEPs distintos presentes em ambos", br(c['ceps_distintos_intersecao'],0)], ["CEPs distintos inéditos no teste", br(c['ceps_distintos_ineditos'],0)],
            ["Registros com CEP inédito", str(c['registros_ineditos']) + ' (' + br(c['registros_ineditos']/g['n']*100) + '% do teste)'],
            ["CEPs ausentes: registros treino / teste", str(c['ausentes_treino']) + ' / ' + str(c['ausentes_teste'])],
        ]),
        "Comparação literal de CEP_8 preservando os tipos originais. Nenhuma normalização adicional, imputação manual ou recomposição da divisão foi feita.",
        "## 4. Desempenho por cobertura de CEP",
        metric_table(c["metricas"]),
        f"Nos inéditos, o MAE é **{br((unseen['MAE']/known['MAE']-1)*100)}% maior**; a mediana percentual passa de {br(known['erro_percentual_absoluto_mediano'])}% para {br(unseen['erro_percentual_absoluto_mediano'])}%. O RMSE dos conhecidos é maior, apesar do MAE menor: ele dá mais peso aos extremos e os grupos têm composições diferentes.",
        "### Conferência da composição por valor real",
        "Entre os conhecidos há **135** imóveis acima de R$ 5 milhões; entre os inéditos, **6**. A comparação abaixo é um estrato descritivo do mesmo teste, **sem excluir registros da base nem da avaliação global**:",
        metric_table(c["metricas_sem_alto_valor"]),
        "No estrato de até R$ 5 milhões, o RMSE dos inéditos supera o dos conhecidos. Isso mostra que a ordenação pelo RMSE agregado não é suficiente para avaliar cobertura. A estratificação não controla todas as diferenças de perfil.",
        "### Separação entre conhecidos frequentes e infrequentes",
        metric_table(e["metricas"]),
        "Os conhecidos infrequentes têm MAE e RMSE maiores que os conhecidos frequentes. Isso sustenta avaliar representações espaciais que compartilhem informação entre CEPs, sem demonstrar que a codificação seja a única responsável.",
        "## 5. OneHotEncoder: comportamento verificado",
        f"O encoder ajustado usa `min_frequency={e['min_frequency']}` e `handle_unknown='{e['handle_unknown']}'`. Aprendeu **{br(e['categorias_aprendidas'],0)} categorias de CEP**, das quais **{br(e['categorias_infrequentes'],0)} CEPs distintos** tiveram menos de cinco registros no treino. Esses CEPs correspondem a **{br(e['registros_treino_infrequentes'],0)} registros de treino**. São {br(e['categorias_frequentes'],0)} categorias frequentes mais uma coluna compartilhada: **{br(e['colunas_saida'],0)} colunas de saída**.",
        f"O bucket infrequente existe. A transformação foi conferida para **todos** os registros: os **{e['ineditos_mapeados_bucket']} inéditos** e os **{br(e['teste_conhecidos_infrequentes'],0)} conhecidos infrequentes** ativam a **mesma coluna** `infrequent_sklearn`, com soma de linha igual a 1 no bloco CEP. Assim, **{br(e['teste_infrequentes_ou_ineditos'],0)} registros de teste ({br(e['teste_infrequentes_ou_ineditos']/g['n']*100)}%)** compartilham essa representação espacial. Os inéditos não recebem um bloco todo zerado neste pipeline.",
        "Esta conclusão foi conferida no encoder já ajustado, sem construir ou ajustar outro encoder. Padrão IPTU usa `handle_unknown='ignore'`, mas há **zero categorias inéditas de padrão** no teste, inclusive entre os negativos. Logo, bloco de padrão zerado por categoria desconhecida não explica esses casos.",
        "## 6. Gráficos",
        "![Valores reais e previstos, incluindo todos os extremos e negativos](01_real_vs_previsto.png)",
        "Figura 1: todos os registros do teste, sem amostragem, corte de outliers ou truncamento. X logarítmico; Y logarítmico simétrico, linear entre −R$ 100 mil e R$ 100 mil para preservar zero e negativos. A linha tracejada indica previsão perfeita; a vertical marca R$ 5 milhões.",
        "![MAE e RMSE por cobertura e frequência de CEP](02_erros_por_cobertura_cep.png)",
        "Figura 2: erros em R$ mil, com n por grupo. Métricas calculadas sobre todos os registros de cada grupo; diferenças de perfil permanecem.",
        "## 7. Principais conclusões",
        "1. Há subestimação concentrada no segmento de alto valor: 129/141 casos; o viés mediano é −R$ 2,27 milhões. Sua participação nos erros quadráticos (73,98%) mostra por que o RMSE global é sensível a esse segmento. Isso não autoriza eliminar extremos.",
        "2. As 326 previsões negativas se concentram em áreas menores e ACCs mais baixos; a decomposição confirma como os termos do modelo ajustado compõem valores negativos. A maioria ocorre em CEPs conhecidos, inclusive frequentes, e nenhum padrão IPTU é inédito.",
        "3. Cobertura espacial importa para a avaliação: inéditos têm MAE e erro percentual mediano maiores; conhecidos infrequentes também têm pior desempenho que frequentes. Comparações observacionais e R² de grupos distintos não demonstram efeito causal do CEP.",
        "## 8. Até três recomendações para o futuro Modelo V2",
        "1. **Avaliar uma formulação com saída positiva**, por exemplo target logarítmico com inversão adequada. Motivação: 326/12.566 previsões negativas e erro percentual mediano de 136,81% nesses casos. Comparar erros em reais e percentuais, viés e desempenho no alto valor; a inversão pode introduzir viés e comprimir o topo. Não adotar truncamento como correção deste diagnóstico.",
        "2. **Avaliar não linearidades e interações de área, ACC, padrão e localização.** Motivação: subestimação em 91,49% dos imóveis de alto valor, MAE elevado no padrão 25 e perfis distintos dos negativos. Isso é hipótese a testar; a correlação fraca entre área e erro no alto valor não sustenta uma explicação por área isolada.",
        "3. **Avaliar representação hierárquica de CEP**, com prefixos e compartilhamento de informação entre raros/inéditos. Motivação: 413 CEPs inéditos e 2.837 registros de teste no mesmo bucket; MAE de R$ 432.055,38 nos conhecidos infrequentes contra R$ 250.321,04 nos frequentes. Comparar cobertura e métricas por grupo, com quaisquer estatísticas aprendidas somente no treino e sem vazamento do target.",
        "Estas são propostas para experimentos futuros. A escolha de V2 exigirá validação temporal no treino ou nova janela de confirmação: este teste já foi inspecionado. Nenhuma das recomendações foi implementada ou teve ganho demonstrado nesta etapa.",
        "## 9. Integridade e reprodução",
        "**Os quatro artefatos originais permaneceram intactos:** SHA-256 antes/depois idêntico, inclusive para o manifesto. Nenhum fit, ajuste, reconstrução de bases ou execução de células foi realizado. As bases, a divisão temporal, os índices, o Ridge e os outliers foram preservados. O notebook também não foi editado. Somente este diretório, o script de análise e o plano desta etapa foram acrescentados.",
        table(["Arquivo", "SHA-256 antes = depois"], [[rel, '`'+digest+'`'] for rel, digest in r['validacao']['hashes_antes'].items()]),
        "Versões efetivas: " + "; ".join(name + " " + version for name, version in r['validacao']['versoes'].items()) + ". Correspondem às versões destas bibliotecas no manifesto de recuperação. A .venv original aponta para um executável ausente; foi usado o Python 3.13.5 disponível, sem instalar dependências.",
        "Comando executado, a partir da raiz: `python src/diagnosticar_baseline_v1.py`. O script valida tudo antes da análise e verifica novamente os artefatos ao final. Para reproduzir os cálculos com a entrega já existente: `python src/diagnosticar_baseline_v1.py --inspect`, que somente imprime os cálculos. Os arquivos desta entrega são criados exclusivamente em diretório novo para impedir sobrescrita acidental. O JSON contém métricas agregadas e evidências de validação, sem exportar microdados.",
        "Verificação adicional: `python -m unittest discover -s tests -v` (2 testes). O teste de proteção invoca o script de recuperação, que recusa os artefatos existentes antes de qualquer treinamento; nenhuma célula é executada.",
        "**Etapa encerrada.** Nenhum Modelo V2 foi treinado ou implementado.",
    ]
    return "\n\n".join(blocks) + "\n"


def write_outputs(result, frame):
    # Nenhuma saída é criada antes de validar integralmente os artefatos.
    if OUT.exists():
        raise FileExistsError(f"Entrega já existe e será preservada: {OUT}. Use --inspect para reproduzir os cálculos sem escrever arquivos.")
    report = report_text(result)
    summary = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUT.mkdir(parents=True, exist_ok=False)
    make_plots(frame, result, OUT)
    (OUT / "relatorio.md").write_text(report, encoding="utf-8")
    (OUT / "resumo_metricas.json").write_text(summary, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    notebook = ROOT / "notebooks/01_exploracao.ipynb"
    notebook_before = sha(notebook)
    manifest, before, pipeline, parts, pred = validate()
    pipeline_before = joblib.hash(pipeline)
    result, frame = analyze(manifest, pipeline, parts, pred)
    result["validacao"] = {"caminhos": {k: str(p.resolve()) for k, p in PATHS.items()}, "hashes_antes": before,
        "assinaturas_particoes": "6/6, conteúdo e índice ordenado", "previsoes": "iguais bit a bit ao CSV",
        "parametros": "iguais ao manifesto", "limite_historico": manifest["historical_membership_validation"],
        "versoes": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "matplotlib"]}}
    after = {rel: sha(ROOT / rel) for rel in before}
    require(after == before, "artefatos alterados durante a análise")
    for key, part in parts.items():
        require(signature(part) == manifest["partition_signatures"][key], f"partição em memória alterada: {key}")
    require(joblib.hash(pipeline) == pipeline_before, "pipeline em memória alterado")
    require(sha(notebook) == notebook_before, "notebook alterado durante o diagnóstico")
    result["validacao"]["hashes_depois"] = after
    result["validacao"]["hash_notebook_antes_depois"] = notebook_before
    result["validacao"]["pipeline_em_memoria_intacto"] = True
    if args.inspect:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        write_outputs(result, frame)
        require({rel: sha(ROOT / rel) for rel in before} == before, "hash final divergente")
        require(sha(notebook) == notebook_before, "notebook alterado durante a entrega")
        print(json.dumps({"relatorio": str(OUT / "relatorio.md"), "validacao": "OK", "global": result["global"], "alto_valor": result["alto_valor"], "negativos": result["negativos"], "cobertura": result["cobertura"]}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
