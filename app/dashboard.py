"""streamlit run app/dashboard.py — mercado ITBI e demonstração do modelo."""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

px.defaults.color_discrete_sequence=["#176B59","#C4863B","#688F85","#BC6650"]

from app.helpers import filter_market, market_summary, money
from src.config import AREA, CEP, STANDARD, TARGET, DATE, DATA_PATH, REPORTS
from src.artifact_contract import artifact_fingerprint, load_metadata
from src.data_processing import load_modeling_data
from src.inference import estimate_price, load_bundle
from src.inference_errors import InferenceError

LOGGER = logging.getLogger("smartprice.dashboard")

st.set_page_config(page_title="SmartPrice AI · São Paulo", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.block-container {max-width:1420px; padding-top:2.6rem; padding-bottom:2rem;}
h1 {font-size:3.15rem !important; letter-spacing:-.045em; font-weight:600 !important;}
h2 {letter-spacing:-.025em;} h3 {font-size:1.35rem !important;}
[data-testid="stMetric"] {padding:1.1rem 1.25rem; border:1px solid #d9ded5; border-radius:6px; background:#fffefa;}
[data-testid="stMetricValue"] {font-size:1.8rem; color:#176B59;}
[data-testid="stSidebar"] {border-right:1px solid #d9ded5;}
.eyebrow {font:600 .72rem sans-serif;letter-spacing:.2em;text-transform:uppercase;color:#176B59;margin-bottom:12px;}
.intro {max-width:780px;font:1.04rem/1.65 sans-serif;color:#55716b;}
.rule {height:1px;background:#ccd7ce;margin:1.7rem 0;}
.smallprint {font:.78rem/1.65 sans-serif;color:#5b7169;}
@media(max-width:640px){h1{font-size:2.1rem !important;} .block-container{padding-top:1.8rem;} [data-testid="stMetricValue"]{font-size:1.35rem;}}
</style>""",unsafe_allow_html=True)


@st.cache_data
def market_data():
    data=load_modeling_data(DATA_PATH)
    data["Preço por m²"]=data[TARGET]/data[AREA]
    return data


@st.cache_resource(max_entries=2)
def cached_model_bundle(fingerprint):
    return load_bundle(expected_fingerprint=fingerprint)


def model_bundle():
    # Sempre confere os arquivos antes de consultar um recurso já em cache.
    return cached_model_bundle(artifact_fingerprint())


@st.cache_data
def diagnostics_data():
    return pd.read_csv(REPORTS/"holdout_plot_sample.csv.gz",parse_dates=[DATE])


def chart(fig):
    fig.update_layout(template="plotly_white",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, serif",color="#183B35"), margin=dict(l=10,r=15,t=40,b=15),height=340,
        colorway=["#176B59","#C4863B","#688F85","#BC6650"])
    st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})


def cards(data):
    values=market_summary(data)
    for column,label,value in zip(st.columns(4),["Transações analisadas","Preço mediano","Área construída mediana","Preço/m² mediano"],
        [f"{values['transactions']:,}".replace(",","."),money(values['price_median']),f"{values['area_median']:.0f} m²",money(values['m2_median'])]):
        column.metric(label,value)


def price_distribution(data):
    plot=pd.DataFrame({"Preço (R$)":np.log10(data[TARGET])})
    fig=px.histogram(plot,x="Preço (R$)",nbins=50,color_discrete_sequence=["#176B59"])
    fig.update_xaxes(title="Preço declarado · escala logarítmica",tickvals=[4,5,6,7,8],ticktext=["10 mil","100 mil","1 mi","10 mi","100 mi"])
    fig.update_yaxes(title="Transações")
    return fig


def header(title,subtitle):
    st.markdown('<div class="eyebrow">São Paulo / ITBI / 2025</div>',unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="intro">{subtitle}</p>',unsafe_allow_html=True)


def overview(data,metadata):
    header("O mercado, pelos dados.","SmartPrice AI conecta transações imobiliárias reais, análise de mercado e uma estimativa experimental de preço de apartamentos em São Paulo.")
    st.caption(f"Coorte: {data[DATE].min():%d/%m/%Y} a {data[DATE].max():%d/%m/%Y} · Compras e vendas integrais · Dados públicos da Prefeitura")
    cards(data)
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.subheader("Quanto custaram as transações?")
        chart(price_distribution(data))
    with right:
        st.subheader("Como o preço mudou no ano?")
        monthly=data.groupby(data[DATE].dt.to_period("M")).agg(preco=(TARGET,"median"),n=(TARGET,"size")).reset_index()
        monthly[DATE]=monthly[DATE].astype(str)
        chart(px.line(monthly,x=DATE,y="preco",markers=True,labels={DATE:"Mês da transação","preco":"Preço mediano (R$)"}))
    left,right=st.columns(2)
    with left:
        st.subheader("Preço por m² construído")
        chart(px.histogram(data,x="Preço por m²",nbins=60,range_x=[0,float(data["Preço por m²"].quantile(.99))]))
        st.caption("Eixo limitado ao percentil 99 para leitura; estatísticas usam todos os registros.")
    with right:
        st.subheader("CEPs com mais transações")
        top=data[CEP].value_counts().head(10).rename_axis("CEP").reset_index(name="Transações")
        chart(px.bar(top,x="Transações",y="CEP",orientation="h"))


def market(data):
    header("Explore suas referências.","Aplique filtros ao conjunto de transações. CEP é uma referência postal; os dados publicados aqui não incluem endereço ou identificação pessoal.")
    with st.expander("Filtros de mercado",expanded=True):
        a,b,c=st.columns(3)
        dates=a.date_input("Período",value=(data[DATE].min().date(),data[DATE].max().date()),min_value=data[DATE].min().date(),max_value=data[DATE].max().date())
        prices=b.slider("Preço declarado (R$)",min_value=0,max_value=int(data[TARGET].max()),value=(0,int(data[TARGET].max())),step=10000)
        areas=c.slider("Área construída (m²)",min_value=0,max_value=int(data[AREA].max()),value=(0,int(data[AREA].max())))
        ceps=st.multiselect("CEPs",options=sorted(data[CEP].unique()),placeholder="Todos os CEPs")
    if len(dates)!=2:
        st.info("Selecione início e fim do período.")
        return
    filtered=filter_market(data,dates,prices,areas,ceps)
    if filtered.empty:
        st.info("Nenhuma transação encontrada. Amplie os filtros.")
        return
    cards(filtered)
    left,right=st.columns(2)
    with left:
        st.subheader("Distribuição do recorte")
        chart(price_distribution(filtered))
    with right:
        st.subheader("Área e preço")
        sample=filtered.sample(min(2500,len(filtered)),random_state=42)
        chart(px.scatter(sample,x=AREA,y=TARGET,opacity=.4,log_y=True,hover_data=[CEP]))
        st.caption("Amostra de até 2.500 transações para o gráfico; cards usam todo o recorte.")
    grouped=filtered.groupby(CEP).agg(transacoes=(TARGET,"size"),preco_mediano=(TARGET,"median"),m2_mediano=("Preço por m²","median")).sort_values("transacoes",ascending=False)
    st.dataframe(grouped.head(30),width="stretch")


def square_meter(data):
    header("O preço de cada metro.","Preço por m² = valor declarado ÷ área construída cadastral. É uma medida descritiva, sensível à definição de área e à composição das transações.")
    cards(data)
    left,right=st.columns(2)
    with left:
        st.subheader("Distribuição de preço/m²")
        plot=pd.DataFrame({"log10_m2":np.log10(data["Preço por m²"])})
        fig=px.histogram(plot,x="log10_m2",nbins=60)
        fig.update_xaxes(title="R$/m² · log10")
        chart(fig)
    with right:
        st.subheader("CEPs com maior volume")
        top=data.groupby(CEP).agg(n=(TARGET,"size"),mediana=("Preço por m²","median")).nlargest(15,"n").reset_index()
        chart(px.bar(top,x="mediana",y=CEP,orientation="h",labels={"mediana":"Mediana (R$/m²)"}))
    monthly=data.groupby(data[DATE].dt.to_period("M"))["Preço por m²"].median().reset_index()
    monthly[DATE]=monthly[DATE].astype(str)
    st.subheader("Evolução da mediana mensal")
    chart(px.line(monthly,x=DATE,y="Preço por m²",markers=True))
    st.caption("O baseline por CEP calcula a mediana de preço/m² somente no treino. Modelos ML nunca recebem preço/m² como feature.")


def distributions(data):
    header("Veja além da mediana.","Os extremos permanecem na análise. A cauda superior e valores declarados atípicos afetam a qualidade das estimativas.")
    a,b=st.columns(2)
    with a:
        chart(price_distribution(data))
    with b:
        chart(px.histogram(data,x=AREA,nbins=60))
    quantiles=data[[TARGET,AREA,"Preço por m²"]].quantile([0,.01,.25,.5,.75,.95,.99,1])
    st.subheader("Quantis do conjunto completo")
    st.dataframe(quantiles,width="stretch")


def estimator(data,metadata):
    header("Uma referência para o imóvel.","Consulte o modelo usando características cadastrais. Os valores refletem o mercado observado em 2025, sem atualização monetária para anos posteriores.")
    try:
        bundle=model_bundle()
    except InferenceError as exc:
        st.error(str(exc))
        return
    left,right=st.columns([1.15,1])
    with left:
        with st.form("estimate"):
            st.subheader("Características do apartamento")
            area=st.number_input("Área construída cadastral (m²)",min_value=1.,max_value=10000.,value=70.,step=1.)
            cep=st.text_input("CEP",value="01234-000",max_chars=12)
            standards=metadata["training_categories"][STANDARD]
            standard=st.selectbox("Padrão IPTU (código cadastral)",options=standards,index=standards.index("22") if "22" in standards else 0)
            year=st.number_input("ACC IPTU (ano cadastral de construção)",min_value=1900,max_value=2025,value=2009,step=1)
            submitted=st.form_submit_button("Estimar preço",type="primary",width="stretch")
        st.caption("Área útil, dormitórios, vagas e estado de conservação não estão disponíveis no modelo. Use a área construída cadastral para manter o contrato dos dados.")
    with right:
        st.subheader("Como interpretar")
        st.write("A faixa usa o percentil 80 do erro absoluto em validação temporal, agrupado pelo preço previsto. Grupos com poucos exemplos usam o erro global.")
        st.write("Ela descreve erros históricos; não é intervalo de confiança estatístico nem garantia de preço de venda.")
        st.caption(f"Modelo selecionado: {metadata['selected_model']} · seleção por MAE em validação")
        if submitted:
            try:
                result=estimate_price(area,cep,standard,year,bundle)
                st.metric("Preço estimado",money(result["prediction"]))
                st.success(f"Faixa baseada no erro histórico do modelo: {money(result['lower'])} a {money(result['upper'])}")
                reference=result["reference_period"]
                st.caption(f"Referência: {reference['start'][:4]} · transações de {reference['start']} a {reference['end']}. Sem atualização monetária.")
                for warning in result["warnings"]:
                    st.warning(warning)
            except InferenceError as exc:
                st.error(str(exc))
    st.info("Modelo experimental baseado em transações de ITBI de São Paulo. Não substitui avaliação imobiliária profissional.")


def performance(metadata):
    header("Performance sem atalhos.","Modelo escolhido nas janelas de validação temporal. Os resultados abaixo são do holdout de 22/10 a 31/12/2025, sem uso na seleção desta etapa.")
    values=metadata["final_holdout_metrics"]
    for c,label,value in zip(st.columns(4),["MAE · reais","RMSE · reais","R²","MedAE · reais"],[money(values["MAE"]),money(values["RMSE"]),f"{values['R2']:.4f}",money(values["MedAE"])]):
        c.metric(label,value)
    st.caption(f"Selecionado: {metadata['selected_model']} · MdAPE: {values['MdAPE_pct']:.2f}% · n = {values['n']:,} · política explícita de previsão ≥ 0")
    st.subheader("Comparação no holdout")
    table=pd.DataFrame(metadata["holdout_comparison"]).T[["MAE","RMSE","R2","MedAE","MdAPE_pct","negative_predictions"]]
    st.dataframe(table,width="stretch")
    st.caption("Comparação descritiva depois da seleção; candidatos sem clipping. O Ridge V1 persistido permanece como referência original.")
    sample=diagnostics_data()
    left,right=st.columns(2)
    with left:
        st.subheader("Real × previsto")
        fig=px.scatter(sample,x="real",y="predicted",opacity=.35,log_x=True,log_y=True,labels={"real":"Real (R$)","predicted":"Previsto (R$)"})
        fig.add_trace(go.Scatter(x=[10000,70000000],y=[10000,70000000],mode="lines",name="Previsão perfeita",line=dict(color="#C4863B",dash="dash")))
        chart(fig)
    with right:
        st.subheader("Resíduos · previsto − real")
        sample["residual"]=sample["predicted"]-sample["real"]
        chart(px.scatter(sample,x="predicted",y="residual",opacity=.35,log_x=True,labels={"predicted":"Previsto (R$)","residual":"Erro (R$)"}))
    st.caption("Gráficos: amostra determinística de 2.500 casos. Métricas e segmentos: holdout completo.")
    st.subheader("Erro por faixa do preço real")
    st.dataframe(pd.DataFrame(metadata["final_holdout_segments"]),width="stretch")
    monthly=json.loads((REPORTS/"holdout_monthly.json").read_text(encoding="utf-8"))
    st.subheader("Erro por período")
    chart(px.bar(pd.DataFrame(monthly),x="month",y="MAE",labels={"month":"Mês","MAE":"MAE (R$)"}))


def diagnosis(metadata):
    header("Onde o modelo erra.","A qualidade da estimativa varia por segmento. Categorias novas, extremos e mudanças de mercado exigem atenção ao interpretar os resultados.")
    tail=metadata["final_holdout_segments"][-1]
    st.warning(f"Acima de R$ 5 milhões: {tail['n']} transações · MAE {money(tail['MAE'])} · subestimação em {tail['underestimate_pct']:.2f}% dos casos.")
    policy=metadata["prediction_policy"]
    st.write(f"Ridge V1 original: {policy['ridge_original_negative_holdout']} previsões negativas. Modelo escolhido antes do limite físico: {policy['selected_raw_negative_holdout']}. Modelo final: {metadata['final_holdout_metrics']['negative_predictions']}.")
    for name,coverage in metadata["unseen_categories"].items():
        st.write(f"{name}: {coverage['unique']} categorias inéditas, presentes em {coverage['rows']} registros do holdout.")
    st.subheader("Seleção e estabilidade temporal")
    ranking=[{"model":row["model"],"MAE validação":row["metrics"]["MAE"],"Desvio MAE entre janelas":row["fold_mae_std"],"RMSE validação":row["metrics"]["RMSE"]} for row in metadata["selection"]["ranking"]]
    st.dataframe(pd.DataFrame(ranking),width="stretch")
    st.write(f"Cobertura observada da faixa no holdout: {metadata['observational_band_holdout_coverage_pct']:.2f}%. Esse resultado não garante a mesma cobertura em novos imóveis.")
    st.caption(metadata["selection"]["caveat"])
    st.subheader("Limitações que acompanham a demonstração")
    st.markdown("- Valores declarados para ITBI podem divergir do preço de mercado.\n- Área construída cadastral pode incluir componentes diferentes da área útil.\n- Sem dormitórios, vagas, conservação ou geocodificação; CEP não é bairro.\n- Coorte de compras e vendas integrais de apartamentos em 2025; não cobre todo o mercado.\n- Sem generalização validada para outras cidades ou para períodos posteriores.\n- Faixas de erro calibradas após seleção no mesmo conjunto de validação podem ser otimistas.")


def main():
    with st.sidebar:
        st.markdown("## SmartPrice AI")
        st.caption("INTELIGÊNCIA IMOBILIÁRIA\nSão Paulo · dados de 2025")
        page=st.radio("Navegação",["Visão geral","Mercado","Preço por m²","Distribuições","Modelo","Performance","Diagnóstico"],label_visibility="collapsed")
        st.divider()
        st.caption("Da transação à estimativa.\nDados públicos, método explícito.")
        st.markdown("[Código e metodologia](https://github.com/TenorioDevfullStack/smartprice-ai)")
    try:
        data=market_data()
    except (OSError,ValueError) as exc:
        LOGGER.error("MARKET_UNAVAILABLE exception_type=%s",type(exc).__name__)
        st.error("Os dados de mercado estão temporariamente indisponíveis.")
        st.stop()
    try:
        metadata=load_metadata()
    except InferenceError:
        metadata=None
        st.warning("Estimativas e diagnósticos do modelo temporariamente indisponíveis. As análises de mercado continuam disponíveis.")
    if page in ["Modelo","Performance","Diagnóstico"] and metadata is None:
        st.error("Estimativa temporariamente indisponível. Tente novamente mais tarde.")
        return
    if page=="Visão geral": overview(data,metadata)
    elif page=="Mercado": market(data)
    elif page=="Preço por m²": square_meter(data)
    elif page=="Distribuições": distributions(data)
    elif page=="Modelo": estimator(data,metadata)
    elif page=="Performance": performance(metadata)
    else: diagnosis(metadata)
    st.divider()
    st.markdown('<div class="smallprint">SmartPrice AI · Transações de ITBI da Prefeitura de São Paulo · Modelo experimental. Não substitui avaliação imobiliária profissional.</div>',unsafe_allow_html=True)


if __name__=="__main__":
    main()
