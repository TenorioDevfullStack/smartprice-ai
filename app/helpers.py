import numpy as np
import pandas as pd

from src.config import AREA, CEP, TARGET, DATE


def money(value):
    return "R$ " + f"{value:,.0f}".replace(",",".")


def market_summary(frame):
    if frame.empty:
        return {"transactions":0,"price_median":0,"area_median":0,"m2_median":0}
    return {"transactions":len(frame),"price_median":float(frame[TARGET].median()),
        "area_median":float(frame[AREA].median()),"m2_median":float((frame[TARGET]/frame[AREA]).median())}


def filter_market(frame, dates, prices, areas, ceps=None):
    mask=frame[DATE].between(pd.Timestamp(dates[0]),pd.Timestamp(dates[1])) & frame[TARGET].between(*prices) & frame[AREA].between(*areas)
    if ceps:
        mask &= frame[CEP].isin(ceps)
    return frame.loc[mask].copy()
