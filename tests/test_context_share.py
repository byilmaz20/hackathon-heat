import numpy as np
import pandas as pd

from heattents.models.context_share import ContextShare, day_context, event_calendar


def test_event_calendar_rules():
    ev = event_calendar([2025])
    assert ev[pd.Timestamp("2025-06-29")] == "pride"  # last Sunday of June
    assert ev[pd.Timestamp("2025-06-08")] == "puerto_rican_day"  # second Sunday of June
    assert ev[pd.Timestamp("2025-09-01")] == "west_indian_day"  # Labor Day (a parade wins over the US Open)
    assert ev[pd.Timestamp("2025-08-26")] == "us_open"
    assert len(event_calendar([2020])) == 0  # cancelled years


def _synthetic_panel(seed=0):
    rng = np.random.default_rng(seed)
    areas = ["10014", "11224", "10036", "10451"]
    dates = pd.date_range("2016-06-01", "2019-08-31")
    dates = dates[dates.month.isin([6, 7, 8])]
    P = pd.DataFrame([(a, d) for d in dates for a in areas], columns=["modzcta", "date"])
    P["weekend"] = (P.date.dt.day_of_week >= 5).astype(int)
    P["holiday"] = 0
    P["atmax"] = 32.0
    P["jobs"], P["subway_entries"], P["pop_est"] = 1000.0, 100.0, 5000.0
    P["park_share"] = P.modzcta.map({"11224": 0.25}).fillna(0.02)
    P = day_context(P)
    base = P.modzcta.map({"10014": 1.0, "11224": 1.0, "10036": 2.0, "10451": 1.0}).to_numpy()
    lam = 2.0 * base * np.where((P.modzcta == "11224") & (P.leisure == 1), 3.0, 1.0)
    lam = lam * np.where((P.event == "pride") & (P.modzcta == "10014"), 8.0, 1.0)
    P["y"] = rng.poisson(lam)
    P["mapped_total"] = P.groupby("date").y.transform("sum")
    P["year"] = P.date.dt.year
    return P


def test_shares_normalised_and_day_specific():
    P = _synthetic_panel()
    m = ContextShare(10.0).fit(P)
    sh = P.assign(share=m.shares(P))
    assert np.allclose(sh.groupby("date").share.sum(), 1.0)
    beach = sh[sh.modzcta == "11224"].groupby("leisure").share.mean()
    assert beach[1] > 1.8 * beach[0]  # beach gains on weekends
    pride = sh[(sh.modzcta == "10014") & (sh.event == "pride")].share.mean()
    assert pride > 2 * sh[(sh.modzcta == "10014") & (sh.event == "none")].share.mean()
