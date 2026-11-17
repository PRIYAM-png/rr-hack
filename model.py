"""Pressure-performance model. Input: raw DataFrames -> ranked batters/bowlers."""
import numpy as np, pandas as pd

ALIAS = {
 "match_id": ["match_id", "matchid", "game_id"],
 "innings": ["innings", "inning", "innings_no", "inns", "innings_number"],
 "over": ["over", "overs", "over_number"],
 "ball": ["ball", "ball_number", "ball_in_over", "delivery"],
 "batter": ["batter", "batsman", "striker", "batter_id", "batsman_id", "striker_id", "batter_name", "batsman_name"],
 "bowler": ["bowler", "bowler_id", "bowler_name"],
 "bat_runs": ["batter_runs", "runs_off_bat", "batsman_runs", "runs_batter", "bat_runs", "runs_scored"],
 "total_runs": ["total_runs", "runs_total", "runs", "ball_runs", "total"],
 "wides": ["wides", "wide", "extras_wides"], "noballs": ["noballs", "noball", "no_balls", "extras_noballs"],
 "byes": ["byes", "extras_byes"], "legbyes": ["legbyes", "leg_byes", "extras_legbyes"],
 "extra_type": ["extra_type", "extras_type", "extra_types"],
 "wicket_type": ["wicket_type", "dismissal_type", "dismissal_kind", "wicket_kind"],
 "is_wicket": ["is_wicket", "wicket", "is_out", "wicket_fell"],
 "player_out": ["player_out", "player_dismissed", "dismissed_player", "dismissed_player_id"],
}
NON_BOWLER = "run out|retired|obstructing|timed out"

def norm(df):
    df = df.copy(); df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]; return df

def classify(df):
    c = set(df.columns)
    if c & set(ALIAS["batter"]) and c & set(ALIAS["bowler"]): return "bbb"
    if "match_format" in c: return "competitions"
    if {"home_team", "away_team"} <= c: return "matches"
    if "batting_order" in c: return "playing_eleven"
    if {"player_name", "player_id"} <= c: return "mapping"
    if {"team_id", "team_name"} <= c: return "teams"
    return None

def _col(df, key):
    for a in ALIAS[key]:
        if a in df.columns: return df[a]
    return None

def _truthy(s):
    if s.dtype == object:
        return s.fillna("").astype(str).str.strip().str.lower().isin(["", "0", "false", "none", "nan", "0.0"]).eq(False)
    return s.fillna(0).astype(float) != 0

def build_balls(bbb, matches, comps, mapping, W=15.0, q=0.75):
    d = pd.DataFrame(index=bbb.index)
    for k in ("match_id", "innings", "batter", "bowler"):
        s = _col(bbb, k)
        if s is None: raise ValueError(f"Ball-by-ball data is missing a '{k}' column (accepted names: {', '.join(ALIAS[k])}).")
        d[k] = s
    # players may be ids or names
    name2id = {n.lower(): i for i, n in mapping.drop_duplicates("player_id").set_index("player_id")["player_name"].items()}
    for k in ("batter", "bowler"):
        num = pd.to_numeric(d[k], errors="coerce")
        d[k] = num if num.notna().mean() > .9 else d[k].astype(str).str.lower().map(name2id)
    d["match_id"] = pd.to_numeric(d["match_id"], errors="coerce"); d["innings"] = pd.to_numeric(d["innings"], errors="coerce")
    tr, br = _col(bbb, "total_runs"), _col(bbb, "bat_runs")
    if tr is None and br is None: raise ValueError("Ball-by-ball data needs a runs column (total_runs or batter_runs).")
    d["total_runs"] = pd.to_numeric(tr if tr is not None else br, errors="coerce").fillna(0)
    d["bat_runs"] = pd.to_numeric(br if br is not None else tr, errors="coerce").fillna(0)
    wd, nb, et = _col(bbb, "wides"), _col(bbb, "noballs"), _col(bbb, "extra_type")
    d["wide"] = _truthy(wd) if wd is not None else False
    d["noball"] = _truthy(nb) if nb is not None else False
    if et is not None:
        e = et.fillna("").astype(str).str.lower(); d["wide"] |= e.str.contains("wide"); d["noball"] |= e.str.contains("no")
    wt = _col(bbb, "wicket_type"); wt = wt.fillna("").astype(str).str.lower() if wt is not None else pd.Series("", index=d.index)
    iw = _col(bbb, "is_wicket")
    d["wkt"] = _truthy(iw) if iw is not None else (_truthy(_col(bbb, "player_out")) if _col(bbb, "player_out") is not None else wt.ne("") & wt.ne("nan"))
    d["bowler_wkt"] = d["wkt"] & ~wt.str.contains(NON_BOWLER)
    d["bat_out"] = d["wkt"] & ~wt.str.contains("run out|retired hurt|retired not")
    byes, lb = _col(bbb, "byes"), _col(bbb, "legbyes")
    d["bowl_runs"] = d["total_runs"] - (pd.to_numeric(byes, errors="coerce").fillna(0) if byes is not None else 0) \
                     - (pd.to_numeric(lb, errors="coerce").fillna(0) if lb is not None else 0)
    d["_i"] = np.arange(len(d))
    o, b = _col(bbb, "over"), _col(bbb, "ball")
    d["_o"] = pd.to_numeric(o, errors="coerce").fillna(0) if o is not None else 0
    d["_b"] = pd.to_numeric(b, errors="coerce").fillna(0) if b is not None else 0
    d = d.dropna(subset=["match_id", "innings", "batter", "bowler"]); d = d[d.innings.isin([1, 2])]
    d = d.sort_values(["match_id", "innings", "_o", "_b", "_i"], kind="stable")
    # match context
    m = matches.merge(comps[["comp_id", "match_format", "season"]], on="comp_id", how="left")
    m = m.set_index("match_id")
    d["fmt"] = d.match_id.map(m["match_format"]).fillna("T20"); d["date"] = d.match_id.map(pd.to_datetime(m["match_date"], errors="coerce"))
    d["ko"] = d.match_id.map(m["knock_out"]).fillna(0).astype(float)
    # state before each delivery
    d["legal"] = (~d.wide & ~d.noball).astype(int)
    g = d.groupby(["match_id", "innings"])
    d["balls_before"] = g["legal"].cumsum() - d["legal"]
    d["runs_before"] = g["total_runs"].cumsum() - d["total_runs"]
    d["wkts_before"] = g["wkt"].cumsum().astype(int) - d["wkt"].astype(int)
    inn1 = d[d.innings == 1].groupby("match_id")["total_runs"].sum()
    d["target"] = np.where(d.innings == 2, d.match_id.map(inn1) + 1, np.nan)
    rt = d.match_id.map(m["revised_target"]) if "revised_target" in m else np.nan
    d["target"] = np.where((d.innings == 2) & pd.Series(rt, index=d.index).notna(), rt, d["target"])
    ro = d.match_id.map(m["second_inn_revised_overs"]) if "second_inn_revised_overs" in m else pd.Series(np.nan, index=d.index)
    r1 = d.match_id.map(m["first_inn_revised_overs"]) if "first_inn_revised_overs" in m else pd.Series(np.nan, index=d.index)
    d["max_balls"] = np.where(d.innings == 2, ro.fillna(20), r1.fillna(20)) * 6
    d["balls_left"] = (d.max_balls - d.balls_before).clip(lower=1)
    d["phase"] = pd.cut(d.balls_before, [-1, 35, 89, 10**4], labels=["PP", "MID", "DEATH"]).astype(str)
    # ---- Pressure index (0..1) ----
    stage = (d.balls_before / d.max_balls).clip(0, 1) ** 1.5
    wk = (d.wkts_before / 8).clip(0, 1)
    crr = np.where(d.balls_before > 0, d.runs_before / d.balls_before.clip(lower=1) * 6, np.nan)
    need = d.target - d.runs_before
    rrr = need / d.balls_left * 6
    gap = ((rrr - np.nan_to_num(crr, nan=0)) / 6).clip(0, 1)
    chase = 0.45 * (0.5 * ((rrr - 7) / 8).clip(0, 1) + 0.5 * gap) + 0.30 * wk + 0.25 * stage
    behind = np.where(d.balls_before >= 12, ((8.5 - np.nan_to_num(crr, nan=8.5)) / 4).clip(0, 1), 0)
    setb = 0.45 * wk + 0.35 * stage + 0.20 * behind
    d["pi"] = np.where(d.innings == 2, chase, setb) + 0.08 * d.ko
    d.loc[(d.innings == 2) & (need <= 0), "pi"] = 0
    thr = d.groupby(["fmt", "innings"])["pi"].transform(lambda s: s.quantile(q))
    d["pressure"] = d.pi >= thr
    return d

def _expect(d, rows, runs, out):
    key = ["fmt", "innings", "phase", "pressure"]
    x = d.loc[rows]
    return x.groupby(key)[runs].transform("mean"), x.groupby(key)[out].transform("mean")

def _rate(side, runs, balls):
    return runs / balls * (100 if side == "bat" else 6)

def score_side(d, side, W, min_balls, min_matches, K):
    bat = side == "bat"
    x = d[d.legal == 1].copy() if bat else d.copy()
    pid, rcol, ocol = ("batter", "bat_runs", "bat_out") if bat else ("bowler", "bowl_runs", "bowler_wkt")
    key = ["fmt", "innings", "phase", "pressure"]
    x["er"] = x.groupby(key)[rcol].transform("mean"); x["eo"] = x.groupby(key)[ocol].transform("mean")
    x["v"] = ((x[rcol] - x.er) - W * (x[ocol].astype(float) - x.eo)) * (1 if bat else -1)
    P, N = x[x.pressure], x[~x.pressure]
    ag = lambda s: s.groupby(pid).agg(n=("v", "size"), v=("v", "sum"), runs=(rcol, "sum"), ev=(ocol, "sum"), m=("match_id", "nunique"))
    p, n = ag(P), ag(N)
    r = p.join(n, rsuffix="_b", how="left").fillna(0)
    r = r[(r.n >= min_balls) & (r.m >= min_matches)].copy()
    if r.empty: return r, 0.0
    r["p_rate"] = _rate(side, r.runs, r.n); r["b_rate"] = _rate(side, r.runs_b, r.n_b.where(r.n_b > 0))
    r["v100"] = r.v / (r.n + K) * 100
    r["delta"] = r.v100 - r.v_b / (r.n_b + K) * 100
    pm = P.groupby([pid, "match_id"]).agg(n=("v", "size"), v=("v", "sum")).reset_index()
    pm = pm[pm.n >= 4]; pm["pos"] = (pm.v >= 0).astype(int)
    c = pm.groupby(pid).agg(mm=("pos", "size"), pos=("pos", "sum"))
    r["consistency"] = ((c.pos + 2) / (c.mm + 4)).reindex(r.index).fillna(.5)
    # split-half reliability (odd vs even matches) on eligible players
    rk = pd.Series(pd.factorize(P.match_id, sort=True)[0], index=P.index)
    h = P[P[pid].isin(r.index)].assign(par=rk % 2).groupby([pid, "par"])["v"].sum().unstack()
    rel = float(h.corr(method="spearman").iloc[0, 1]) if h.shape[1] == 2 and len(h) > 5 else float("nan")
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    r["score"] = 0.6 * z(r.v100) + 0.2 * z(r.delta) + 0.2 * z(r.consistency)
    return r.sort_values("score", ascending=False), rel

def run(frames, W=15.0, q=0.75, min_balls=60, min_matches=6, K=60):
    bbb = pd.concat(frames["bbb"], ignore_index=True)
    d = build_balls(bbb, frames["matches"], frames["competitions"], frames["mapping"], W, q)
    mp = frames["mapping"]
    names = mp.drop_duplicates("player_id", keep="last").set_index("player_id")
    nm, tm = names["player_name"], names["team_name"]
    avg_order = None
    if "playing_eleven" in frames:
        pe = frames["playing_eleven"]; avg_order = pe[pe.batting_order > 0].groupby("player_id")["batting_order"].mean()
    out = {}
    for side in ("bat", "bowl"):
        r, rel = score_side(d, side, W, min_balls, min_matches, K)
        rows = []
        for rank, (pid, x) in enumerate(r.iterrows(), 1):
            rows.append(dict(rank=rank, player_id=int(pid), name=nm.get(pid, str(int(pid))), team=tm.get(pid, ""),
                pressure_balls=int(x.n), matches=int(x.m), pressure_rate=round(x.p_rate, 2), baseline_rate=None if pd.isna(x.b_rate) else round(x.b_rate, 2),
                events=int(x.ev), value_per_100=round(x.v100, 2), delta=round(x.delta, 2), consistency=round(x.consistency, 3), score=round(x.score, 3),
                usual_position=None if avg_order is None or pid not in avg_order.index else round(float(avg_order[pid]), 1)))
        out[side] = dict(rows=rows, reliability=None if pd.isna(rel) else round(rel, 3))
    ctx = []
    for inn in (1, 2):
        for flag in (True, False):
            s = d[(d.innings == inn) & (d.pressure == flag) & (d.legal == 1)]
            if len(s): ctx.append(dict(innings=inn, pressure=flag, balls=len(s), strike_rate=round(s.bat_runs.mean() * 100, 1),
                                       dismissal_per_100=round(s.bat_out.mean() * 100, 2), avg_pi=round(s.pi.mean(), 3)))
    out["context"] = ctx
    out["meta"] = dict(deliveries=len(d), matches=int(d.match_id.nunique()), pressure_share=round(float(d.pressure.mean()), 3),
        formats=d.fmt.value_counts().to_dict(), params=dict(wicket_runs=W, pressure_percentile=q, min_balls=min_balls, min_matches=min_matches, shrink_k=K),
        date_from=str(d.date.min())[:10], date_to=str(d.date.max())[:10])
    return out
