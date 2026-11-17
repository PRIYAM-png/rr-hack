"""Creates demo_ball_by_ball.csv (SYNTHETIC, for testing the pipeline only) from real matches + playing XIs."""
import numpy as np, pandas as pd, sys
src = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(7)
m = pd.read_csv(f"{src}/matches.csv"); pe = pd.read_csv(f"{src}/playing_eleven.csv")
pe = pe[pe.is_played.astype(str).str.lower() == "true"]
rows = []
for r in m.itertuples():
    t = pe[pe.match_id == r.match_id]
    teams = [r.home_team_id, r.away_team_id]
    if t.team_id.nunique() < 2: continue
    first = r.toss_win_team_id if str(r.toss_opted).lower().startswith("bat") else (set(teams) - {r.toss_win_team_id}).pop()
    order = [first, (set(teams) - {first}).pop()]; target = None
    for inn, bt in enumerate(order, 1):
        bat = t[t.team_id == bt].sort_values("batting_order").player_id.tolist(); bowl = t[t.team_id != bt].player_id.tolist()[-6:]
        runs = wk = 0; legal = 0; striker = 0
        while legal < 120 and wk < 10 and striker < len(bat) and not (target and runs >= target):
            o, b = divmod(legal, 6); w = rng.random() < .03; x = rng.choice([0, 1, 2, 3, 4, 6], p=[.35, .33, .08, .02, .15, .07])
            out = rng.random() < .045
            rows.append(dict(match_id=r.match_id, innings=inn, over=o, ball=b + 1, batter=bat[min(striker, len(bat)-1)], bowler=bowl[o % len(bowl)],
                             batter_runs=0 if w else x, total_runs=(1 if w else x), wides=int(w), noballs=0, is_wicket=int(out and not w), wicket_type="bowled" if out and not w else ""))
            runs += 1 if w else x
            if not w: legal += 1
            if out and not w: wk += 1; striker = wk + 1
        if inn == 1: target = runs + 1
pd.DataFrame(rows).to_csv("demo_ball_by_ball.csv", index=False); print(len(rows), "rows")
