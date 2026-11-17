# Clutch Index – pressure-performance dashboard
## Run
    pip install -r requirements.txt
    python app.py            # open http://localhost:5000
Drop all CSVs onto the page and click **Run analysis**.

## Required files
- Ball-by-ball CSV(s) (one or many files; one row per delivery). Column names are auto-matched, e.g. match_id, innings, over, ball,
  batter/batsman/striker, bowler, batter_runs/runs_off_bat, total_runs, wides, noballs, is_wicket/wicket_type (ids or player names both work).
- matches.csv, competitions.csv, competition_team_player_mapping.csv
- Optional: playing_eleven.csv (shows usual batting position), teams.csv

## Test without real ball-by-ball data
    python make_demo_data.py /path/to/csv/folder    # SYNTHETIC data, for checking the pipeline only. Its rankings are meaningless.

## Files
- model.py: pressure index, expected-runs baselines, shrinkage, scoring (edit weights here)
- app.py: Flask API (/api/analyze, /api/download/<bat|bowl>.csv)
- static/index.html: dashboard UI
