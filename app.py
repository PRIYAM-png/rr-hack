"""Run: python app.py  ->  http://localhost:5000"""
import io, csv, pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response
import model

import os
BASE = os.path.dirname(os.path.abspath(__file__))
# works whether index.html sits next to app.py or inside a static/ folder
WEB = BASE if os.path.exists(os.path.join(BASE, "index.html")) else os.path.join(BASE, "static")
app = Flask(__name__, static_folder=None); app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024
LAST = {}
REQUIRED = {"bbb": "ball-by-ball", "matches": "matches", "competitions": "competitions", "mapping": "competition-team-player mapping"}

@app.get("/")
def home(): return send_from_directory(WEB, "index.html")

@app.post("/api/analyze")
def analyze():
    frames, seen = {"bbb": []}, []
    for f in request.files.getlist("files"):
        try: df = model.norm(pd.read_csv(f))
        except Exception as e: return jsonify(error=f"Could not read {f.filename}: {e}"), 400
        kind = model.classify(df); seen.append(dict(file=f.filename, detected=kind, rows=len(df)))
        if kind == "bbb": frames["bbb"].append(df)
        elif kind: frames[kind] = df
    missing = [v for k, v in REQUIRED.items() if (not frames["bbb"] if k == "bbb" else k not in frames)]
    if missing: return jsonify(error="Missing required file(s): " + ", ".join(missing), files=seen), 400
    try:
        g = lambda k, t, dflt: t(request.form.get(k, dflt))
        res = model.run(frames, W=g("wicket_runs", float, 15), q=g("pressure_pct", float, 75) / 100,
                        min_balls=g("min_balls", int, 60), min_matches=g("min_matches", int, 6))
    except ValueError as e: return jsonify(error=str(e), files=seen), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=f"Analysis failed: {type(e).__name__}: {e}. Check that the ball-by-ball file has batter, bowler, innings, match_id and runs columns.", files=seen), 500
    res["files"] = seen; LAST.update(res)
    return jsonify(res)

@app.get("/api/download/<side>.csv")
def dl(side):
    rows = LAST.get(side, {}).get("rows")
    if not rows: return "Run an analysis first", 404
    s = io.StringIO(); w = csv.DictWriter(s, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return Response(s.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={side}_pressure_rankings.csv"})

if __name__ == "__main__":
    print("Open http://localhost:5000 in your browser (do not open index.html directly)")
    app.run(debug=False, port=5000)
