from flask import Flask, render_template, request, jsonify, Response
import pickle
import numpy as np
import os
import json
import csv
import io
from urllib.parse import urlparse
from flask_cors import CORS

SKLEARN_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)
CORS(app)

DATA_FILE = "bad_word_counts.json"

class DummyVectorizer:
    def transform(self, data):
        return np.zeros((len(data), 1))

class DummyModel:
    def predict_proba(self, X):
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)) * 0.5, np.ones((n, 1)) * 0.5])


def load_pickle_or_dummy(path, kind="model"):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: failed loading {path}: {e}")

    if kind == "vect":
        if SKLEARN_AVAILABLE:
            return TfidfVectorizer()
        return DummyVectorizer()
    else:
        if SKLEARN_AVAILABLE:
            return RandomForestClassifier()
        return DummyModel()


def load_tracking_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: failed loading {DATA_FILE}: {e}")
    return {"pages": {}}


def save_tracking_data(data):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: failed saving {DATA_FILE}: {e}")


def normalize_url(url):
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    return url, parsed.netloc or "unknown"

# vectorizers
tox = load_pickle_or_dummy("toxic_vect.pkl", kind="vect")
sev = load_pickle_or_dummy("severe_toxic_vect.pkl", kind="vect")
obs = load_pickle_or_dummy("obscene_vect.pkl", kind="vect")
ins = load_pickle_or_dummy("insult_vect.pkl", kind="vect")
thr = load_pickle_or_dummy("threat_vect.pkl", kind="vect")
ide = load_pickle_or_dummy("identity_hate_vect.pkl", kind="vect")

# models
tox_model = load_pickle_or_dummy("toxic_model.pkl", kind="model")
sev_model = load_pickle_or_dummy("severe_toxic_model.pkl", kind="model")
obs_model = load_pickle_or_dummy("obscene_model.pkl", kind="model")
ins_model = load_pickle_or_dummy("insult_model.pkl", kind="model")
thr_model = load_pickle_or_dummy("threat_model.pkl", kind="model")
ide_model = load_pickle_or_dummy("identity_hate_model.pkl", kind="model")


@app.route("/")
def home():
    return render_template('index_toxic.html')


@app.route("/dashboard")
def dashboard():
    tracking_data = load_tracking_data()
    pages = list(tracking_data.get("pages", {}).values())

    domain_counts = {}
    bad_word_counts = {}
    for page in pages:
        domain_counts[page["domain"]] = domain_counts.get(page["domain"], 0) + page.get("badWordCount", 0)
        for word, count in page.get("badWords", {}).items():
            bad_word_counts[word] = bad_word_counts.get(word, 0) + count

    top_domains = sorted(domain_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    top_bad_words = sorted(bad_word_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    labels = [item[0] for item in top_domains]
    values = [item[1] for item in top_domains]
    total_bad_words = sum(values)
    total_pages = len(pages)
    average_bad_words = round(total_bad_words / total_pages, 2) if total_pages else 0
    most_flagged_word = top_bad_words[0] if top_bad_words else ("N/A", 0)

    return render_template(
        'dashboard.html',
        pages=pages,
        labels=json.dumps(labels),
        values=json.dumps(values),
        total_bad_words=total_bad_words,
        top_domains=top_domains,
        top_bad_words=top_bad_words,
        average_bad_words=average_bad_words,
        most_flagged_word=most_flagged_word,
    )


@app.route("/dashboard/export")
def export_dashboard():
    tracking_data = load_tracking_data()
    pages = list(tracking_data.get("pages", {}).values())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Page", "Domain", "Bad Words", "Visits", "Top Bad Words"])

    for page in pages:
        top_words = page.get("badWords", {})
        top_summary = "; ".join(
            f"{word}:{count}" for word, count in sorted(top_words.items(), key=lambda item: item[1], reverse=True)[:5]
        )
        writer.writerow([
            page.get("title") or page.get("url"),
            page.get("domain"),
            page.get("badWordCount", 0),
            page.get("visitCount", 0),
            top_summary,
        ])

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=toxic_dashboard_report.csv"
    return response


@app.route("/predict", methods=['POST'])
def predict():
    user_input = request.form['text']
    data = [user_input]

    vects = [tox, sev, obs, ins, thr, ide]
    models = [tox_model, sev_model, obs_model, ins_model, thr_model, ide_model]

    preds = []
    for vect, model in zip(vects, models):
        v = vect.transform(data)
        p = model.predict_proba(v)[:, 1]
        preds.append(round(p[0], 2))

    return render_template('index_toxic.html',
                           pred_tox=f"Prob (Toxic): {preds[0]}",
                           pred_sev=f"Prob (Severe Toxic): {preds[1]}",
                           pred_obs=f"Prob (Obscene): {preds[2]}",
                           pred_ins=f"Prob (Insult): {preds[3]}",
                           pred_thr=f"Prob (Threat): {preds[4]}",
                           pred_ide=f"Prob (Identity Hate): {preds[5]}")


@app.route("/api/predict", methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "no text provided"}), 400

    inputs = [text]

    scores = {
        "toxic": float(tox_model.predict_proba(tox.transform(inputs))[:,1][0]),
        "severe_toxic": float(sev_model.predict_proba(sev.transform(inputs))[:,1][0]),
        "obscene": float(obs_model.predict_proba(obs.transform(inputs))[:,1][0]),
        "insult": float(ins_model.predict_proba(ins.transform(inputs))[:,1][0]),
        "threat": float(thr_model.predict_proba(thr.transform(inputs))[:,1][0]),
        "identity_hate": float(ide_model.predict_proba(ide.transform(inputs))[:,1][0]),
    }

    return jsonify(scores)


@app.route("/api/report", methods=['POST'])
def api_report():
    payload = request.get_json() or {}
    page_url = payload.get("pageUrl") or payload.get("url") or payload.get("page_url")
    bad_word_count = payload.get("badWordCount")
    bad_words = payload.get("badWords", [])
    page_title = payload.get("pageTitle", payload.get("title", ""))

    if not page_url:
        return jsonify({"error": "pageUrl required"}), 400

    page_url, domain = normalize_url(page_url)
    try:
        bad_word_count = int(bad_word_count or 0)
    except (TypeError, ValueError):
        bad_word_count = 0

    tracking_data = load_tracking_data()
    pages = tracking_data.setdefault("pages", {})
    page_record = pages.setdefault(page_url, {
        "url": page_url,
        "domain": domain,
        "title": page_title,
        "badWordCount": 0,
        "visitCount": 0,
        "badWords": {},
    })

    page_record["title"] = page_title or page_record.get("title", "")
    page_record["domain"] = domain
    page_record["badWordCount"] = bad_word_count
    page_record["visitCount"] = page_record.get("visitCount", 0) + 1

    existing_words = page_record.setdefault("badWords", {})
    for word in bad_words:
        existing_words[word] = existing_words.get(word, 0) + 1

    pages[page_url] = page_record
    save_tracking_data(tracking_data)

    return jsonify({"status": "updated", "page": page_record})


if __name__ == "__main__":
    app.run(debug=True)
