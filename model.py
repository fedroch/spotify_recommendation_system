import os
import re
import unicodedata
from urllib.parse import quote
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import requests
from dotenv import load_dotenv
from api_kernel import get_reccobeats_features, _norm, get_liked_songs, get_recently_played
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

load_dotenv()
LASTFM_KEY = os.environ["api_key_lastfm"]
YANDEX_TOKEN = os.environ["yandex_music_token"]

def get_similar_recobeats(song_ids: list, size: int = 10, **features) -> pd.DataFrame:
    tracks_total = pd.DataFrame()
    for i in range(0,len(song_ids),5):
        params = {"seeds": ','.join(song_ids[i:i+5]), "size": size, **features}
        r = requests.get("https://api.reccobeats.com/v1/track/recommendation", params=params)
        if r.status_code != 200:
                print(r.status_code, r.text)
                continue
        r.raise_for_status()
        tracks = pd.DataFrame(r.json().get("content", []))
        if tracks.empty: continue
        tracks["spotify_id"] = tracks["href"].str.rsplit("/").str[-1]
        tracks["artists"] = tracks["artists"].apply(lambda x: x[0]["name"])
        tracks.drop(columns=["href", "id", "durationMs", "ean", "upc", "availableCountries"], inplace=True)
        tracks_total = pd.concat([tracks_total, tracks])
    return tracks_total

def get_similar_lastfm(song_name: str, artist_name: str, limit: int = 15) -> pd.DataFrame:
    params = {"method": "track.getSimilar", "api_key": LASTFM_KEY, "track": song_name, "artist": artist_name, "format": "json", "autocorrect": 1, "limit": limit}
    r = requests.get("https://ws.audioscrobbler.com/2.0/", params=params)
    r.raise_for_status()
    if "error" in r.json():
        print(r.json())
        return pd.DataFrame()
    items = r.json().get("similartracks", {}).get("track", [])
    items = items if isinstance(items, list) else [items]
    return pd.DataFrame([{"name": t["name"], "artist": t["artist"]["name"],
                          "match": float(t["match"]), "playcount": int(t["playcount"])}
                         for t in items])


def _yandex_name_key(value: str) -> str:
    """Сопоставляет регистр, пунктуацию, пробелы и простую транслитерацию."""
    transliteration = str.maketrans(dict(zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k",
         "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts",
         "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"],
    )))
    value = unicodedata.normalize("NFKC", value).casefold().translate(transliteration)
    return "".join(char for char in value if char.isalnum())


def _yandex_without_feature(value: str) -> str:
    return re.sub(r"\s*[\(\[]?\s*\b(?:feat\.?|ft\.?|featuring)\s+.*$",
                  "", value, flags=re.IGNORECASE).strip()


def _yandex_request(method: str, path: str, token: str | None = YANDEX_TOKEN, **kwargs) -> dict:
    request = requests.get if method == "GET" else requests.post
    headers = {"Authorization": f"OAuth {token}"} if token else {}
    response = request("https://api.music.yandex.net" + path,
                       headers=headers, timeout=20, **kwargs)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(str(data["error"]))
    return data["result"]


def find_yandex_track_id(song_name: str, artist_name: str, *,
                         token: str | None = YANDEX_TOKEN, max_pages: int = 3) -> str | None:
    """Ищет ID по названию и артисту; нормализация, запасные запросы и пагинация."""
    title = _yandex_without_feature(song_name)
    exact = _yandex_name_key(title)
    base = _yandex_name_key(_norm(title))
    artist = _yandex_name_key(artist_name)
    queries = dict.fromkeys([f"{artist_name} {song_name}", f"{artist_name} {_norm(title)}", song_name])
    fallback = None
    for query in queries:
        for page in range(max_pages):
            result = _yandex_request("GET", "/search", token, params={
                "text": query, "type": "track", "page": page, "nocorrect": "false",
            })
            search = result.get("tracks") or {}
            tracks = search.get("results") or []
            for track in tracks:
                if not track.get("id") or not track.get("title"):
                    continue
                artists = [_yandex_name_key(a["name"]) for a in (track.get("artists") or [])]
                if artist not in artists:
                    continue
                name = _yandex_without_feature(track["title"])
                full_name = f"{name} {track.get('version') or ''}"
                if _yandex_name_key(full_name) == exact:
                    return str(track["id"])
                if _yandex_name_key(_norm(full_name)) == base:
                    fallback = fallback or str(track["id"])
            per_page, total = search.get("perPage") or len(tracks), search.get("total")
            if not tracks or (total is not None and (page + 1) * per_page >= total):
                break
    return fallback


def get_yandex_radio(track_id: str, limit: int = 15, *, token: str | None = YANDEX_TOKEN,
                     max_batches: int = 5) -> pd.DataFrame:
    """Получает радио по ID. match = 1 / позиция; playcount = 0 — нет статистики."""
    columns = ["name", "artist", "match", "playcount"]
    if limit <= 0 or max_batches <= 0:
        return pd.DataFrame(columns=columns)
    queue, songs = [str(track_id)], {}
    seen = set(queue)
    result = _yandex_request("POST", "/rotor/session/new", token, json={
        "seeds": [f"track:{track_id}"], "queue": queue.copy(),
        "includeTracksInResponse": True, "incognito": True,
    })
    session_id = quote(result["radioSessionId"], safe="")
    for batch in range(max_batches):
        if result.get("unknownSession"):
            raise ValueError("Радиосессия Яндекса не найдена")
        count_before = len(seen)
        for item in result.get("sequence") or []:
            track = item.get("track")
            if item.get("type", "track") != "track" or not track or not track.get("id"):
                continue
            identifier = str(track["id"])
            if identifier in seen:
                continue
            seen.add(identifier)
            queue.append(identifier)
            if not track.get("title") or not track.get("artists"):
                continue
            name = track["title"]
            if track.get("version"):
                name += f" - {track['version']}"
            artist = track["artists"][0]["name"]
            songs.setdefault((_yandex_name_key(name), _yandex_name_key(artist)), (name, artist))
            if len(songs) >= limit:
                break
        if len(songs) >= limit or result.get("terminated") or (batch > 0 and len(seen) == count_before):
            break
        if batch + 1 < max_batches:
            result = _yandex_request("POST", f"/rotor/session/{session_id}/tracks", token,
                                    json={"queue": queue.copy()})
    frame = pd.DataFrame(songs.values(), columns=["name", "artist"]).reset_index(drop=True)
    frame["match"] = 1.0 / np.arange(1, len(frame) + 1)
    frame["playcount"] = 0
    return frame[columns]


def get_simmilar_yandex(song_name: str, artist_name: str, limit: int = 15, *, max_pages: int = 3, token: str | None = YANDEX_TOKEN, max_batches: int = 5) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["name", "artist", "match", "playcount"])
    if not song_name or not artist_name or limit <= 0:
        return empty
    try:
        track_id = find_yandex_track_id(song_name, artist_name, token=token, max_pages=max_pages)
        if track_id is None:
            print(f"Яндекс Музыка: не удалось сопоставить {artist_name} — {song_name}")
            return empty
        frame = get_yandex_radio(track_id, limit, token=token, max_batches=max_batches)
        if frame.empty:
            print(f"Яндекс Музыка: радио по треку {track_id} не вернуло кандидатов")
        return frame
    except (requests.RequestException, ValueError, KeyError) as error:
        response = getattr(error, "response", None)
        message = f"HTTP {response.status_code}: {response.text[:500]}" if response is not None else str(error)
        print("Яндекс Музыка:", message.replace(token, "[REDACTED]") if token else message)
        return empty


AUDIO_FEATURES = ["acousticness", "danceability", "energy", "instrumentalness",
                  "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

def track_key(df: pd.DataFrame, name_col: str, artist_col: str) -> pd.Series:
    return df[name_col].map(_norm) + " | " + df[artist_col].map(_norm)

def count_artist_songs_match(liked: pd.DataFrame, recomendations: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(index=recomendations.index)
    res["artist_match"] = recomendations["artist"].isin(liked["artist_name"])
    res["match"] = recomendations["match"]
    res["match_mean"] = recomendations["match_mean"]
    res["n_seeds"] = recomendations["n_seeds"]
    liked_f = get_reccobeats_features(liked["id"].to_list())[AUDIO_FEATURES]
    mu, sd = liked_f.mean(), liked_f.std() + 1e-8
    rec_f = get_reccobeats_features(recomendations["spotify_id"].to_list()).set_index("spotify_id")[AUDIO_FEATURES]
    z = (rec_f - mu) / sd
    dist = np.sqrt((z ** 2).sum(axis=1))
    res["liked_distance"] = recomendations["spotify_id"].map(dist)
    res["id"] = recomendations["spotify_id"]
    return res


from api_kernel import find_spotify_ids
import sys

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "dataset":
        songs = pd.read_csv("data/tracks.csv")
        # print(get_similar_recobeats(["4VCOQxobkSUXQ93sFT0O8X"]).info())
        # print(get_similar_recobeats(["4VCOQxobkSUXQ93sFT0O8X"]).shape[0])
        # print(get_similar_lastfm("Shape of You", "Ed Sheeran"))

        similar_extraction, check = train_test_split(songs, train_size=0.3, random_state=42)
        # check.to_csv("data/liked.csv") лишнее
        print(len(similar_extraction), len(check))
        # print(type(similar_extraction))
        df = pd.DataFrame()
        cnt=0
        for i in similar_extraction.itertuples():
            cnt+=1
            print(f"lastfm similar search: {cnt}")
            _rec = get_similar_lastfm(i.name, i.artist_name, limit=20)
            if _rec.empty:
                print(f"lastfm similar search: {cnt} - empty")
            _rec_y = get_simmilar_yandex(i.name, i.artist_name, limit=20)
            _rec_full = pd.concat([_rec, _rec_y])
            df = pd.concat([df, _rec_full])

        df = df.groupby(["name", "artist"], as_index=False).agg(
            match=("match", "max"), match_mean=("match", "mean"),
            n_seeds=("match", "size"), playcount=("playcount", "first"))

        df = find_spotify_ids(df, name_col="name", artist_col="artist")
        print('done')
        df = df.dropna(subset=["spotify_id"])
        df = df[~track_key(df, "name", "artist").isin(track_key(similar_extraction, "name", "artist_name"))]
        df.to_csv("soso.csv", index=False)
        print(df)

        df_emb = count_artist_songs_match(liked=similar_extraction, recomendations=df)
        df_emb["liked"] = track_key(df, "name", "artist").isin(track_key(check, "name", "artist_name"))
        df_emb = df_emb.dropna()
        df_emb.to_csv("data/recomendation_embeddings.csv", index=False)
    if len(sys.argv)>1 and sys.argv[1] == "learn":
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', solver = "newton-cholesky"))
        df = pd.read_csv("data/recomendation_embeddings.csv")
        X = df.drop(columns=["id", "liked"])
        Y = df["liked"]
        model.fit(X,Y)
        joblib.dump(model, 'model.pkl')
        print(accuracy_score(Y, model.predict(X)))
        print(roc_auc_score(Y, model.predict_proba(X)[:,1]))
        print(f1_score(Y, model.predict(X)))
    else:
        pass
