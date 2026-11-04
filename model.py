import os
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


def get_simmilar_yandex(song_name: str, artist_name: str, limit: int = 15) -> pd.DataFrame:
    """Ищет трек и получает похожие через неофициальный API Яндекс Музыки.

    Работает без токена, если сервер разрешает анонимный доступ. Возвращает
    столбцы get_similar_lastfm: name, artist, match, playcount, в том числе
    для пустого результата. match = 1 / позиция в итоговом списке (не оценка Яндекса
    и не Last.fm match); playcount = 0 означает, что статистика неизвестна.
    При ошибке API выводит причину и возвращает пустую таблицу.
    """
    columns = ["name", "artist", "match", "playcount"]
    empty = pd.DataFrame(columns=columns)
    if limit <= 0 or not all(isinstance(s, str) and s.strip()
                             for s in (song_name, artist_name)):
        return empty

    def fetch(path: str, **params) -> dict:
        response = requests.get(
            "https://api.music.yandex.net" + path, params=params, timeout=20
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("API вернул неожиданный формат JSON")
        if data.get("error"):
            raise ValueError(f"ошибка API: {data['error']}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise ValueError("в ответе API нет объекта result")
        return result

    def artist_names(track: dict) -> list[str]:
        artists = track.get("artists") or []
        if not isinstance(artists, list):
            return []
        return [a["name"] for a in artists
                if isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"].strip()]

    try:
        result = fetch("/search", text=f"{artist_name} {song_name}",
                       type="track", page=0, nocorrect="true")
        search = result.get("tracks") or {}
        if not isinstance(search, dict):
            raise ValueError("API вернул неожиданный формат результатов поиска")
        search_tracks = search.get("results") or []
        if not isinstance(search_tracks, list):
            raise ValueError("API вернул результаты поиска вместо списка")
        title_key, artist_key = _norm(song_name), _norm(artist_name)
        target = next((track for track in search_tracks
                       if isinstance(track, dict) and track.get("id") is not None
                       and isinstance(track.get("title"), str)
                       and _norm(track["title"]) == title_key
                       and artist_key in map(_norm, artist_names(track))), None)
        if target is None:
            print(f"Яндекс Музыка: трек не найден: {artist_name} — {song_name}")
            return empty

        result = fetch(f"/tracks/{target['id']}/similar")
        similar = result.get("similarTracks") or []
        if not isinstance(similar, list):
            raise ValueError("API вернул похожие треки вместо списка")
        rows, seen = [], set()
        for track in similar:
            if not isinstance(track, dict):
                continue
            name, artists = track.get("title"), artist_names(track)
            if not isinstance(name, str) or not name.strip() or not artists:
                continue
            if (str(track.get("id")) == str(target["id"])
                    or (_norm(name) == title_key and artist_key in map(_norm, artists))):
                continue
            version = track.get("version")
            if isinstance(version, str) and version.strip():
                name = f"{name} - {version}"
            key = (name.casefold(), artists[0].casefold())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"name": name, "artist": artists[0],
                         "match": 1.0 / (len(rows) + 1), "playcount": 0})
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows, columns=columns)
    except (requests.RequestException, ValueError) as error:
        print(f"Яндекс Музыка: {type(error).__name__}: {error}")
        return empty


AUDIO_FEATURES = ["acousticness", "danceability", "energy", "instrumentalness",
                  "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

def track_key(df: pd.DataFrame, name_col: str, artist_col: str) -> pd.Series:
    # "название | артист" без регистра и feat./remaster — разные релизы одного трека совпадут
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
            _rec = get_similar_lastfm(i.name, i.artist, limit=100)
            if _rec.empty:
                print(f"lastfm similar search: {cnt} - empty")
                _rec = get_simmilar_yandex(i.name, i.artist, limit=100)
            df = pd.concat([df, _rec])

        df = df.groupby(["name", "artist"], as_index=False).agg(
            match=("match", "max"), match_mean=("match", "mean"),
            n_seeds=("match", "size"), playcount=("playcount", "first"))

        df = find_spotify_ids(df, name_col="name", artist_col="artist")
        print('done')
        df = df.dropna(subset=["spotify_id"])
        df = df[~track_key(df, "name", "artist").isin(track_key(similar_extraction, "name", "artist"))]
        df.to_csv("soso.csv", index=False)
        print(df)

        df_emb = count_artist_songs_match(liked=similar_extraction, recomendations=df)
        df_emb["liked"] = track_key(df, "name", "artist").isin(track_key(check, "name", "artist"))
        df_emb = df_emb.dropna()
        df_emb.to_csv("data/recomendation_embeddings.csv", index=False)
    if len(sys.argv)>1 and sys.argv[1] == "learn":
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
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
