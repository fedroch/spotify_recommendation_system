from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
import base64
import os
import re
import time
import requests
from urllib.parse import urlencode
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

SCOPES = "user-top-read user-library-read user-read-recently-played playlist-modify-private"
CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("client_secret")
REDIRECT_URI = "http://127.0.0.1:8888/callback"

ID2TOKEN = {} # telegram_id : {"access_token": str, "refresh_token": str, "expires_in": int}

class CallbackResponse(Exception): pass

@app.get("/callback")
def callback(code: str = None, state: str = None, error: str = None):
    if error: raise CallbackResponse
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post("https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "authorization_code",
              "code": code, "redirect_uri": REDIRECT_URI})
    tokens = r.json()
    ID2TOKEN[state] = {'access_token': tokens['access_token'], 'refresh_token': tokens['refresh_token'], 'expires_in': tokens['expires_in']}
    print("Spotify подключён")
    return RedirectResponse("/done")

@app.get("/done")
def done():
    return HTMLResponse("<h1>Spotify подключён</h1><p>Можно закрывать окно</p>")

def refresh_token(telegram_id: str):
    if telegram_id not in ID2TOKEN:
        raise Exception("User not authorized")
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post("https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "refresh_token",
              "refresh_token": ID2TOKEN[telegram_id]['refresh_token']})
    tokens = r.json()
    ID2TOKEN[telegram_id]['access_token'] = tokens['access_token']
    ID2TOKEN[telegram_id]['expires_in'] = tokens['expires_in']
    if "refresh_token" in tokens:
        ID2TOKEN[telegram_id]['refresh_token'] = tokens['refresh_token']

def get_liked_songs(telegram_id: str) -> list:
    token = ID2TOKEN.get(telegram_id)
    if not token:
        raise Exception("User not authorized")
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    tracks = []
    url = "https://api.spotify.com/v1/me/tracks?limit=50"
    while url:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        page = r.json()
        tracks.extend(item["track"] for item in page["items"])
        url = page["next"]
    return tracks

def get_recently_played(telegram_id: str) -> list:
    token = ID2TOKEN.get(telegram_id)
    if not token:
        raise Exception("User not authorized")
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    tracks = []
    url = "https://api.spotify.com/v1/me/player/recently-played?limit=50"
    r = requests.get(url, headers = headers)
    r.raise_for_status()
    page = r.json()
    tracks.extend(item["track"] for item in page["items"])
    return tracks

def make_auth_url(telegram_id: str) -> str:
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": telegram_id,
    }
    return "https://accounts.spotify.com/authorize?" + urlencode(params)

RB_URL = "https://api.reccobeats.com/v1"
LB_URL = "https://labs.api.listenbrainz.org/spotify-id-from-metadata/json"

FEATURES_CACHE = {}  # spotify_id -> dict признаков ReccoBeats или None (ReccoBeats трека не знает)

def _norm(s: str) -> str:
    """Нижний регистр, без feat./remaster/live в скобках и после дефиса, без лишних символов."""
    s = s.lower()
    s = re.sub(r"[\(\[].*?(feat|ft\.|with|remaster|live|version|edit|mix).*?[\)\]]", "", s)
    s = re.sub(r"\s-\s.*(remaster|live|version|edit|mix|acoustic).*$", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def _rb_get(path: str, retries: int = 5, **params) -> dict:
    """GET к ReccoBeats с повторами: он иногда рвёт соединение и отвечает 429
    ("retry after 5 seconds"). При неудаче возвращает {}."""
    for attempt in range(retries):
        try:
            r = requests.get(RB_URL + path, params=params, timeout=20)
        except requests.RequestException as e:
            print("ReccoBeats", path, type(e).__name__)
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(5)
            continue
        print("ReccoBeats", path, r.status_code, r.text[:200])
        if r.status_code in (400, 404):
            return {}
        time.sleep(2 * (attempt + 1))
    return {}

def get_reccobeats_features(spotify_ids: list[str]) -> pd.DataFrame:
    """Аудио-признаки ReccoBeats. Уже запрошенные ID берутся из FEATURES_CACHE.
    Треки, которых ReccoBeats не знает, в результат не попадают."""
    unique_ids = list(dict.fromkeys(spotify_ids))
    todo = [i for i in unique_ids if i not in FEATURES_CACHE]
    for batch in range(0, len(todo), 30):
        ids = todo[batch:batch+30]
        data = _rb_get("/audio-features", ids=",".join(ids))
        if "content" not in data:
            continue
        got = {}
        for f in data["content"]:
            got[f["href"].rsplit("/", 1)[-1]] = {k: v for k, v in f.items() if k not in ("href", "isrc", "id")}
        for i in ids:
            FEATURES_CACHE[i] = got.get(i)
    rows = [{**FEATURES_CACHE[i], "spotify_id": i} for i in unique_ids if FEATURES_CACHE.get(i)]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["spotify_id"])

def _rb_match(t: dict, clean_name: str, clean_artist: str) -> dict | None:
    """Трек из ответа ReccoBeats подходит, если совпал один из артистов и название (после _norm)."""
    if clean_artist not in [_norm(a["name"]) for a in t["artists"]]:
        return None
    title = _norm(t["trackTitle"])
    if not (title == clean_name or title.startswith(clean_name) or clean_name.startswith(title)):
        return None
    return {
        "spotify_id": t["href"].rsplit("/", 1)[-1],
        "name": t["trackTitle"],
        "artist_name": t["artists"][0]["name"],
        "isrc": t.get("isrc"),
        "exact": title == clean_name,
    }

def _rb_scan(path: str, max_pages: int, clean_name: str, clean_artist: str, found: dict, **params):
    """Листает страницы по 50, складывает подходящие треки в found. Стоп на точном совпадении."""
    for page in range(max_pages):
        content = _rb_get(path, size=50, page=page, **params).get("content", [])
        for t in content:
            m = _rb_match(t, clean_name, clean_artist)
            if m:
                found[m["spotify_id"]] = m
        if any(m["exact"] for m in found.values()) or len(content) < 50:
            return

def search_song_reccobeats(song_name: str, artist_name: str,
                           max_pages: int = 4, max_artist_pages: int = 10) -> list:
    """Ищет Spotify ID трека в базе ReccoBeats (без запросов к Spotify).
    Возвращает подходящие треки (точное совпадение названия первым) или [], если не нашлось.
    1) /track/search по названию (ищет по префиксу, только по названию, максимум 200 результатов);
    2) если не нашлось — /artist/search, затем все треки артиста постранично."""
    clean_name = _norm(song_name) or song_name.lower()
    clean_artist = _norm(artist_name) or artist_name.lower()
    found = {}
    short_name = re.split(r" - | \(| \[", song_name)[0].strip()
    for q in dict.fromkeys([song_name, short_name]):
        _rb_scan("/track/search", max_pages, clean_name, clean_artist, found, searchText=q)
        if found:
            break

    if not found:
        for a in _rb_get("/artist/search", searchText=artist_name, size=10).get("content", []):
            if _norm(a["name"]) == clean_artist:
                _rb_scan(f"/artist/{a['id']}/track", max_artist_pages, clean_name, clean_artist, found)
                if found:
                    break

    return sorted(found.values(), key=lambda t: not t["exact"])

def _lb_spotify_ids(pairs: list[tuple[str, str]], batch: int = 250) -> list[list[str]]:
    """ListenBrainz Labs: (артист, название) -> Spotify ID всех релизов трека ([] если не нашёл).
    Ответ приходит в том же порядке, что и запрос."""
    out = []
    for b in range(0, len(pairs), batch):
        chunk = pairs[b:b+batch]
        payload = [{"artist_name": a, "track_name": t, "release_name": ""} for a, t in chunk]
        try:
            r = requests.post(LB_URL, json=payload, timeout=60)
            r.raise_for_status()
            res = r.json()
            if len(res) != len(chunk):
                raise ValueError(f"ответ {len(res)} != запрос {len(chunk)}")
            out += [x.get("spotify_track_ids") or [] for x in res]
        except (requests.RequestException, ValueError) as e:
            print("ListenBrainz", type(e).__name__, e)
            out += [[] for _ in chunk]
    return out

def find_spotify_ids(tracks: pd.DataFrame, name_col: str = "name", artist_col: str = "artist",
                     fallback: bool = True) -> pd.DataFrame:
    """Добавляет к таблице столбцы spotify_id и id_source ("listenbrainz" / "reccobeats" / None).
    1) ListenBrainz пачкой — быстро,
    2) для ненайденных (в основном русские треки) — медленный search_song_reccobeats.
    Признаки попадают в FEATURES_CACHE, повторный get_reccobeats_features по ним бесплатный."""
    keys = [(a, t) if isinstance(a, str) and isinstance(t, str) else None
            for a, t in zip(tracks[artist_col], tracks[name_col])]
    pairs = list(dict.fromkeys(k for k in keys if k))
    lb = {p: ids for p, ids in zip(pairs, _lb_spotify_ids(pairs)) if ids}
    get_reccobeats_features([ids[0] for ids in lb.values()])
    get_reccobeats_features([i for ids in lb.values() if not FEATURES_CACHE.get(ids[0]) for i in ids[1:]])
    chosen = {}
    for p, ids in lb.items():
        with_features = [i for i in ids if FEATURES_CACHE.get(i)]
        chosen[p] = ((with_features or ids)[0], "listenbrainz")
    missing = [p for p in pairs if p not in chosen]
    print(f"ListenBrainz нашёл {len(chosen)}/{len(pairs)}" + (f", ищу в ReccoBeats: {len(missing)}" if fallback else ""))
    cnt_reco =0
    if fallback:
        for a, t in missing:
            cnt_reco+=1
            print(f"recobeats id search {cnt_reco}/{len(missing)}")
            res = search_song_reccobeats(t, a)
            if res:
                chosen[(a, t)] = (res[0]["spotify_id"], "reccobeats")

    out = tracks.copy()
    out["spotify_id"] = [chosen.get(k, (None, None))[0] for k in keys]
    out["id_source"] = [chosen.get(k, (None, None))[1] for k in keys]
    return out



def get_spotify_song_from_id(id: str) -> dict:
    r = requests.get(
    f"https://api.spotify.com/v1/tracks/{id}",
    headers={"Authorization": f"Bearer {ID2TOKEN['12345']['access_token']}"},
    timeout=20,
    )
    r.raise_for_status()
    track = r.json()
    return track["name"], ", ".join(a["name"] for a in track["artists"])
def create_spotify_playlist(name: str, track_ids: list[str], access_token: str) -> str:
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.post("https://api.spotify.com/v1/me/playlists", headers=headers, json={"name": name, "public": False}, timeout=20,)
    r.raise_for_status()
    playlist = r.json()
    uris = [f"spotify:track:{track_id}" for track_id in track_ids]
    for start in range(0, len(uris), 100):
        r = requests.post(f"https://api.spotify.com/v1/playlists/{playlist['id']}/items",headers=headers,json={"uris": uris[start:start + 100]},timeout=20,)
        r.raise_for_status()
    return playlist["external_urls"]["spotify"], playlist['id']


import sys
if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "data":
        print(make_auth_url('12345'))
        import uvicorn
        import threading
        import time
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8888)
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(15) 
        liked = get_liked_songs('12345')
        recent = get_recently_played('12345')
        print(len(liked), len(recent))
        recent_emb = get_reccobeats_features([t["id"] for t in recent])
        Liked_songs_emb = get_reccobeats_features([t["id"] for t in liked])
        Liked_songs_emb["liked"] = 2
        recent_emb["liked"] = 1
        df = pd.concat([Liked_songs_emb, recent_emb], ignore_index=True)
        df = df.drop_duplicates(subset="spotify_id", keep="first")    
        df.to_csv("data/embeddings.csv", index=False)
        all_tracks = [
            {
                "name": t["name"],
                "id": t["id"],
                "artist": t["artists"][0]["name"] if t.get("artists") else None,
                "liked": 2 if t in liked else 1 
            }
            for t in (liked + recent)
        ]
        df2 = pd.DataFrame(all_tracks, columns=["name", "id", "artist", "liked"]).drop_duplicates(subset=["id"])
        print(df2)
        df2.to_csv("data/tracks.csv", index= False)
    if len(sys.argv)>1 and sys.argv[1] == "predict":
        import joblib
        from tqdm import tqdm
        from model import get_similar_lastfm, count_artist_songs_match, track_key, get_simmilar_yandex
        model = joblib.load('model.pkl')
        print(make_auth_url('12345'))
        import uvicorn
        import threading
        import time
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8888)
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(15) 
        liked = get_liked_songs('12345')
        recent = get_recently_played('12345')
        liked_ids = {t["id"] for t in liked}
        tracks = pd.DataFrame([{
            "name": t["name"],
            "id": t["id"],
            "artist": t["artists"][0]["name"] if t.get("artists") else None,
            "liked": 2 if t["id"] in liked_ids else 1
        } for t in (liked + recent)[:100]]) # [:100] нужно чтобы не закончилась квота спотика на запросы 
        tracks = tracks.drop_duplicates(subset="id", keep="first")
        lfm_rec = pd.DataFrame()
        rec = pd.DataFrame()
        for i in tqdm(tracks.itertuples(), total=len(tracks)):
            print(f"lastfm similar search: {i.Index+1}/{len(tracks)}")
            df = get_similar_lastfm(_norm(i.name), _norm(i.artist), limit=10)
            if df.empty: 
                print(f"No recommendations found for {i.name} by {i.artist} lastfm")
                df = get_simmilar_yandex(i.name, i.artist, limit=10)
            df = find_spotify_ids(df, name_col="name", artist_col="artist")
            df = df.dropna(subset=["spotify_id"])
            print(f"df: {df}")
            lfm_rec = pd.concat([lfm_rec,df], ignore_index = True)
        lfm_rec = lfm_rec.groupby(["name", "artist"], as_index=False).agg(
                                match=("match", "max"), match_mean=("match", "mean"),
                                n_seeds=("match", "size"), playcount=("playcount", "first"), spotify_id=("spotify_id", "first"))
        lfm_rec = lfm_rec.dropna()
        if not lfm_rec.empty:
            df_emb = count_artist_songs_match(liked=tracks.rename(columns = {"artist": "artist_name"}), recomendations=lfm_rec)
            df_emb = df_emb.dropna()
            X = df_emb.drop(columns=["id"])
            Y_pred = model.predict(X)
            Y_prob = model.predict_proba(X)[:, 1]
            df_emb["predicted_liked"] = Y_pred
            df_emb["predicted_liked_prob"] = Y_prob
            print(df_emb[["id", "predicted_liked", "predicted_liked_prob"]])
            rec = pd.concat([rec, df_emb], ignore_index=True)
        print(rec)
        f = "model_recomendations.csv"
        rec.to_csv(f, index= False)
        sys.exit(0)
    if len(sys.argv)>1 and sys.argv[1] == "test":
        # df = pd.read_csv("data/tracks.csv")
        # y_rec = pd.DataFrame()
        # for i in df.itertuples():
        #     from model import get_simmilar_yandex, count_artist_songs_match, track_key
        #     print(_norm(i.name), _norm(i.artist_name))
        #     tmp = get_simmilar_yandex(_norm(i.name), _norm(i.artist_name), limit=10)
        #     y_rec = pd.concat([y_rec, tmp], ignore_index=True)
        # f = "soso2.csv"
        # y_rec.to_csv(f, index= False)
        print(make_auth_url('12345'))
        import uvicorn
        import threading
        import time
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8888)
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(15) 
        df = pd.read_csv("model_recomendations.csv")
        df = df.loc[df["predicted_liked"] == 1].drop_duplicates(subset=["id"], keep="first")
        print(df)
        f = open("soso3.txt", "w", encoding="utf-8")
        for i in df.itertuples():
            time.sleep(1)
            name, artist = get_spotify_song_from_id(i.id)
            f.write(f"{name} | {artist} | {i.id}\n")
        f.close()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "playlist":
        print(make_auth_url('12345'))
        import uvicorn
        import threading
        import time
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8888)
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(15) 
        ids = pd.read_csv("model_recomendations.csv").loc[lambda x : x["predicted_liked"] == 1, "id"].drop_duplicates().sample(frac=1).to_list()
        url, playlist_id = create_spotify_playlist("Model recommendations", ids, ID2TOKEN['12345']['access_token'])
        
    else:
        from model import get_simmilar_yandex
        print(get_simmilar_yandex("супермаркет", "madk1d")) 
        sys.exit(0)
