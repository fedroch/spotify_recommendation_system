""" gptшные проверки сопоставления записей и сетевых ошибок без реальных запросов.

Запуск: .venv/bin/python -B -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

os.environ.setdefault("api_key_lastfm", "test-lastfm-key")
from model import find_yandex_track_id, get_simmilar_yandex, get_yandex_radio

COLUMNS = ["name", "artist", "match", "playcount"]


def track(identifier, title, *artists, **extra):
    return {"id": identifier, "title": title,
            "artists": [{"name": name} for name in artists], **extra}


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "https://api.music.yandex.net"
    result.encoding = "utf-8"
    result._content = json.dumps(payload, ensure_ascii=False).encode()
    return result


def search_result(tracks, **extra):
    return response({"result": {"tracks": {
        "results": tracks, "total": len(tracks), "perPage": 20, **extra,
    }}})


def radio_result(tracks, **extra):
    return response({"result": {
        "radioSessionId": "radio-session", "terminated": True,
        "sequence": [{"type": "track", "track": t} for t in tracks], **extra,
    }})


class YandexRecommendationsTests(unittest.TestCase):
    def run_call(self, function, args, replies, **kwargs):
        network = Mock(side_effect=replies)
        with patch("model.requests.get", side_effect=network), \
                patch("model.requests.post", side_effect=network):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = function(*args, **kwargs)
        return result, network.call_args_list, output.getvalue()

    def call(self, name, artist, replies, **kwargs):
        result, calls, output = self.run_call(get_simmilar_yandex, (name, artist), replies, **kwargs)
        self.assertEqual(result.columns.tolist(), COLUMNS)
        return result, calls, output

    def test_user_examples(self):
        for title, artist, canonical in [
            ("супермаркет", "madk1d", "madk1d"),
            ("карма", "юпи", "Юпи"),
            ("простужен", "fortuna812", "FORTUNA 812"),
        ]:
            with self.subTest(title=title):
                result, calls, _ = self.call(title, artist, [
                    search_result([track("seed", title, canonical, "madk1d")]),
                    radio_result([track("one", "Новая", "Другой")]),
                ])
                self.assertEqual(result["name"].tolist(), ["Новая"])
                self.assertEqual(result["match"].tolist(), [1.0])
                self.assertEqual(result["playcount"].tolist(), [0])
                self.assertTrue(calls[1].args[0].endswith("/rotor/session/new"))
                self.assertEqual(calls[1].kwargs["json"]["seeds"], ["track:seed"])
                self.assertTrue(calls[1].kwargs["json"]["incognito"])

    def test_transliteration_and_feature_credits(self):
        for title, artist, seed in [
            ("Кукушка", "Kino", track("seed", "Кукушка", "Кино")),
            ("Песня feat. Другой", "Артист", track("seed", "Песня", "Артист", "Другой")),
            ("Всё — пройдёт!", "Группа", track("seed", "Все пройдет", "Группа")),
        ]:
            with self.subTest(title=title):
                result, _, _ = self.run_call(find_yandex_track_id, (title, artist), [search_result([seed])])
                self.assertEqual(result, "seed")

    def test_next_page_and_requested_version(self):
        result, calls, _ = self.run_call(find_yandex_track_id, ("Песня - Live", "Артист"), [
            search_result([track("original", "Песня", "Артист")], total=2, perPage=1),
            search_result([track("live", "Песня", "Артист", version="Live")], total=2, perPage=1),
        ])
        self.assertEqual(result, "live")
        self.assertEqual([c.kwargs["params"]["page"] for c in calls], [0, 1])

    def test_fallback_to_title_only_search(self):
        result, calls, _ = self.run_call(find_yandex_track_id, ("песня", "артист"), [
            search_result([]), search_result([track("seed", "песня", "артист")]),
        ])
        self.assertEqual(result, "seed")
        self.assertEqual(calls[-1].kwargs["params"]["text"], "песня")

    def test_rejects_other_artist_or_title_prefix(self):
        for wrong in [track("wrong", "Love", "Different Artist"), track("wrong", "Love Yourself", "Artist")]:
            with self.subTest(wrong=wrong):
                result, calls, _ = self.call("Love", "Artist", lambda *a, **kw: search_result([wrong]))
                self.assertTrue(result.empty)
                self.assertTrue(all(c.args[0].endswith("/search") for c in calls))

    def test_multiple_batches_queue_and_limit(self):
        first, second = track("one", "Первая", "Артист"), track("two", "Вторая", "Артист")
        result, calls, _ = self.run_call(get_yandex_radio, ("seed",), [
            radio_result([track("seed", "Исходная", "Артист"), first], terminated=False),
            radio_result([first, second], terminated=False),
        ], limit=2)
        self.assertEqual(result["name"].tolist(), ["Первая", "Вторая"])
        self.assertEqual(result["match"].tolist(), [1.0, .5])
        self.assertEqual(calls[0].kwargs["json"]["queue"], ["seed"])
        self.assertEqual(calls[1].kwargs["json"], {"queue": ["seed", "one"]})
        self.assertTrue(calls[1].args[0].endswith("/rotor/session/radio-session/tracks"))

    def test_duplicates_ads_and_missing_metadata(self):
        result, _, _ = self.run_call(get_yandex_radio, ("seed",), [
            radio_result([], sequence=[
                {"type": "ad", "track": track("ad", "Реклама", "Реклама")},
                *[{"type": "track", "track": t} for t in [
                    track("seed", "Исходная", "Артист"), None, track("bad", "Без артиста"),
                    track("one", "Новая", "Кино", version="Live"),
                    track("one", "Новая", "Кино", version="Live"),
                    track("two", "Новая", "Kino", version="Live"),
                ]],
            ]),
        ])
        self.assertEqual(result["name"].tolist(), ["Новая - Live"])

    def test_empty_initial_batch_and_repeated_batch_stop(self):
        first = track("one", "Первая", "Артист")
        result, calls, _ = self.run_call(get_yandex_radio, ("seed",), [
            radio_result([], terminated=False),
            radio_result([first], terminated=False),
            radio_result([first], terminated=False),
        ])
        self.assertEqual(result["name"].tolist(), ["Первая"])
        self.assertEqual(len(calls), 3)

    def test_max_batches(self):
        result, calls, _ = self.run_call(get_yandex_radio, ("seed",), [
            radio_result([track("one", "Первая", "Артист")], terminated=False),
            radio_result([track("two", "Вторая", "Артист")], terminated=False),
        ], limit=20, max_batches=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(calls), 2)

    def test_empty_search_and_empty_radio(self):
        result, calls, message = self.call("песня", "артист", lambda *a, **kw: search_result([]))
        self.assertTrue(result.empty)
        self.assertIn("не удалось сопоставить", message)
        self.assertTrue(all(c.args[0].endswith("/search") for c in calls))
        result, _, message = self.call("песня", "артист", [
            search_result([track("seed", "песня", "артист")]), radio_result([]),
        ])
        self.assertTrue(result.empty)
        self.assertIn("радио", message)

    def test_http_error_reports_body_and_hides_token(self):
        token = "test-private-token"
        for status in (401, 451):
            with self.subTest(status=status):
                result, calls, message = self.call("песня", "артист", [
                    search_result([track("seed", "песня", "артист")]),
                    response({"error": token}, status),
                ], token=token)
                self.assertTrue(result.empty)
                self.assertIn(f"HTTP {status}", message)
                self.assertNotIn(token, message)
                self.assertEqual(calls[-1].kwargs["headers"], {"Authorization": f"OAuth {token}"})
                self.assertEqual(calls[-1].kwargs["timeout"], 20)

    def test_network_error(self):
        result, _, message = self.call("песня", "артист", requests.Timeout("timeout"))
        self.assertTrue(result.empty)
        self.assertIn("timeout", message)

    def test_zero_limit_makes_no_requests(self):
        result, calls, _ = self.call("песня", "артист", [], limit=0)
        self.assertTrue(result.empty)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
