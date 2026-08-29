import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from sound_diversity.core import (
    DiscogsClient, ITunesClient, JsonCache, clean_discogs_artist, discogs_artist_credit,
    is_discogs_release, is_usable_artist,
    mean_pairwise_diversity, normalize, panel_skeleton, stable_seed,
)
from sound_diversity.chorus import (
    TimedWord, classify_section_header, clean_annotation_source_rows, detect_counted_sections,
    evaluate_annotations, legacy_filename_title, parse_intervals, parse_lyric_sections,
    resolve_track_identity, stratified_annotation_sample,
)
from sound_diversity.preview_rebuild import apple_match_is_valid, corrected_identity, safe_filename_part


class CoreTests(unittest.TestCase):
    def test_diversity_identical(self):
        self.assertAlmostEqual(mean_pairwise_diversity(np.array([[1, 0], [1, 0]])), 0.0)

    def test_diversity_orthogonal(self):
        self.assertAlmostEqual(mean_pairwise_diversity(np.array([[1, 0], [0, 1]])), 1.0)

    def test_diversity_uses_each_pair_once(self):
        value = mean_pairwise_diversity(np.array([[1, 0], [0, 1], [-1, 0]]))
        self.assertAlmostEqual(value, 4 / 3)

    def test_less_than_two_is_missing(self):
        self.assertIsNone(mean_pairwise_diversity(np.array([[1, 0]])))

    def test_normalization(self):
        self.assertEqual(normalize("Bar/None & Records"), "bar none and records")

    def test_track_artist_credit_removes_discogs_suffix(self):
        artists = [{"name": "Artist A (2)", "join": "&"}, {"name": "Artist B", "join": ""}]
        self.assertEqual(discogs_artist_credit(artists), "Artist A & Artist B")

    def test_various_is_not_usable(self):
        self.assertFalse(is_usable_artist("Various"))
        self.assertFalse(is_usable_artist("Various Artists"))
        self.assertTrue(is_usable_artist("Artist A"))

    def test_artist_named_chorus_headers_are_recognized(self):
        self.assertEqual(classify_section_header("Singer: Chorus"), "chorus")
        self.assertEqual(classify_section_header("Chorus: Singer"), "chorus")

    def test_pre_and_post_chorus_are_not_folded_into_chorus(self):
        self.assertEqual(classify_section_header("Singer: Pre-Chorus"), "pre_chorus")
        self.assertEqual(classify_section_header("Post Chorus: Singer"), "post_chorus")

    def test_lyric_sections_preserve_taxonomy(self):
        sections = parse_lyric_sections(
            "[Singer: Pre-Chorus]\nAlmost there now\n"
            "[Singer: Chorus]\nThis is the main repeated line\n"
            "[Post-Chorus: Singer]\nOh oh oh oh\n"
        )
        self.assertEqual([section.kind for section in sections], ["pre_chorus", "chorus", "post_chorus"])
        self.assertEqual([section.counted_as_chorus for section in sections], [False, True, False])

    def test_various_release_prefers_track_artist(self):
        artist, title, source = resolve_track_identity({
            "release_artist": "Various", "track_artist": "Softball",
            "artist": "Softball", "track_title": "Choice",
        })
        self.assertEqual((artist, title, source), ("Softball", "Choice", "track_artist"))

    def test_various_without_track_credit_can_use_apple_artist(self):
        artist, title, source = resolve_track_identity({
            "release_artist": "Various Artists", "track_artist": "",
            "artist": "Various", "apple_artist": "Actual Artist",
            "track_title": "Track (Live)",
        })
        self.assertEqual((artist, title, source), ("Actual Artist", "Track (Live)", "apple_artist"))

    def test_legacy_filename_title_preserves_version(self):
        title = legacy_filename_title(r"C:\previews\000028_Artist_-_Song (Steve Aoki Remix).m4a")
        self.assertEqual(title, "Song (Steve Aoki Remix)")

    def test_ordered_detection_rejects_bag_of_words(self):
        lyrics = "[Artist: Chorus]\none two three four five"
        words = [
            TimedWord(token, index, index + 0.5)
            for index, token in enumerate("five four three two one".split())
        ]
        intervals, matches = detect_counted_sections(parse_lyric_sections(lyrics), words, min_score=0.72)
        self.assertEqual(intervals, [])
        self.assertEqual(matches, [])

    def test_pre_chorus_is_not_counted_even_when_words_match(self):
        lyrics = "[Artist: Pre-Chorus]\none two three four five"
        words = [
            TimedWord(token, index, index + 0.5)
            for index, token in enumerate("one two three four five".split())
        ]
        intervals, _ = detect_counted_sections(parse_lyric_sections(lyrics), words)
        self.assertEqual(intervals, [])

    def test_annotation_evaluation_separates_presence_and_timing(self):
        metrics = evaluate_annotations([
            {"human_chorus_present": "yes", "human_chorus_intervals": "10-20", "predicted_intervals": "12-18"},
            {"human_chorus_present": "no", "human_chorus_intervals": "", "predicted_intervals": ""},
        ])
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertAlmostEqual(metrics["temporal_precision"], 1.0)
        self.assertAlmostEqual(metrics["temporal_recall"], 0.6)

    def test_parse_intervals_accepts_preview_minute_seconds_format(self):
        self.assertEqual(parse_intervals("0:05-0:14;0:20-0:30"), [(5.0, 14.0), (20.0, 30.0)])

    def test_annotation_sample_avoids_duplicate_artist_titles(self):
        rows = [
            {"artist": f"Artist {index}", "title": f"Song {index}", "status": "ok", "chorus_percent": str(index)}
            for index in range(50)
        ]
        rows.append(dict(rows[0]))
        sample = stratified_annotation_sample(rows, size=20)
        keys = {(row["artist"], row["title"]) for row in sample}
        self.assertEqual(len(keys), len(sample))

    def test_preview_identity_replaces_various_with_track_artist(self):
        detail = {
            "artists": [{"name": "Various", "join": ""}],
            "tracklist": [{
                "type_": "track", "position": "C2", "title": "Superstar",
                "artists": [{"name": "Young Stoner Life", "join": ","}, {"name": "Young Thug", "join": ""}],
            }],
        }
        identity = corrected_identity(detail, "C2", "Superstar")
        self.assertEqual(identity[:3], ("Young Stoner Life , Young Thug", "Superstar", "track"))

    def test_preview_identity_rejects_unresolved_various(self):
        detail = {
            "artists": [{"name": "Various", "join": ""}],
            "tracklist": [{"type_": "track", "position": "1", "title": "Unknown", "artists": []}],
        }
        with self.assertRaisesRegex(ValueError, "metadata_unresolved"):
            corrected_identity(detail, "1", "Unknown")

    def test_safe_filename_removes_windows_reserved_characters(self):
        self.assertEqual(safe_filename_part('Artist: A/B?'), "Artist A B")

    def test_clean_annotation_join_never_restores_legacy_various(self):
        clean = [{
            "song_id": "31", "status": "ok", "artist": "Hermitude",
            "title": "The Buzz", "artist_source": "track", "audio_source": "verified_legacy_copy",
            "output_path": r"C:\clean\000031_Hermitude_-_The Buzz.m4a",
        }]
        old = [{
            "audio_path": r"C:\old\000031_Various_-_The Buzz.m4a", "artist": "Various",
            "title": "The Buzz", "status": "error", "chorus_percent": "", "intervals": "",
        }]
        joined = clean_annotation_source_rows(clean, old)
        self.assertEqual(joined[0]["artist"], "Hermitude")
        self.assertEqual(joined[0]["metadata_status"], "verified_track_artist")

    def test_discogs_release_detection_uses_resource_url(self):
        self.assertTrue(is_discogs_release({"resource_url": "https://api.discogs.com/releases/123"}))
        self.assertFalse(is_discogs_release({"resource_url": "https://api.discogs.com/masters/123"}))

    def test_discogs_404_is_cached_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = DiscogsClient("token", "tests", JsonCache(Path(tmp)))
            response = Mock(status_code=404, headers={})
            client.session.get = Mock(return_value=response)
            result = client.get("/releases/123")
            self.assertEqual(result["_http_status"], 404)
            self.assertEqual(client.get("/releases/123"), result)
            client.session.get.assert_called_once()

    @patch("sound_diversity.core.time.sleep")
    def test_discogs_500_retries_then_is_cached_as_failed(self, _sleep):
        with tempfile.TemporaryDirectory() as tmp:
            client = DiscogsClient("token", "tests", JsonCache(Path(tmp)))
            response = Mock(status_code=500, headers={})
            client.session.get = Mock(return_value=response)
            result = client.get("/releases/456")
            self.assertEqual(result["_http_status"], 500)
            self.assertEqual(client.session.get.call_count, 5)
            self.assertEqual(client.get("/releases/456"), result)
            self.assertEqual(client.session.get.call_count, 5)

    @patch("sound_diversity.core.time.sleep")
    def test_discogs_connection_failure_retries_until_success(self, _sleep):
        with tempfile.TemporaryDirectory() as tmp:
            client = DiscogsClient("token", "tests", JsonCache(Path(tmp)))
            failure = __import__("requests").ConnectionError("temporary DNS failure")
            success = Mock(status_code=200, headers={})
            success.json.return_value = {"id": 789}
            client.session.get = Mock(side_effect=[failure, failure, success])
            self.assertEqual(client.get("/releases/789"), {"id": 789})
            self.assertEqual(client.session.get.call_count, 3)

    @patch("sound_diversity.core.time.sleep")
    def test_itunes_connection_failure_retries(self, _sleep):
        with tempfile.TemporaryDirectory() as tmp:
            client = ITunesClient("US", 0.82, JsonCache(Path(tmp)))
            failure = __import__("requests").ConnectionError("temporary outage")
            success = Mock(status_code=200, headers={})
            success.json.return_value = {"results": []}
            client.session.get = Mock(side_effect=[failure, success])
            self.assertIsNone(client.search_track("Artist", "Track"))
            self.assertEqual(client.session.get.call_count, 2)

    @patch("sound_diversity.core.time.sleep")
    def test_itunes_403_retries_quickly(self, sleep):
        with tempfile.TemporaryDirectory() as tmp:
            client = ITunesClient("US", 0.82, JsonCache(Path(tmp)))
            forbidden = Mock(status_code=403, headers={})
            success = Mock(status_code=200, headers={})
            success.json.return_value = {"results": []}
            client.session.get = Mock(side_effect=[forbidden, success])
            self.assertIsNone(client.search_track("Artist", "Track"))
            self.assertEqual(client.session.get.call_count, 2)
            sleep.assert_any_call(1)

    @patch("sound_diversity.core.time.sleep")
    def test_itunes_repeated_403_is_recorded(self, _sleep):
        with tempfile.TemporaryDirectory() as tmp:
            client = ITunesClient("US", 0.82, JsonCache(Path(tmp)))
            forbidden = Mock(status_code=403, headers={})
            client.session.get = Mock(return_value=forbidden)
            self.assertIsNone(client.search_track("Artist", "Track"))
            self.assertEqual(client.last_lookup_status, "http_403")
            self.assertEqual(client.session.get.call_count, 3)

    @patch("sound_diversity.core.time.sleep")
    def test_itunes_cached_403_can_be_retried(self, _sleep):
        with tempfile.TemporaryDirectory() as tmp:
            client = ITunesClient("US", 0.82, JsonCache(Path(tmp)))
            forbidden = Mock(status_code=403, headers={})
            client.session.get = Mock(return_value=forbidden)
            self.assertIsNone(client.search_track("Artist", "Track"))
            self.assertEqual(client.session.get.call_count, 3)

            success = Mock(status_code=200, headers={})
            success.json.return_value = {"results": []}
            client.session.get = Mock(return_value=success)
            self.assertIsNone(
                client.search_track("Artist", "Track", retry_cached_403=True)
            )
            self.assertEqual(client.session.get.call_count, 1)
            self.assertEqual(client.last_lookup_status, "no_match")

    def test_clean_artist_only_removes_numeric_suffix(self):
        self.assertEqual(clean_discogs_artist("The Band (2)"), "The Band")
        self.assertEqual(clean_discogs_artist("The Band (Live)"), "The Band (Live)")

    def test_seed_is_stable_and_scoped(self):
        self.assertEqual(stable_seed(42, "x"), stable_seed(42, "x"))
        self.assertNotEqual(stable_seed(42, "x"), stable_seed(42, "y"))

    def test_treatment_fields(self):
        rows = panel_skeleton([{"label": "X", "start_year": "2018", "end_year": "2020", "cohort_year": "2019", "acquirer": "A", "big_three": "", "ever_treated": "1"}])
        self.assertEqual([r["treated"] for r in rows], [0, 1, 1])
        self.assertEqual([r["event_time"] for r in rows], [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()
