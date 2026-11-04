"""Standalone SupoClip pipeline: YouTube URL -> vertical viral clips.

Runs the same building blocks as the worker (download, transcription, AI
segment selection, 9:16 rendering with subtitles and hook titles) without the
database, Redis or the web app. Designed for GitHub Actions, where it uses
local Whisper for transcription and GitHub Models (OpenAI-compatible) for the
AI selection, so no paid API keys are needed.

Usage:
    python -m scripts.make_clips --url URL --out clips_out [--max-clips 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.ai import get_most_relevant_parts_by_transcript  # noqa: E402
from src.video_utils import (  # noqa: E402
    create_clips_from_segments,
    get_transcript_text_in_range,
    get_video_transcript,
    load_cached_transcript_data,
    parse_timestamp_to_seconds,
)
from src.youtube_utils import download_youtube_video, get_youtube_video_title  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("make_clips")


FALLBACK_PLAYER_CLIENTS = [
    "tv_simply",
    "tv",
    "web_safari",
    "mweb",
    "android_vr",
    "ios",
    "web_embedded",
]


def fallback_download(url: str, dest_dir: Path) -> Path | None:
    """Retry the download with alternate YouTube player clients and optional cookies.

    Datacenter IPs (like GitHub runners) often hit YouTube's bot check with the
    default client. A cookies file (YTDLP_COOKIES_FILE) is the reliable fix;
    alternate clients sometimes work without one.
    """
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    cookies = os.getenv("YTDLP_COOKIES_FILE")
    if cookies and not Path(cookies).is_file():
        cookies = None
    for client in FALLBACK_PLAYER_CLIENTS:
        opts: Dict[str, Any] = {
            "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            "retries": 3,
            "extractor_args": {"youtube": {"player_client": [client]}},
        }
        if cookies:
            opts["cookiefile"] = cookies
        try:
            log.info("Fallback download with player_client=%s cookies=%s", client, bool(cookies))
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            files = [
                f for f in dest_dir.glob(f"{info['id']}.*")
                if f.suffix.lower() in (".mp4", ".mkv", ".webm")
            ]
            if files:
                return max(files, key=lambda f: f.stat().st_size)
        except Exception as exc:
            log.warning("player_client=%s failed: %s", client, str(exc)[:300])
    return None


def chunk_transcript(transcript: str, max_chars: int) -> List[str]:
    """Split the timestamped transcript on line boundaries into chunks."""
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for line in transcript.splitlines():
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


async def analyze_in_chunks(transcript: str, max_chars: int) -> List[Dict[str, Any]]:
    chunks = chunk_transcript(transcript, max_chars)
    log.info("Analyzing transcript in %d chunk(s)", len(chunks))
    segments: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, 1):
        for attempt in range(1, 5):
            try:
                analysis = await get_most_relevant_parts_by_transcript(chunk)
                break
            except Exception as exc:  # rate limits / transient errors
                wait = 20 * attempt
                log.warning(
                    "Chunk %d attempt %d failed: %s (retry in %ss)",
                    index, attempt, exc, wait,
                )
                if attempt == 4:
                    analysis = None
                    break
                time.sleep(wait)
        if analysis is None:
            continue
        for seg in analysis.most_relevant_segments:
            virality = seg.virality.model_dump() if seg.virality else {}
            segments.append(
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "relevance_score": seg.relevance_score,
                    "reasoning": seg.reasoning,
                    "virality_score": virality.get("total_score", 0),
                    "hook_score": virality.get("hook_score", 0),
                    "engagement_score": virality.get("engagement_score", 0),
                    "value_score": virality.get("value_score", 0),
                    "shareability_score": virality.get("shareability_score", 0),
                    "hook_type": virality.get("hook_type"),
                    "hook_title": seg.hook_title,
                }
            )
        log.info("Chunk %d/%d -> %d segment(s) so far", index, len(chunks), len(segments))
        time.sleep(5)  # stay well under free-tier rate limits
    return segments


def dedupe_and_rank(segments: List[Dict[str, Any]], max_clips: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        segments,
        key=lambda s: (s.get("virality_score") or 0, s.get("relevance_score") or 0),
        reverse=True,
    )
    chosen: List[Dict[str, Any]] = []
    for seg in ranked:
        try:
            start = parse_timestamp_to_seconds(seg["start_time"])
            end = parse_timestamp_to_seconds(seg["end_time"])
        except Exception:
            continue
        if end - start < 5:
            continue
        overlaps = any(
            start < parse_timestamp_to_seconds(c["end_time"])
            and end > parse_timestamp_to_seconds(c["start_time"])
            for c in chosen
        )
        if not overlaps:
            chosen.append(seg)
        if len(chosen) >= max_clips:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", default="clips_out")
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--caption-template", default="default")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s", args.url)
    video_path = None
    if not os.getenv("YTDLP_COOKIES_FILE"):
        video_path = download_youtube_video(args.url)
    if not video_path:
        video_path = fallback_download(
            args.url, Path(os.getenv("TEMP_DIR", "/tmp")) / "downloads"
        )
    if not video_path:
        log.error("Download failed")
        return 1
    video_path = Path(video_path)
    title = None
    try:
        title = get_youtube_video_title(args.url)
    except Exception:
        pass
    log.info("Downloaded: %s (%s)", video_path, title)

    transcript = get_video_transcript(video_path, source_url=args.url)
    (out_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    log.info("Transcript: %d chars", len(transcript))

    segments = asyncio.run(analyze_in_chunks(transcript, args.chunk_chars))
    if not segments:
        log.error("AI selected no segments")
        return 1
    chosen = dedupe_and_rank(segments, args.max_clips)

    transcript_data = load_cached_transcript_data(video_path)
    for seg in chosen:
        if transcript_data:
            grounded = get_transcript_text_in_range(
                transcript_data,
                parse_timestamp_to_seconds(seg["start_time"]),
                parse_timestamp_to_seconds(seg["end_time"]),
            ).strip()
            if grounded:
                seg["text"] = grounded

    clips = create_clips_from_segments(
        video_path,
        chosen,
        out_dir,
        caption_template=args.caption_template,
        output_format="vertical",
        add_subtitles=True,
    )

    summary = {"url": args.url, "title": title, "clips": []}
    for clip in sorted(clips, key=lambda c: c.get("virality_score") or 0, reverse=True):
        clip_path = Path(clip["path"])
        clip.pop("keep_ranges", None)
        clip["path"] = clip_path.name
        summary["clips"].append(clip)
    (out_dir / "clips.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Done: %d clip(s) in %s", len(clips), out_dir)
    return 0 if clips else 1


if __name__ == "__main__":
    sys.exit(main())
