//! Speaker diarization via the local WhisperX + pyannote sidecar
//! (see `diarization-sidecar/` at the repo root).
//!
//! ponytail: best-effort enrichment, not a recording dependency. If the
//! sidecar is unreachable or errors, we log and leave `speaker` null —
//! recording, import and transcription must never fail because diarization
//! is down.
//!
//! Two modes, chosen by the `DIARIZATION_MODE` env var:
//!
//! - `labels` (default): ask the sidecar for speaker turns only (`/diarize`)
//!   and stamp each existing transcript row with its dominant speaker. Keeps
//!   Meetily's own transcript untouched, and skips the sidecar's Whisper pass.
//! - `replace`: run the sidecar's full pipeline (`/transcribe`) and replace the
//!   meeting's transcript rows with WhisperX's speaker-labeled segments. Finer
//!   segmentation and better attribution, but re-transcribes the whole file.

use std::path::{Path, PathBuf};
use std::time::Duration;

use log::{info, warn};
use sqlx::SqlitePool;

use crate::database::models::Transcript;
use crate::database::repositories::transcript::TranscriptsRepository;

/// Generous ceiling: diarizing a multi-hour recording legitimately takes a
/// long time, and this whole task is fire-and-forget. The old 10-minute cap
/// silently killed long meetings.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

fn sidecar_base_url() -> String {
    std::env::var("DIARIZATION_SIDECAR_URL").unwrap_or_else(|_| "http://127.0.0.1:8765".to_string())
}

/// `replace` swaps in WhisperX's transcript; anything else keeps ours.
fn replace_mode() -> bool {
    std::env::var("DIARIZATION_MODE")
        .map(|m| m.eq_ignore_ascii_case("replace"))
        .unwrap_or(false)
}

#[derive(Debug, serde::Deserialize)]
struct DiarizeSegment {
    start: f64,
    end: f64,
    speaker: Option<String>,
    /// Only present on the `/transcribe` (replace-mode) response.
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct DiarizeResponse {
    segments: Vec<DiarizeSegment>,
}

/// The recording folder holds exactly one `audio.*` file, but the extension
/// varies by source: the live recorder finalizes `audio.mp4`, while an import
/// keeps the original container (`audio.m4a`, `audio.wav`, ...). Resolve it by
/// stem rather than guessing the extension.
fn resolve_audio_path(folder: &Path) -> anyhow::Result<PathBuf> {
    for entry in std::fs::read_dir(folder)? {
        let path = entry?.path();
        if path.is_file() && path.file_stem().and_then(|s| s.to_str()) == Some("audio") {
            return Ok(path);
        }
    }
    anyhow::bail!("no audio.* file in {}", folder.display())
}

/// Diarize a finalized meeting and write speaker labels onto its transcript.
/// Runs fire-and-forget after a recording is saved or an import completes;
/// never propagates errors.
pub async fn diarize_and_merge(pool: SqlitePool, meeting_id: String, folder: PathBuf) {
    if let Err(e) = run(&pool, &meeting_id, &folder).await {
        warn!(
            "Diarization skipped for meeting {}: {} (transcript speakers left unset)",
            meeting_id, e
        );
    }
}

async fn run(pool: &SqlitePool, meeting_id: &str, folder: &Path) -> anyhow::Result<()> {
    let audio_path = resolve_audio_path(folder)?;
    let bytes = tokio::fs::read(&audio_path).await?;
    let filename = audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.mp4")
        .to_string();

    let replace = replace_mode();
    let endpoint = if replace { "transcribe" } else { "diarize" };

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;

    let part = reqwest::multipart::Part::bytes(bytes).file_name(filename);
    let form = reqwest::multipart::Form::new().part("file", part);

    info!(
        "Requesting diarization for meeting {} via /{} ({})",
        meeting_id,
        endpoint,
        audio_path.display()
    );
    let resp = client
        .post(format!("{}/{}", sidecar_base_url(), endpoint))
        .multipart(form)
        .send()
        .await?
        .error_for_status()?
        .json::<DiarizeResponse>()
        .await?;

    if replace {
        replace_transcript(pool, meeting_id, &resp.segments).await
    } else {
        merge_speaker_labels(pool, meeting_id, &resp.segments).await
    }
}

/// Default mode: keep our transcript, stamp each row with the speaker who
/// overlaps it most.
async fn merge_speaker_labels(
    pool: &SqlitePool,
    meeting_id: &str,
    segments: &[DiarizeSegment],
) -> anyhow::Result<()> {
    let transcripts: Vec<Transcript> =
        sqlx::query_as::<_, Transcript>("SELECT * FROM transcripts WHERE meeting_id = ?")
            .bind(meeting_id)
            .fetch_all(pool)
            .await?;

    let mut updated = 0;
    for t in transcripts {
        let (Some(seg_start), Some(seg_end)) = (t.audio_start_time, t.audio_end_time) else {
            continue;
        };

        let best = segments
            .iter()
            .filter_map(|d| {
                let overlap = (seg_end.min(d.end) - seg_start.max(d.start)).max(0.0);
                d.speaker.as_ref().map(|s| (overlap, s))
            })
            .filter(|(overlap, _)| *overlap > 0.0)
            .max_by(|a, b| a.0.total_cmp(&b.0));

        if let Some((_, speaker)) = best {
            TranscriptsRepository::update_speaker(pool, &t.id, speaker).await?;
            updated += 1;
        }
    }

    info!(
        "Diarization merged for meeting {}: {} transcript rows labeled",
        meeting_id, updated
    );
    Ok(())
}

/// `replace` mode: swap our transcript rows for WhisperX's speaker-labeled
/// segments. Only reached after the sidecar has already returned a full
/// response, so a sidecar failure can never destroy an existing transcript.
async fn replace_transcript(
    pool: &SqlitePool,
    meeting_id: &str,
    segments: &[DiarizeSegment],
) -> anyhow::Result<()> {
    let rows: Vec<(String, f64, f64, String)> = segments
        .iter()
        .filter_map(|s| {
            let text = s.text.as_ref()?.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some((
                text,
                s.start,
                s.end,
                s.speaker.clone().unwrap_or_else(|| "SPEAKER_?".to_string()),
            ))
        })
        .collect();

    if rows.is_empty() {
        anyhow::bail!("sidecar returned no usable segments; keeping existing transcript");
    }

    let count = rows.len();
    TranscriptsRepository::replace_transcripts(pool, meeting_id, &rows).await?;

    info!(
        "Diarization replaced transcript for meeting {}: {} labeled segments",
        meeting_id, count
    );
    Ok(())
}
