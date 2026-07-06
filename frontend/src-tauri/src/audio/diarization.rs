//! Speaker diarization via the local WhisperX + pyannote sidecar
//! (see `diarization-sidecar/` at the repo root).
//!
//! ponytail: best-effort enrichment, not a recording dependency. If the
//! sidecar is unreachable or errors, we log and leave `speaker` null —
//! recording and transcription must never fail because diarization is down.

use std::path::Path;
use std::time::Duration;

use log::{info, warn};
use sqlx::SqlitePool;

use crate::database::models::Transcript;
use crate::database::repositories::transcript::TranscriptsRepository;

fn sidecar_base_url() -> String {
    std::env::var("DIARIZATION_SIDECAR_URL").unwrap_or_else(|_| "http://127.0.0.1:8765".to_string())
}

#[derive(Debug, serde::Deserialize)]
struct DiarizeSegment {
    start: f64,
    end: f64,
    speaker: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct DiarizeResponse {
    segments: Vec<DiarizeSegment>,
}

/// Diarize the finalized meeting audio and write speaker labels onto the
/// already-saved transcript rows, matched by timestamp overlap. Runs
/// fire-and-forget after a recording is saved; never propagates errors.
pub async fn diarize_and_merge(pool: SqlitePool, meeting_id: String, audio_path: std::path::PathBuf) {
    if let Err(e) = run(&pool, &meeting_id, &audio_path).await {
        warn!(
            "Diarization skipped for meeting {}: {} (transcript speakers left unset)",
            meeting_id, e
        );
    }
}

async fn run(pool: &SqlitePool, meeting_id: &str, audio_path: &Path) -> anyhow::Result<()> {
    let bytes = tokio::fs::read(audio_path).await?;
    let filename = audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.mp4")
        .to_string();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let part = reqwest::multipart::Part::bytes(bytes).file_name(filename);
    let form = reqwest::multipart::Form::new().part("file", part);

    info!("Requesting diarization for meeting {}", meeting_id);
    let resp = client
        .post(format!("{}/transcribe", sidecar_base_url()))
        .multipart(form)
        .send()
        .await?
        .error_for_status()?
        .json::<DiarizeResponse>()
        .await?;

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

        let best = resp
            .segments
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
