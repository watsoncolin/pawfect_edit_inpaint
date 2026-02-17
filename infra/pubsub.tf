locals {
  project = "pawfect-edit"
  region  = "us-east4"
}

# ── Topic ────────────────────────────────────────────────────────
resource "google_pubsub_topic" "session_processing" {
  project = local.project
  name    = "session-processing"
}

# ── Dead Letter Topic ────────────────────────────────────────────
resource "google_pubsub_topic" "session_processing_deadletter" {
  project = local.project
  name    = "session-processing-deadletter"
}

resource "google_pubsub_subscription" "session_processing_deadletter_sub" {
  project = local.project
  name    = "session-processing-deadletter-sub"
  topic   = google_pubsub_topic.session_processing_deadletter.id
}

# ── Pull Subscription ───────────────────────────────────────────
resource "google_pubsub_subscription" "session_processing_sub" {
  project = local.project
  name    = "session-processing-sub"
  topic   = google_pubsub_topic.session_processing.id

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.session_processing_deadletter.id
    max_delivery_attempts = 5
  }
}
