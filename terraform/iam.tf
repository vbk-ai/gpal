locals {
  runtime_iam_roles = toset([
    "roles/aiplatform.user",
    "roles/modelarmor.user",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectUser",
  ])
  runtime_sa_email = "${var.agent_name}-sa-${var.environment}@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_service_account" "gpal_runtime_sa" {
  count = var.manage_iam ? 1 : 0

  project      = var.project_id
  account_id   = "${var.agent_name}-sa-${var.environment}"
  display_name = "GPAL Agent Runtime Service Account (${var.environment})"
  description  = "Dedicated least-privilege runtime identity for the GPAL ADK agent in ${var.environment}."

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_project_iam_member" "gpal_runtime_roles" {
  for_each = var.manage_iam ? local.runtime_iam_roles : toset([])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gpal_runtime_sa[0].email}"
}

