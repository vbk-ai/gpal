provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

provider "google-beta" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

locals {
  common_labels = merge(var.labels, {
    environment = var.environment
  })

  required_apis = toset([
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "aiplatform.googleapis.com",
    "modelarmor.googleapis.com",
    "dlp.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
  ])
}

# ==============================================================================
# 1. Enable Required Google Cloud APIs
# ==============================================================================
resource "google_project_service" "required_apis" {
  for_each = var.manage_project_apis ? local.required_apis : toset([])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ==============================================================================
# 2. Sensitive Data Protection (Cloud DLP) Templates for PII Scrubbing
# ==============================================================================
resource "google_data_loss_prevention_inspect_template" "gpal_pii_inspect" {
  parent       = "projects/${var.project_id}/locations/${var.region}"
  description  = "GPAL (${var.environment}) Sensitive Data Protection inspect template for user PII"
  display_name = "${var.agent_name}-pii-inspect-${var.environment}"
  template_id  = "${replace(var.agent_name, "-", "_")}_pii_inspect_${var.environment}"

  inspect_config {
    min_likelihood = "POSSIBLE"

    dynamic "info_types" {
      for_each = var.pii_info_types
      content {
        name = info_types.value
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_data_loss_prevention_deidentify_template" "gpal_pii_deidentify" {
  parent       = "projects/${var.project_id}/locations/${var.region}"
  description  = "GPAL (${var.environment}) Sensitive Data Protection de-identification template replacing PII with infoType tags"
  display_name = "${var.agent_name}-pii-deidentify-${var.environment}"
  template_id  = "${replace(var.agent_name, "-", "_")}_pii_deidentify_${var.environment}"

  deidentify_config {
    info_type_transformations {
      transformations {
        primitive_transformation {
          replace_with_info_type_config = true
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

# ==============================================================================
# 3. Google Cloud Model Armor Safety & Prompt-Injection Template
# ==============================================================================
resource "google_model_armor_template" "gpal_safety_armor" {
  project     = var.project_id
  location    = var.model_armor_location
  template_id = "${var.agent_name}-armor-${var.environment}"
  labels      = local.common_labels

  filter_config {
    pi_and_jailbreak_filter_settings {
      filter_enforcement = "ENABLED"
      confidence_level   = "LOW_AND_ABOVE"
    }

    malicious_uri_filter_settings {
      filter_enforcement = "ENABLED"
    }

    sdp_settings {
      advanced_config {
        inspect_template    = google_data_loss_prevention_inspect_template.gpal_pii_inspect.id
        deidentify_template = google_data_loss_prevention_deidentify_template.gpal_pii_deidentify.id
      }
    }

    rai_settings {
      rai_filters {
        filter_type      = "HATE_SPEECH"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "HARASSMENT"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "SEXUALLY_EXPLICIT"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "DANGEROUS"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_data_loss_prevention_inspect_template.gpal_pii_inspect,
    google_data_loss_prevention_deidentify_template.gpal_pii_deidentify,
  ]
}

# ==============================================================================
# 4. Secret Manager, Vertex AI Staging Bucket & Artifact Registry
# ==============================================================================
resource "google_secret_manager_secret" "gpal_api_secret" {
  project   = var.project_id
  secret_id = "${var.agent_name}-api-key-${var.environment}"
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_storage_bucket" "gpal_staging_bucket" {
  project                     = var.project_id
  name                        = "${var.project_id}-${var.agent_name}-staging-${var.environment}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.environment != "prod"
  labels                      = local.common_labels

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_artifact_registry_repository" "gpal_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = "${var.agent_name}-${var.environment}"
  description   = "Docker container repository for GPAL Agent (${var.environment})"
  format        = "DOCKER"
  labels        = local.common_labels

  depends_on = [
    google_project_service.required_apis,
  ]
}
