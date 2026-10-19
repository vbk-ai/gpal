variable "project_id" {
  type        = string
  description = "Target Google Cloud Project ID where GPAL infrastructure will be provisioned."

  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "The project_id variable must be a non-empty GCP project identifier."
  }
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Primary Google Cloud region for Vertex AI Agent Engine, Cloud DLP, and storage resources."
}

variable "model_armor_location" {
  type        = string
  default     = "us-central1"
  description = "Location for the Google Cloud Model Armor safety template (e.g., us-central1, europe-west4)."
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment tier (dev, staging, or prod) used to namespace all resources."

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "The environment variable must be one of: \"dev\", \"staging\", or \"prod\"."
  }
}

variable "agent_name" {
  type        = string
  default     = "gpal-agent"
  description = "Base identifier prefix for GPAL resources."
}

variable "pii_info_types" {
  type = list(string)
  default = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_CARD_NUMBER",
    "IP_ADDRESS",
  ]
  description = "Cloud DLP infoTypes inspected and redacted before user prompts enter SessionService or MemoryService."
}

variable "labels" {
  type = map(string)
  default = {
    "app"        = "gpal-agent"
    "managed-by" = "terraform"
  }
  description = "Common resource labels applied across all supported Google Cloud resources."
}

variable "manage_project_apis" {
  type        = bool
  default     = true
  description = "Whether Terraform should enable project-level Google Cloud APIs. Set to false when using a least-privilege deployer without roles/serviceusage.serviceUsageAdmin."
}

variable "manage_iam" {
  type        = bool
  default     = true
  description = "Whether Terraform should create the runtime Service Account and project IAM bindings. Set to false when using a least-privilege deployer without roles/iam.serviceAccountAdmin and roles/resourcemanager.projectIamAdmin."
}

