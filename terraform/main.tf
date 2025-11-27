provider "google" {
  project = var.project_id
  region  = var.region
}

terraform {
  backend "gcs" {
    bucket = "terraform-state-data-science-faggruppe-rag"
    prefix = "terraform/state"
  }
}
