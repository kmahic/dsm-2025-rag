data "google_storage_bucket" "terraform_state" {
  name = var.tf_bucket_name
}

# Bucket for storing PDF documents and processed data
resource "google_storage_bucket" "pdf_data" {
  name     = "${var.project_id}-pdf-data"
  location = var.region

  versioning {
    enabled = true
  }
}
