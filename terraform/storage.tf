data "google_storage_bucket" "terraform_state" {
  name = var.tf_bucket_name
}
