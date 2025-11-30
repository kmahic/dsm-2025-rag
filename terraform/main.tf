provider "google" {
  project = var.project_id
  region  = var.region
}

provider "postgresql" {
  host            = google_sql_database_instance.vector_db_instance.public_ip_address
  port            = 5432
  database        = "postgres"
  username        = "postgres"
  password        = data.google_secret_manager_secret_version.postgres_password.secret_data
  sslmode         = "require"
  connect_timeout = 15
}

terraform {
  backend "gcs" {
    bucket = "terraform-state-data-science-faggruppe-rag"
    prefix = "terraform/state"
  }
}
