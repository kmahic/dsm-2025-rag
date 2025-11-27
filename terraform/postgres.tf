# Create a Cloud SQL instance with PostgreSQL
resource "google_sql_database_instance" "vector_db_instance" {
  name             = var.vector_db_instance_name
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier      = "db-f1-micro"
    disk_type = "PD_HDD"
    disk_size = 10 # Size in GB

    backup_configuration {
      enabled = false
    }

    ip_configuration {
      ipv4_enabled = true

      # TODO: tighten this to specific trusted IPs / networks
      authorized_networks {
        value = "0.0.0.0/0"
      }
    }
  }
}

# Create the vector database within the Cloud SQL instance
resource "google_sql_database" "vector_db" {
  name     = var.vector_db_name
  instance = google_sql_database_instance.vector_db_instance.name
}

# Create a database user for the vector DB, using the password from Secret Manager
resource "google_sql_user" "vector_db_user" {
  name     = var.vector_db_user_name
  instance = google_sql_database_instance.vector_db_instance.name
  password = data.google_secret_manager_secret_version.db_password_secret.secret_data
}

# Enable pgvector extension in the vector_db database
# This uses the `postgresql` provider, which must be configured to connect to the Cloud SQL instance.
resource "postgresql_extension" "pgvector" {
  name     = "vector" # pgvector extension name in Cloud SQL
  database = google_sql_database.vector_db.name

  depends_on = [
    google_sql_database.vector_db
  ]
}
