variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "data-science-faggruppe-rag"
}

variable "region" {
  description = "The GCP region"
  default     = "europe-west1"
}


variable "vector_db_instance_name" {
  description = "The name of the Cloud SQL PostgreSQL instance"
  type        = string
  default     = "vector-db-instance"
}

variable "vector_db_name" {
  description = "The name of the vector database"
  type        = string
  default     = "vector_db"
}

variable "vector_db_user_name" {
  description = "The name of the database user for the vector DB"
  type        = string
  default     = "vector_db_user"
}

variable "postgres_password_secret_id" {
  description = "The Secret Manager secret ID for postgres password"
  type        = string
  default     = "postgres-password"
}

variable "db_password_secret_id" {
  description = "The Secret Manager secret ID for vector_db_user password"
  type        = string
  default     = "vector-db-password"
}

variable "tf_bucket_name" {
  description = "Name of bucket storing Terraform state"
  default     = "terraform-state-data-science-faggruppe-rag"
}
