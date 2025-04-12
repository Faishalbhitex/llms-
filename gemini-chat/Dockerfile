# --- Build Stage ---
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./
# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
# -ldflags="-w -s" reduces binary size
# CGO_ENABLED=0 produces a static binary suitable for minimal containers like alpine
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /active_server main.go

# --- Final Stage ---
FROM alpine:latest

WORKDIR /app

# Copy the static binary from the builder stage
COPY --from=builder /active_server /app/active_server

# Copy static files and potentially config directory if needed (though env vars are better)
COPY static ./static
# COPY config ./config # Sebaiknya tidak perlu jika semua config via Env Var

# Expose the port the app runs on (read from ENV or default 8080)
# Platform biasanya akan override ini, tapi bagus untuk dokumentasi
EXPOSE 8080

# Command to run the application
# User non-root direkomendasikan tapi untuk simpel bisa root dulu
# RUN adduser -D nonroot
# USER nonroot
CMD ["/app/active_server"]
