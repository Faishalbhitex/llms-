package config

import (
	"errors"
	"os"

	"github.com/joho/godotenv"
)

// Config menyimpan konfigurasi aplikasi
type Config struct {
	// Server settings
	ServerHost string
	ServerPort string
	
	// Gemini API settings
	GeminiAPIKey string
	GeminiAPIURL string
}

// Load memuat konfigurasi dari file .env
func Load() (*Config, error) {
	// Load .env file if exists
	_ = godotenv.Load() // ignore error if .env doesn't exist

	cfg := &Config{
		ServerHost:   getEnv("SERVER_HOST", "localhost"),
		ServerPort:   getEnv("SERVER_PORT", "8080"),
		GeminiAPIKey: os.Getenv("GEMINI_API_KEY"),
		GeminiAPIURL: getEnv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"),
	}

	// Validasi konfigurasi yang diperlukan
	if cfg.GeminiAPIKey == "" {
		return nil, errors.New("GEMINI_API_KEY is required")
	}

	return cfg, nil
}

// getEnv mendapatkan nilai environment variable atau nilai default jika tidak ada
func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}
