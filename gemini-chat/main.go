package main

import (
	"gemini-chat/config"
	"gemini-chat/handler"
	"log"
	"net/http"
)

func main() {
	// Load konfigurasi
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Setup handler
	chatHandler := handler.NewChatHandler(cfg)

	// File statis (HTML, CSS, JS)
	fs := http.FileServer(http.Dir("./static"))
	http.Handle("/", fs)

	// API endpoints
	http.HandleFunc("/api/chat", chatHandler.HandleChat)

	// Start server
	addr := cfg.ServerHost + ":" + cfg.ServerPort
	log.Printf("Server running on http://%s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
