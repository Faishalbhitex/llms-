package handler

import (
    "encoding/json"
    "gemini-chat/api"
    "gemini-chat/config"
    "log" // <-- TAMBAHKAN import log
    "net/http"
    "time" // <-- TAMBAHKAN import time (opsional, untuk timing)
)

// ... (struct ChatRequest, ChatResponse, ChatHandler, NewChatHandler tetap sama) ...

// HandleChat menangani permintaan chat HTTP
func (h *ChatHandler) HandleChat(w http.ResponseWriter, r *http.Request) {
    start := time.Now() // <-- Opsional: Mulai timer
    log.Printf("[INFO] Received request for /api/chat from %s", r.RemoteAddr) // <-- Log permintaan masuk

    // Hanya menerima metode POST
    if r.Method != http.MethodPost {
        log.Printf("[WARN] Method not allowed: %s", r.Method) // <-- Log metode salah
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    // Decode permintaan JSON
    var req ChatRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        log.Printf("[ERROR] Error decoding JSON request: %v", err) // <-- Log error decode
        http.Error(w, "Invalid JSON format", http.StatusBadRequest)
        return
    }
    log.Printf("[INFO] Received message: \"%s\"", req.Message) // <-- Log pesan yang diterima

    // Validasi pesan
    if req.Message == "" {
        log.Printf("[WARN] Empty message received") // <-- Log pesan kosong
        http.Error(w, "Message cannot be empty", http.StatusBadRequest)
        return
    }

    // Dapatkan respons dari Gemini API
    log.Println("[INFO] Sending request to Gemini API...") // <-- Log sebelum panggil Gemini
    response, err := h.geminiClient.GenerateResponse(req.Message)
    duration := time.Since(start) // <-- Opsional: Hitung durasi

    if err != nil {
        // Log error spesifik dari GeminiClient
        log.Printf("[ERROR] Error from GeminiClient.GenerateResponse: %v (Duration: %s)", err, duration) // <-- Log error DETAIL
        // Kirim error generik ke klien, tapi catat error spesifik di log
        http.Error(w, "Failed to generate response", http.StatusInternalServerError)
        return
    }

    log.Printf("[INFO] Successfully received response from Gemini API. (Duration: %s)", duration) // <-- Log sukses dari Gemini
    // log.Printf("[DEBUG] Gemini Response: %s", response) // <-- Opsional: Log isi respons Gemini

    // Kirim respons
    w.Header().Set("Content-Type", "application/json")
    errEncode := json.NewEncoder(w).Encode(ChatResponse{Response: response})
    if errEncode != nil {
         log.Printf("[ERROR] Failed to encode or write response: %v", errEncode) // <-- Log jika gagal kirim respons
    } else {
         log.Println("[INFO] Successfully sent response to client.") // <-- Log sukses kirim respons
    }
}
