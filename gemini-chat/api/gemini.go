package api

import (
    // ... (import lainnya) ...
    "log" // <-- TAMBAHKAN import log
)

// ... (struct dan NewGeminiClient tetap sama) ...

func (c *GeminiClient) GenerateResponse(message string) (string, error) {
    // ... (persiapan url, reqBody, jsonData) ...
    if err != nil {
         log.Printf("[ERROR] GeminiClient: Error marshaling request: %v", err) // Log error marshal
         return "", fmt.Errorf("error marshaling request: %w", err)
    }

    // Buat permintaan HTTP
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
         log.Printf("[ERROR] GeminiClient: Error creating request: %v", err) // Log error create request
         return "", fmt.Errorf("error creating request: %w", err)
    }
    req.Header.Set("Content-Type", "application/json")

    // Kirim permintaan
    log.Println("[INFO] GeminiClient: Sending POST request to Gemini API...")
    client := &http.Client{
         // Opsional: Tambahkan Timeout agar tidak menunggu terlalu lama
         // Timeout: 30 * time.Second,
    }
    resp, err := client.Do(req)
    if err != nil {
         log.Printf("[ERROR] GeminiClient: Error sending request to Gemini: %v", err) // Log error kirim request
         return "", fmt.Errorf("error sending request: %w", err)
    }
    defer resp.Body.Close()
    log.Printf("[INFO] GeminiClient: Received response status: %s", resp.Status) // Log status respons

    // Baca body respons
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
         log.Printf("[ERROR] GeminiClient: Error reading response body: %v", err) // Log error baca body
         return "", fmt.Errorf("error reading response: %w", err)
    }

    // Periksa status kode HTTP
    if resp.StatusCode != http.StatusOK {
         log.Printf("[ERROR] GeminiClient: API error: %s - Body: %s", resp.Status, string(body)) // Log API error detail
         return "", fmt.Errorf("API error: %s - %s", resp.Status, string(body))
    }

    // Parse respons JSON
    // ... (lanjutkan dengan unmarshal dan cek lainnya seperti sebelumnya) ...
    // Anda bisa menambahkan log serupa untuk error unmarshal, prompt blocked, no candidates, dll.

    var geminiResp GeminiResponse
    if err := json.Unmarshal(body, &geminiResp); err != nil {
        log.Printf("[ERROR] GeminiClient: Error unmarshaling response: %v", err)
        return "", fmt.Errorf("error unmarshaling response: %w", err)
    }

    // Periksa apakah prompt diblokir
    if geminiResp.PromptFeedback.BlockReason != "" {
        log.Printf("[WARN] GeminiClient: Prompt blocked. Reason: %s", geminiResp.PromptFeedback.BlockReason)
        return "", fmt.Errorf("prompt blocked: %s", geminiResp.PromptFeedback.BlockReason)
    }

    // Periksa apakah ada kandidat respons
    if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
         log.Printf("[WARN] GeminiClient: No response candidates or empty content received.")
         return "", fmt.Errorf("no response candidates or empty content")
    }


    // Ekstrak teks respons
    responseText := ""
    for _, part := range geminiResp.Candidates[0].Content.Parts {
        responseText += part.Text
    }
    log.Printf("[INFO] GeminiClient: Successfully extracted response text.")

    return responseText, nil
}
