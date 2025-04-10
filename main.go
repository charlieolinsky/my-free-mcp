package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/exp/slog"
)

// OllamaRequest represents the request payload to Ollama 0.6.5
type OllamaRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	Stream    bool   `json:"stream"`
	MaxTokens int    `json:"max_tokens,omitempty"`
}

// OllamaResponse represents the response from Ollama 0.6.5
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func main() {
	var projectPath string

	// Initialize structured logging
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))

	rootCmd := &cobra.Command{
		Use:   "my-free-mcp",
		Short: "A CLI tool to assist with Ebitengine game development",
	}

	askCmd := &cobra.Command{
		Use:   "ask [question]",
		Short: "Ask a question about your Ebitengine project",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			question := args[0]
			contextData := getProjectContext(projectPath)
			response, err := queryOllama(ctx, question, contextData)
			if err != nil {
				return fmt.Errorf("failed to query Ollama: %w", err)
			}
			fmt.Println("Response:", response)
			return nil
		},
	}

	// Add --path flag
	askCmd.Flags().StringVarP(&projectPath, "path", "p", ".", "Path to the Ebitengine project directory")
	rootCmd.AddCommand(askCmd)

	if err := rootCmd.Execute(); err != nil {
		slog.Error("Failed to execute command", "error", err)
		os.Exit(1)
	}
}

// getProjectContext reads the file structure from the specified path
func getProjectContext(projectPath string) string {
	var sb strings.Builder
	err := filepath.WalkDir(projectPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && filepath.Ext(path) == ".go" {
			content, err := os.ReadFile(path)
			if err != nil {
				slog.Warn("Failed to read file", "path", path, "error", err)
				return nil // Skip file but continue walking
			}
			_, _ = sb.WriteString(fmt.Sprintf("File: %s\nContent:\n%s\n\n", path, string(content)))
		}
		return nil
	})
	if err != nil {
		slog.Warn("Error walking directory", "path", projectPath, "error", err)
	}
	return sb.String()
}

// queryOllama sends a prompt to the Ollama server and returns the response
func queryOllama(ctx context.Context, question, contextData string) (string, error) {
	prompt := fmt.Sprintf("You are a Go and Ebitengine expert. Here’s my project context:\n%s\nQuestion: %s", contextData, question)
	reqData := OllamaRequest{
		Model:     "qwen2.5-coder:7b-instruct",
		Prompt:    prompt,
		Stream:    false,
		MaxTokens: 500,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://localhost:11434/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request to Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code from Ollama: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	var ollamaResp OllamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if !ollamaResp.Done {
		slog.Warn("Response from Ollama is incomplete")
	}

	return ollamaResp.Response, nil
}
