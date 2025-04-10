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
	"regexp"
	"strings"
	"time"

	"github.com/fatih/color"
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
	Error    string `json:"error,omitempty"`
}

const (
	defaultOllamaURL = "http://localhost:11434/api/generate"
)

func main() {
	var projectPath, ollamaURL string

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})))

	rootCmd := &cobra.Command{
		Use:   "my-free-mcp",
		Short: "A CLI tool to assist with Ebitengine game development",
	}

	askCmd := &cobra.Command{
		Use:   "ask [question]",
		Short: "Ask a question about your Ebitengine project",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := context.WithTimeout(cmd.Context(), 60*time.Second)
			defer cancel()

			question := args[0]
			contextData := getProjectContext(projectPath)
			response, err := queryOllama(ctx, ollamaURL, question, contextData)
			if err != nil {
				return fmt.Errorf("failed to query Ollama: %w", err)
			}
			printFormattedResponse(response)
			return nil
		},
	}

	askCmd.Flags().StringVarP(&projectPath, "path", "p", ".", "Path to the Ebitengine project directory")
	askCmd.Flags().StringVar(&ollamaURL, "ollama-url", defaultOllamaURL, "URL of the Ollama server")
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
				return nil
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
func queryOllama(ctx context.Context, url, question, contextData string) (string, error) {
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

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
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
		body, _ := io.ReadAll(resp.Body)
		slog.Warn("Ollama request failed", "status", resp.StatusCode, "body", string(body))
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

	if ollamaResp.Error != "" {
		return "", fmt.Errorf("Ollama returned an error: %s", ollamaResp.Error)
	}

	if !ollamaResp.Done {
		slog.Warn("Response from Ollama is incomplete")
	}

	return ollamaResp.Response, nil
}

// printFormattedResponse formats and prints the response for better readability
func printFormattedResponse(response string) {
	// Colors for different elements
	headingColor := color.New(color.FgCyan, color.Bold)
	listColor := color.New(color.FgYellow)
	codeColor := color.New(color.FgGreen)
	textColor := color.New(color.FgWhite)

	// Split response into lines
	lines := strings.Split(response, "\n")

	// Regular expressions for detecting code blocks and lists
	codeBlockStart := regexp.MustCompile("```(?:go)?")
	listItem := regexp.MustCompile(`^\d+\.\s+`)

	inCodeBlock := false
	for _, line := range lines {
		line = strings.TrimSpace(line)

		if codeBlockStart.MatchString(line) {
			inCodeBlock = !inCodeBlock
			if inCodeBlock {
				fmt.Println() // Add spacing before code
			}
			continue
		}

		if line == "" {
			fmt.Println() // Preserve paragraph breaks
			continue
		}

		switch {
		case inCodeBlock:
			codeColor.Printf("  %s\n", line) // Indent code
		case listItem.MatchString(line):
			listColor.Printf("%s\n", line)
		case strings.HasPrefix(line, "**") && strings.HasSuffix(line, "**"):
			headingColor.Printf("%s\n", strings.Trim(line, "*"))
		default:
			textColor.Printf("%s\n", line)
		}
	}
}
