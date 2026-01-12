# AGENTS.md - Development Guide for step-llm

This guide provides essential information for AI agents working on the step-llm codebase, a Go-based project demonstrating LLM integration using the langchaingo library.

## Project Overview

step-llm is a learning/demo project that showcases various LLM (Large Language Model) patterns and integrations using Go and the langchaingo framework. The project focuses on prompt engineering, template usage, and API interactions with LLM services.

## Build and Development Commands

### Running Tests

**All tests:**
```bash
go test ./...
```

**Verbose output:**
```bash
go test -v ./...
```

**Single test file:**
```bash
go test -v langchain/01_helloworld_test.go
```

**Single test function:**
```bash
go test -v -run TestHelloWorld ./langchain
```

**Race detection:**
```bash
go test -race ./...
```

**Benchmark tests:**
```bash
go test -bench=. ./...
```

### Building

**Build the project:**
```bash
go build .
```

**Build with optimizations:**
```bash
go build -ldflags="-s -w" .
```

### Code Quality

**Format code (auto-fix):**
```bash
go fmt ./...
```

**Static analysis:**
```bash
go vet ./...
```

**Check for vulnerabilities:**
```bash
go mod tidy
go list -m -u all  # Check for updates
```

**Install dependencies:**
```bash
go mod download
```

**Clean module cache:**
```bash
go clean -modcache
```

## Code Style Guidelines

### Go Conventions

This project follows standard Go conventions and idioms:

#### File Organization
- Test files use `_test.go` suffix
- Package names are lowercase, single word when possible
- Related functionality grouped in subdirectories

#### Imports
```go
import (
    "context"
    "fmt"
    "log"
    "os"
    "testing"

    "github.com/tmc/langchaingo/llms"
    "github.com/tmc/langchaingo/llms/openai"
)
```
- Standard library imports first (alphabetically sorted)
- Blank line separates stdlib from third-party imports
- Third-party imports alphabetically sorted

#### Naming Conventions

**Variables and Functions:**
- camelCase for unexported identifiers
- PascalCase for exported identifiers
- Short, descriptive names preferred
- Single letter variables only for loops/indexes

**Constants:**
```go
const (
    maxRetries = 3
    defaultTimeout = 30 * time.Second
)
```

**Types:**
```go
type Config struct {
    APIKey     string
    ServerAddr string
}

type client struct {  // unexported
    config *Config
}
```

#### Error Handling

**Standard pattern:**
```go
result, err := someFunction()
if err != nil {
    return fmt.Errorf("failed to do something: %w", err)
}
```

**Test error handling:**
```go
if err != nil {
    t.Fatal(err)  // or t.Fatalf("context: %v", err)
}
```

**Logging in tests:**
```go
t.Log("informational message")
t.Logf("formatted: %s", value)
```

#### Context Usage

Always pass context for cancellable operations:
```go
ctx := context.Background()
result, err := llm.GenerateContent(ctx, content, llms.WithMaxTokens(1024))
```

#### Environment Variables

**API Configuration:**
- `API_KEY` - Primary API key
- `SERVER_ADDR` - Server endpoint
- `DASHSCOPE_API_KEY` - Alternative API key for Aliyun

**Usage pattern:**
```go
apiKey := os.Getenv("API_KEY")
if apiKey == "" {
    log.Fatal("API_KEY environment variable is required")
}
```

### Testing Patterns

#### Test Structure
```go
func TestFunctionName(t *testing.T) {
    // Setup
    // ...

    // Execute
    result, err := testedFunction(input)

    // Assert
    if err != nil {
        t.Fatal(err)
    }
    if result != expected {
        t.Errorf("expected %v, got %v", expected, result)
    }
}
```

#### Test Naming
- `TestXxx` for basic functionality
- `TestXxx_Error` for error cases
- `TestXxx_EdgeCase` for boundary conditions

#### Table-Driven Tests
```go
func TestPromptTemplate(t *testing.T) {
    tests := []struct {
        name     string
        template string
        input    map[string]any
        expected string
    }{
        {
            name:     "basic template",
            template: "Hello {{.name}}",
            input:    map[string]any{"name": "World"},
            expected: "Hello World",
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // test implementation
        })
    }
}
```

### LLM Integration Patterns

#### Basic LLM Call
```go
llm, err := openai.New(
    openai.WithBaseURL("https://api.example.com/v1"),
    openai.WithModel("gpt-3.5-turbo"),
)
if err != nil {
    return err
}

content := []llms.MessageContent{
    llms.TextParts(llms.ChatMessageTypeSystem, "You are a helpful assistant."),
    llms.TextParts(llms.ChatMessageTypeHuman, "Hello!"),
}

response, err := llm.GenerateContent(ctx, content, llms.WithMaxTokens(1024))
```

#### Prompt Templates
```go
template := prompts.NewPromptTemplate(
    "Summarize this {{.content}} in {{.style}} style",
    []string{"content", "style"},
)

prompt, err := template.FormatPrompt(map[string]any{
    "content": "text to summarize",
    "style":   "concise",
})
```

#### Chat Prompt Templates
```go
chatTmpl := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
    prompts.NewSystemMessagePromptTemplate("You are a helpful assistant.", nil),
    prompts.NewHumanMessagePromptTemplate("{{.question}}", []string{"question"}),
})
```

### Security Best Practices

#### API Keys
- Never hardcode API keys
- Use environment variables
- Validate required environment variables at startup
- Consider using secret management services for production

#### Input Validation
```go
func validateInput(input string) error {
    if strings.TrimSpace(input) == "" {
        return errors.New("input cannot be empty")
    }
    if len(input) > maxLength {
        return fmt.Errorf("input too long: max %d characters", maxLength)
    }
    return nil
}
```

### Performance Considerations

#### Context Timeouts
```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := llm.GenerateContent(ctx, content)
```

#### Resource Management
- Close HTTP clients when done
- Use defer for cleanup
- Handle goroutine lifecycles properly

### Common Patterns Used

#### Configuration Struct
```go
type Config struct {
    APIKey      string
    BaseURL     string
    Model       string
    MaxTokens   int
    Temperature float64
}

func NewConfig() *Config {
    return &Config{
        APIKey:      os.Getenv("API_KEY"),
        BaseURL:     getEnvOrDefault("BASE_URL", "https://api.openai.com/v1"),
        Model:       getEnvOrDefault("MODEL", "gpt-3.5-turbo"),
        MaxTokens:   1024,
        Temperature: 0.7,
    }
}
```

#### Error Wrapping
```go
var (
    ErrInvalidConfig = errors.New("invalid configuration")
    ErrAPIError      = errors.New("API error")
)

func (c *Client) CallAPI(ctx context.Context) error {
    // ...
    return fmt.Errorf("%w: %v", ErrAPIError, apiErr)
}
```

This guide should be updated as the project evolves. Always run `go fmt` and `go vet` before committing code changes.</content>
<parameter name="filePath">/home/kwin/code/step-llm/AGENTS.md