package langchain

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/prompts"
)

func TestWindowsMemory(t *testing.T) {
	if os.Getenv("DASHSCOPE_API_KEY") == "" {
		t.Skip("DASHSCOPE_API_KEY is not set")
	}
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// ConversationWindowBuffer keeps only the last N conversation "pairs" (Human+AI).
	// When exceeded, it drops the oldest messages.
	mem := memory.NewConversationWindowBuffer(
		2, // keep only last 2 turns
		memory.WithMemoryKey("history"),
		memory.WithHumanPrefix("Human"),
		memory.WithAIPrefix("AI"),
	)

	chatPrompt := prompts.NewPromptTemplate(
		`You are a helpful assistant.
Always reply with exactly: OK

Conversation so far:
{{.history}}

Human: {{.input}}
AI:`,
		[]string{"input", "history"},
	)

	chatChain := chains.NewLLMChain(
		llm,
		chatPrompt,
		chains.WithMaxTokens(10),
		chains.WithTemperature(0),
	)
	chatChain.Memory = mem

	// Add 5 turns. With window size=2, only the last 2 turns should remain in history.
	turns := []string{
		"FACT0: alpha",
		"FACT1: beta",
		"FACT2: gamma",
		"FACT3: delta",
		"FACT4: epsilon",
	}
	for _, msg := range turns {
		_, err := chains.Run(ctx, chatChain, msg)
		if err != nil {
			t.Fatal(err)
		}
	}

	memVars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	history, _ := memVars["history"].(string)
	t.Logf("History buffer:\n%s", history)

	// Oldest should be pruned, newest should remain.
	if strings.Contains(history, "FACT0") || strings.Contains(history, "FACT1") || strings.Contains(history, "FACT2") {
		t.Fatalf("expected oldest turns to be pruned for window size=2, but history still contains early FACTs")
	}
	if !strings.Contains(history, "FACT3") || !strings.Contains(history, "FACT4") {
		t.Fatalf("expected latest turns to remain in history, but they are missing")
	}
}
