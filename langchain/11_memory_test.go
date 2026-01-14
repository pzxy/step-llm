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

func TestMemory(t *testing.T) {
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

	// ConversationBuffer is the simplest memory: it stores the chat history and exposes it as {{.history}}.
	// The chain runner will:
	// - LoadMemoryVariables() before calling the chain (injects "history" into prompt values)
	// - SaveContext() after the chain returns (appends Human/AI messages into memory)
	mem := memory.NewConversationBuffer(
		memory.WithMemoryKey("history"),
		memory.WithHumanPrefix("Human"),
		memory.WithAIPrefix("AI"),
	)

	chatPrompt := prompts.NewPromptTemplate(
		`You are a helpful assistant.

Conversation so far:
{{.history}}

Human: {{.input}}
AI:`,
		[]string{"input", "history"},
	)

	chatChain := chains.NewLLMChain(
		llm,
		chatPrompt,
		chains.WithMaxTokens(200),
		chains.WithTemperature(0.2),
	)
	chatChain.Memory = mem

	// Turn 1: store a fact in memory
	turn1, err := chains.Run(ctx, chatChain, "My name is Kwin. Please remember it.")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Turn1 AI: %s", turn1)

	// Turn 2: ask for that fact; require a constrained answer for easy assertion
	turn2, err := chains.Run(ctx, chatChain, "What is my name? Reply with only the name.")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Turn2 AI: %s", turn2)

	if !strings.Contains(strings.ToLower(turn2), "kwin") {
		// Print memory buffer to help debugging when a model ignores instructions.
		memVars, _ := mem.LoadMemoryVariables(ctx, nil)
		t.Fatalf("expected memory recall of name 'Kwin', got: %q (history=%v)", turn2, memVars["history"])
	}
}
