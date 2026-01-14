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

func TestTokenMemory(t *testing.T) {
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

	// ConversationTokenBuffer prunes old messages when the buffer exceeds MaxTokenLimit.
	// Note: token counting is approximate (CountTokens over the buffer string), but pruning behavior is deterministic.
	mem := memory.NewConversationTokenBuffer(
		llm,
		80, // intentionally small so we trigger pruning quickly， history must be 80 tokens or less
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
		chains.WithMaxTokens(10), // LLM will return when the token of output is reached 10
		chains.WithTemperature(0),
	)
	chatChain.Memory = mem

	// First turn contains a unique marker we expect to be pruned out later.
	_, err = chains.Run(ctx, chatChain, "FACT0: my secret code is 12345. Please remember it.")
	if err != nil {
		t.Fatal(err)
	}

	// Add enough long turns to exceed the token limit and force pruning.
	longMsg := strings.Repeat("lorem ipsum dolor sit amet ", 20)
	for i := 0; i < 8; i++ {
		_, err = chains.Run(ctx, chatChain, longMsg)
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

	if strings.Contains(history, "FACT0") {
		t.Fatalf("expected FACT0 to be pruned from token buffer history, but it still exists")
	}

	// Also sanity-check we still have *some* memory left.
	msgs, err := mem.ChatHistory.Messages(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) == 0 {
		t.Fatal("expected some messages to remain in memory after pruning")
	}
}
