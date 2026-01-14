package langchain

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
)

func TestMathChain(t *testing.T) {
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

	// LLMMathChain asks the LLM to generate a starlark expression and evaluates it locally.
	// Output is the computed result as string.
	chain := chains.NewLLMMathChain(llm)

	q := "what is forty plus three? take that then multiply it by ten thousand divided by 7324.3"
	answer, err := chains.Run(ctx, chain, q)
	if err != nil {
		t.Fatal(err)
	}
	// Expect around 58.708...
	if !strings.Contains(answer, "58.708") {
		t.Fatalf("unexpected answer: %q", answer)
	}
	t.Logf("Q: %s", q)
	t.Logf("A: %s", answer)
}
