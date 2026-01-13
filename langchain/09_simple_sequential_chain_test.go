package langchain

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func TestSimpleSequentialChain(t *testing.T) {
	// SimpleSequentialChain = pipeline where each chain has:
	// - exactly 1 input key
	// - exactly 1 output key
	// The output of chain1 becomes the input of chain2, etc.
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

	// Chain 1: topic -> short outline
	outlineChain := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Create a 3-bullet outline for a short blog post about: {{.input}}. Return only the outline.",
			[]string{"input"},
		),
		chains.WithMaxTokens(180),
		chains.WithTemperature(0.3),
	)

	// Chain 2: outline -> final post
	writeChain := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Write the full blog post from this outline:\n{{.input}}\nKeep it under 150 words.",
			[]string{"input"},
		),
		chains.WithMaxTokens(260),
		chains.WithTemperature(0.7),
	)
	seq, err := chains.NewSimpleSequentialChain([]chains.Chain{outlineChain, writeChain})
	if err != nil {
		t.Fatal(err)
	}

	post, err := chains.Run(ctx, seq, "LangChain-style chaining patterns in Go (langchaingo)")
	if err != nil {
		t.Fatal(err)
	}
	if post == "" {
		t.Fatal("expected non-empty output")
	}
	t.Log(post)
}

func TestSequentialChain(t *testing.T) {
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

	// SequentialChain = multiple chains in order, where each chain can use custom input/output keys.
	// Important: each chain's output keys must be unique across the sequence.
	outline := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Create a 3-bullet outline for: {{.topic}}. Return only the outline.",
			[]string{"topic"},
		),
		chains.WithMaxTokens(180),
		chains.WithTemperature(0.3),
	)
	outline.OutputKey = "outline"

	draft := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Write a short draft from this outline:\n{{.outline}}\nKeep it under 150 words.",
			[]string{"outline"},
		),
		chains.WithMaxTokens(260),
		chains.WithTemperature(0.7),
	)
	draft.OutputKey = "draft"

	title := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Generate a catchy title for this draft:\n{{.draft}}\nReturn only the title.",
			[]string{"draft"},
		),
		chains.WithMaxTokens(40),
		chains.WithTemperature(0.7),
	)
	title.OutputKey = "title"

	seq, err := chains.NewSequentialChain(
		[]chains.Chain{outline, draft, title},
		[]string{"topic"},
		[]string{"title"},
	)
	if err != nil {
		t.Fatal(err)
	}

	out, err := chains.Call(ctx, seq, map[string]any{
		"topic": "LangChain-style chaining patterns in Go (langchaingo)",
	})
	if err != nil {
		t.Fatal(err)
	}
	if out["title"] == "" {
		t.Fatalf("expected non-empty title, got: %#v", out["title"])
	}
	t.Logf("Title: %v", out["title"])
}
