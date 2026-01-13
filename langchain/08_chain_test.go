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

func TestChain(t *testing.T) {
	// This test demonstrates the "Chain" idea from LangChain (Python) using langchaingo (Go):
	// - LLMChain = (PromptTemplate + LLM) -> text
	// - SimpleSequentialChain = run multiple chains in a pipeline (output of one becomes input of next)

	apiKey := firstEnvNonEmpty("OPENAI_API_KEY", "API_KEY", "DASHSCOPE_API_KEY", "POLOAI_API_KEY")
	if apiKey == "" {
		t.Skip("no API key found; set one of OPENAI_API_KEY / API_KEY / DASHSCOPE_API_KEY / POLOAI_API_KEY to run this test")
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		os.Setenv("OPENAI_API_KEY", apiKey)
	}

	baseURL := os.Getenv("BASE_URL")
	model := os.Getenv("MODEL")
	if baseURL == "" && os.Getenv("DASHSCOPE_API_KEY") != "" {
		baseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
		if model == "" {
			model = "qwen-long-latest"
		}
	}
	if baseURL == "" && os.Getenv("POLOAI_API_KEY") != "" {
		baseURL = "https://fast.poloai.top/v1"
		if model == "" {
			model = "gpt-5.2"
		}
	}
	if model == "" {
		// Safe default; override via MODEL env var if you want.
		model = "gpt-4o-mini"
	}

	openaiOpts := []openai.Option{
		openai.WithModel(model),
	}
	if baseURL != "" {
		openaiOpts = append(openaiOpts, openai.WithBaseURL(baseURL))
	}
	llm, err := openai.New(openaiOpts...)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// ------------------------------------------------------------
	// 1) Single LLMChain (PromptTemplate -> LLM -> output string)
	// ------------------------------------------------------------
	sloganChain := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Write a catchy 1-sentence marketing slogan for: {{.product}}. Return only the slogan.",
			[]string{"product"},
		),
		chains.WithMaxTokens(80),
		chains.WithTemperature(0.7),
	)
	sloganChain.OutputKey = "slogan"

	sloganOut, err := chains.Call(ctx, sloganChain, map[string]any{
		"product": "a Go library that brings LangChain-style chaining to Go",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Slogan: %v", sloganOut["slogan"])

	// ------------------------------------------------------------
	// 2) SimpleSequentialChain (pipeline: chain1 output -> chain2 input)
	// ------------------------------------------------------------
	// Note: SimpleSequentialChain requires each chain to have exactly:
	// - one input variable (we use {{.input}})
	// - one output key (LLMChain default output is "text")
	makeSlogan := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Write a catchy 1-sentence marketing slogan for: {{.input1}}. Return only the slogan.",
			[]string{"input1"},
		),
		chains.WithMaxTokens(80),
		chains.WithTemperature(0.7),
	)
	makeHashtags := chains.NewLLMChain(
		llm,
		prompts.NewPromptTemplate(
			"Given this slogan:\n{{.input2}}\nGenerate 5 short hashtags. Return as a comma-separated list only.",
			[]string{"input2"},
		),
		chains.WithMaxTokens(60),
		chains.WithTemperature(0.3),
	)

	pipe, err := chains.NewSimpleSequentialChain([]chains.Chain{makeSlogan, makeHashtags})
	if err != nil {
		t.Fatal(err)
	}
	hashtagsCSV, err := chains.Run(ctx, pipe, "a Go library that brings LangChain-style chaining to Go")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Hashtags: %s", hashtagsCSV)
}

func firstEnvNonEmpty(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}
