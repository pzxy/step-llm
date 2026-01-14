package langchain

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

func TestCalculatorTool(t *testing.T) {
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

	// Tool: a simple calculator (input is an expression string).
	calculator := tools.Calculator{}

	// Agent: MRKL/ReAct-style agent that calls tools via plain-text "Action: ... / Action Input: ...".
	// This works with many "OpenAI-compatible" endpoints even when native function-calling isn't supported.
	agent := agents.NewOneShotAgent(
		llm,
		[]tools.Tool{calculator},
		agents.WithPromptFormatInstructions(
			mrklFormatInstructions([]string{calculator.Name()}),
		),
	)

	// Executor: runs the agent loop (plan -> tool -> observe -> ... -> final).
	executor := agents.NewExecutor(agent, agents.WithMaxIterations(6))

	// IMPORTANT: do NOT ask for "only the number" here; MRKL parsing expects "Final Answer:".
	result, err := chains.Run(ctx, executor, "What is 15 multiplied by 4?")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Agent response: %s", result)
	if !strings.Contains(result, "60") {
		t.Fatalf("expected result to contain 60, got: %q", result)
	}
}

func TestAgentWithMultipleTools(t *testing.T) {
	// Same MRKL-style tool usage as TestCalculatorTool, but with multiple tools to show selection.
	// We keep everything local (no network), so you can always "see the effect".
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

	calculator := tools.Calculator{}
	kv := keyValueTool{}
	upper := upperTool{}

	toolList := []tools.Tool{calculator, kv, upper}
	toolNames := []string{calculator.Name(), kv.Name(), upper.Name()}

	agent := agents.NewOneShotAgent(
		llm,
		toolList,
		agents.WithPromptFormatInstructions(mrklFormatInstructions(toolNames)), // Override the agent's default prompt template with a custom one
	)
	executor := agents.NewExecutor(agent, agents.WithMaxIterations(8))

	// The agent should use key_value to fetch "beijing_population_million", then calculator to add 1.
	// It might add explanation; we just assert the numeric signal is present.
	result, err := chains.Run(ctx, executor, "Using tools: fetch beijing_population_million, then compute (that + 1).")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Agent response: %s", result)
	if !strings.Contains(result, "22") { // key_value returns 21, +1 => 22
		t.Fatalf("expected result to contain 22, got: %q", result)
	}

	// Also show a non-math tool: uppercase.
	result2, err := chains.Run(ctx, executor, "Use upper tool to uppercase: langchaingo")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Agent response 2: %s", result2)
	if !strings.Contains(result2, "LANGCHAINGO") {
		t.Fatalf("expected result to contain LANGCHAINGO, got: %q", result2)
	}
}

func mrklFormatInstructions(toolNames []string) string {
	return "Use the following format:\n\n" +
		"Question: the input question you must answer\n" +
		"Thought: you should always think about what to do\n" +
		"Action: the action to take, should be one of [" + strings.Join(toolNames, ", ") + "]\n" +
		"Action Input: the input to the action\n" +
		"Observation: the result of the action\n" +
		"... (this Thought/Action/Action Input/Observation can repeat N times)\n" +
		"Thought: I now know the final answer\n" +
		"Final Answer: the final answer to the original question\n\n" +
		`IMPORTANT: your final response MUST start with "Final Answer:"`
}

func truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// --- Local demo tools (no network) ---

type keyValueTool struct{}

func (t keyValueTool) Name() string { return "key_value" }

func (t keyValueTool) Description() string {
	return `A tiny local key-value lookup tool.
Input should be one of: beijing_population_million, capital_of_china.`
}

func (t keyValueTool) Call(_ context.Context, input string) (string, error) {
	switch strings.TrimSpace(strings.ToLower(input)) {
	case "beijing_population_million":
		return "21", nil
	case "capital_of_china":
		return "Beijing", nil
	default:
		return "unknown_key", nil
	}
}

type upperTool struct{}

func (t upperTool) Name() string { return "upper" }

func (t upperTool) Description() string {
	return "Uppercase a string. Input should be any text."
}

func (t upperTool) Call(_ context.Context, input string) (string, error) {
	return strings.ToUpper(input), nil
}
