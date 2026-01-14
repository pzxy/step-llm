package langchain

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/tools"
)

func TestTypeAgent(t *testing.T) {
	if os.Getenv("DASHSCOPE_API_KEY") == "" {
		t.Skip("DASHSCOPE_API_KEY is not set")
	}
	if os.Getenv("TAVILY_API_KEY") == "" {
		t.Skip("TAVILY_API_KEY is not set")
	}
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	// "Type agent" here demonstrates selecting agent type via agents.Initialize(...).
	// ConversationalReactDescription keeps a conversational prompt style.
	tavily := tavilySearchTool{apiKey: os.Getenv("TAVILY_API_KEY")}
	executor, err := agents.Initialize(
		llm,
		[]tools.Tool{tavily},
		agents.ConversationalReactDescription, // Use the conversational agent type
		agents.WithReturnIntermediateSteps(),
		agents.WithMaxIterations(6),
		agents.WithPromptFormatInstructions(conversationalFormatInstructions([]string{tavily.Name()})),
	)
	if err != nil {
		t.Fatal(err)
	}

	out, err := chains.Call(ctx, executor, map[string]any{
		"input": "Use tavily_search to find what Tavily API is used for. Include one source URL in your answer.",
	})
	if err != nil {
		t.Fatal(err)
	}
	assertToolUsed(t, out, tavily.Name())

	final, _ := out["output"].(string)
	t.Logf("Final Answer: %s", final)
	if !strings.Contains(final, "http") {
		t.Fatalf("expected a URL in final answer, got: %q", final)
	}
}

func TestReactAgent(t *testing.T) {
	if os.Getenv("DASHSCOPE_API_KEY") == "" {
		t.Skip("DASHSCOPE_API_KEY is not set")
	}
	if os.Getenv("TAVILY_API_KEY") == "" {
		t.Skip("TAVILY_API_KEY is not set")
	}
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	// ReAct agent: OneShotAgent (MRKL) + Executor loop + tools.
	tavily := tavilySearchTool{apiKey: os.Getenv("TAVILY_API_KEY")}
	calculator := tools.Calculator{}

	toolList := []tools.Tool{tavily, calculator}
	toolNames := []string{tavily.Name(), calculator.Name()}

	agent := agents.NewOneShotAgent( // Use the one-shot agent type
		llm,
		toolList,
		agents.WithPromptFormatInstructions(mrklFormatInstructions(toolNames)), // Override the agent's default prompt template with a custom one
	)
	executor := agents.NewExecutor(agent, agents.WithMaxIterations(8), agents.WithReturnIntermediateSteps())

	out, err := chains.Call(ctx, executor, map[string]any{
		"input": "Use tavily_search to find the number of days in a leap year, then use calculator to compute (that * 2). Include the number in Final Answer.",
	})
	if err != nil {
		t.Fatal(err)
	}
	assertToolUsed(t, out, tavily.Name())
	assertToolUsed(t, out, calculator.Name())

	final, _ := out["output"].(string)
	t.Logf("Final Answer: %s", final)
	if !strings.Contains(final, "732") {
		t.Fatalf("expected 732 in final answer (366*2), got: %q", final)
	}
}

// --- Helpers / local Tavily tool ---

// tavilySearchTool is a minimal tools.Tool implementation for Tavily Search API.
type tavilySearchTool struct {
	apiKey string
}

func (t tavilySearchTool) Name() string { return "tavily_search" }

func (t tavilySearchTool) Description() string {
	return `Search the web using Tavily. Input should be a search query. Returns top results with URLs.`
}

func (t tavilySearchTool) Call(ctx context.Context, input string) (string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "empty query", nil
	}
	if t.apiKey == "" {
		return "", fmt.Errorf("missing TAVILY_API_KEY")
	}

	body := map[string]any{
		"query":        input,
		"search_depth": "advanced",
		"max_results":  3,
	}
	bs, err := json.Marshal(body)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.tavily.com/search", bytes.NewReader(bs))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+t.apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("tavily search HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed struct {
		Results []struct {
			Title   string  `json:"title"`
			URL     string  `json:"url"`
			Content string  `json:"content"`
			Score   float64 `json:"score"`
		} `json:"results"`
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		// Fallback: return raw JSON so the agent can still use it.
		return strings.TrimSpace(string(respBody)), nil
	}

	var b strings.Builder
	if strings.TrimSpace(parsed.Answer) != "" {
		b.WriteString("Answer: ")
		b.WriteString(strings.TrimSpace(parsed.Answer))
		b.WriteString("\n")
	}
	for i, r := range parsed.Results {
		if i >= 3 {
			break
		}
		b.WriteString(fmt.Sprintf("[%d] %s\n%s\n", i+1, strings.TrimSpace(r.Title), strings.TrimSpace(r.URL)))
		if c := strings.TrimSpace(r.Content); c != "" {
			if len(c) > 300 {
				c = c[:300] + "..."
			}
			b.WriteString(c)
			b.WriteString("\n")
		}
	}
	out := strings.TrimSpace(b.String())
	if out == "" {
		return "no results", nil
	}
	return out, nil
}

func assertToolUsed(t *testing.T, out map[string]any, toolName string) {
	t.Helper()

	rawSteps, ok := out["intermediateSteps"]
	if !ok {
		t.Fatalf("expected intermediateSteps in output; enable agents.WithReturnIntermediateSteps()")
	}
	steps, ok := rawSteps.([]schema.AgentStep)
	if !ok {
		t.Fatalf("unexpected intermediateSteps type: %T", rawSteps)
	}
	for _, s := range steps {
		if strings.EqualFold(s.Action.Tool, toolName) {
			return
		}
	}
	t.Fatalf("expected tool %q to be used, but it was not. steps=%v", toolName, steps)
}

func conversationalFormatInstructions(toolNames []string) string {
	// ConversationalAgent finishes when output contains "AI:" (see agents/conversational.go).
	return "Use the following format:\n\n" +
		"Question: the input question you must answer\n" +
		"Thought: you should always think about what to do\n" +
		"Action: the action to take, should be one of [" + strings.Join(toolNames, ", ") + "]\n" +
		"Action Input: the input to the action\n" +
		"Observation: the result of the action\n" +
		"... (this Thought/Action/Action Input/Observation can repeat N times)\n" +
		"Thought: I now know the final answer\n" +
		"AI: the final answer to the original question\n\n" +
		`IMPORTANT: your final response MUST contain "AI:"`
}
