package langchain

import (
	"context"
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func TestPromptFewshotChatTemplate(t *testing.T) {
	// Set up API connection (same as other tests)
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Few-shot learning with chat messages: teaching a custom JSON response format
	// Without examples, the LLM might respond in natural language or inconsistent formats
	// This demonstrates the power of combining few-shot examples with chat message structure

	// Approach 1: Static few-shot examples using MessagesPlaceholder
	// This is the recommended way to include few-shot examples in chat templates

	// Define few-shot example conversations (system + human + AI response)
	fewShotMovieExamples := [][]llms.ChatMessage{
		{
			llms.SystemChatMessage{Content: "You are a movie rating assistant."},
			llms.HumanChatMessage{Content: "Rate the movie 'The Shawshank Redemption'"},
			llms.AIChatMessage{Content: `{"movie": "The Shawshank Redemption", "rating": 9.3, "genre": "Drama", "year": 1994}`},
		},
		{
			llms.SystemChatMessage{Content: "You are a movie rating assistant."},
			llms.HumanChatMessage{Content: "Rate the movie 'Inception'"},
			llms.AIChatMessage{Content: `{"movie": "Inception", "rating": 8.8, "genre": "Sci-Fi", "year": 2010}`},
		},
	}

	// Flatten examples into a single message slice
	flatMovieExamples := make([]llms.ChatMessage, 0)
	for _, ex := range fewShotMovieExamples {
		flatMovieExamples = append(flatMovieExamples, ex...)
	}

	// Create template with MessagesPlaceholder for few-shot examples
	chatTmpl := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate("You are a movie rating assistant.", nil),
		prompts.NewHumanMessagePromptTemplate("Rate the movie '{{.movie}}'", []string{"movie"}),
		// MessagesPlaceholder injects the few-shot examples before the test question
		prompts.MessagesPlaceholder{VariableName: "examples"},
		prompts.NewSystemMessagePromptTemplate("You are a movie rating assistant.", nil),
		prompts.NewHumanMessagePromptTemplate("Rate the movie '{{.movie}}'", []string{"movie"}),
	})

	// Format the chat prompt with test input and few-shot examples
	messages, err := chatTmpl.FormatMessages(map[string]any{
		"movie":    "The Matrix",
		"examples": flatMovieExamples,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Convert ChatMessage to MessageContent for LLM call
	content := make([]llms.MessageContent, len(messages))
	for i, msg := range messages {
		content[i] = llms.TextParts(msg.GetType(), msg.GetContent())
	}

	t.Log("Few-shot chat prompt (movie rating in JSON format):")
	for i, msg := range messages {
		t.Logf("Message %d [%T]: %s", i+1, msg, msg.GetContent())
	}

	// Call LLM - with few-shot examples, it should return JSON in the correct format
	resp, err := llm.GenerateContent(ctx, content)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("LLM response (should be JSON with movie, rating, genre, year):")
	t.Log(resp.Choices[0].Content)

	// Approach 2: Dynamic few-shot examples with MessagesPlaceholder (geography example)
	t.Log("\n--- Approach 2: Dynamic few-shot examples (geography) ---")

	// Define geography few-shot examples (human + AI, no system message needed since it's in template)
	fewShotGeographyExamples := [][]llms.ChatMessage{
		{
			llms.HumanChatMessage{Content: "What is the capital of France?"},
			llms.AIChatMessage{Content: "Paris. (Latitude: 48.8566° N, Longitude: 2.3522° E)"},
		},
		{
			llms.HumanChatMessage{Content: "What is the capital of Japan?"},
			llms.AIChatMessage{Content: "Tokyo. (Latitude: 35.6762° N, Longitude: 139.6503° E)"},
		},
	}

	// Flatten examples into a single message slice
	flatGeographyExamples := make([]llms.ChatMessage, 0)
	for _, ex := range fewShotGeographyExamples {
		flatGeographyExamples = append(flatGeographyExamples, ex...)
	}

	// Create template with MessagesPlaceholder for few-shot examples
	dynamicChatTmpl := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate("You are a geography assistant. Always provide coordinates in your answers.", nil),
		prompts.NewHumanMessagePromptTemplate("What is the capital of {{.country}}?", []string{"country"}),
		// MessagesPlaceholder injects few-shot examples before the test question
		prompts.MessagesPlaceholder{VariableName: "examples"},
		prompts.NewHumanMessagePromptTemplate("What is the capital of {{.country}}?", []string{"country"}),
	})

	// Format with few-shot examples
	dynamicMessages, err := dynamicChatTmpl.FormatMessages(map[string]any{
		"country":  "Germany",
		"examples": flatGeographyExamples,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Convert and call LLM
	dynamicContent := make([]llms.MessageContent, len(dynamicMessages))
	for i, msg := range dynamicMessages {
		dynamicContent[i] = llms.TextParts(msg.GetType(), msg.GetContent())
	}

	t.Log("Dynamic few-shot chat prompt (geography with coordinates):")
	for i, msg := range dynamicMessages {
		t.Logf("Message %d [%T]: %s", i+1, msg, msg.GetContent())
	}

	dynamicResp, err := llm.GenerateContent(ctx, dynamicContent)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("LLM response (should be 'Berlin' with coordinates):")
	t.Log(dynamicResp.Choices[0].Content)
}
