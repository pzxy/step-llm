package langchain

import (
	"context"
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func TestPromptPlaceholder(t *testing.T) {
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	// build chat prompt template, and format to string type
	message,err := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate("Your are a {{.LanguageA}} Expert", []string{"LanguageA"}),
		prompts.NewHumanMessagePromptTemplate("Help me to translate the {{.LanguageA}} text to {{.LanguageB}},text: {{.text}}", []string{"LanguageA", "LanguageB", "text"}),
		prompts.MessagesPlaceholder{
			VariableName: "msgs",
		},
	}).Format(map[string]any{
		"LanguageA": "Chinese",
		"LanguageB": "English",
		"text":      "你好，世界！",
		"msgs":      []llms.ChatMessage{llms.HumanChatMessage{Content: "this is a placeholder message content"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Log(message)
	// call llm with the prompt(string type)
	resp, err := llm.Call(ctx, message)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(resp)
}
