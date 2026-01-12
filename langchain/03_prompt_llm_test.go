package langchain

import (
	"context"
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func TestPrompt(t *testing.T) {
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
	messages, err := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate("Your are a {{.LanguageA}} Expert", []string{"LanguageA"}),
		prompts.NewHumanMessagePromptTemplate("Help me to translate the {{.LanguageA}} text to {{.LanguageB}},text: {{.text}}", []string{"LanguageA", "LanguageB", "text"}),
	}).FormatMessages(map[string]any{
		"LanguageA": "Chinese",
		"LanguageB": "English",
		"text":      "你好，世界！",
	})
	if err != nil {
		t.Fatal(err)
	}
	content := PromptChatMsg2OpenaiMsgContent(messages)
	t.Log(content)
	// call llm with the prompt(string type),but here we use MessageContent type
	r, err := llm.GenerateContent(ctx, content)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(r)
}

func PromptChatMsg2OpenaiMsgContent(messages []llms.ChatMessage) []llms.MessageContent {
	content := make([]llms.MessageContent, len(messages))
	for i, msg := range messages {
		content[i] = llms.TextParts(msg.GetType(), msg.GetContent())
	}
	return content
}
