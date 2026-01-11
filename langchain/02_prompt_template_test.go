package langchain

import (
	"testing"
	"time"

	"github.com/tmc/langchaingo/prompts"
)

// https://pkg.go.dev/github.com/tmc/langchaingo/examples/prompt-template-example#section-readme
func TestPromptTemplate(t *testing.T) {
	// 1. prompt template
	s := prompts.PromptTemplate{
		Template: "{{.address}} time is {{.time}}.",
		InputVariables: []string{"address"},  // if address variable is not provided, will print error info about missing variable
		PartialVariables: map[string]any{ // predefine variable
			"time": time.Now().Format("15:04:05"),
		},
		TemplateFormat : prompts.TemplateFormatGoTemplate, // default is Go template,so this line can be omitted
	}
	o,err := s.FormatPrompt(map[string]any{
		"address": "Beijing",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Log(o)
	// use NewPromptTemplate constructor
	s2 := prompts.NewPromptTemplate("Summarize this {{.content}} in {{.style}} style", []string{"content", "style"})
	prompt, err := s2.FormatPrompt(map[string]any{
		"content": "LangChain is a framework for developing applications powered by language models.",
		"style":   "concise",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Log(prompt)
	// 2. chat prompt template
	chatTmpl := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{	
		prompts.NewSystemMessagePromptTemplate(
			"You are a helpful assistant.",
			nil,
		),
		prompts.NewHumanMessagePromptTemplate(
			"Translate the following text to {{.language}}:\n{{.text}}",
			[]string{"language", "text"},
		),
	})
	chatPrompt, err := chatTmpl.FormatPrompt(map[string]any{
		"language": "French",
		"text":     "Hello, how are you?",
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, msg := range chatPrompt.Messages() {
		t.Logf("%T: %s", msg, msg.GetContent())
	}
}