package langchain

import (
	"context"
	"fmt"
	"log"
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func TestDemo(t *testing.T) {
	// aliyun dashscope qwen api key
	// export DASHSCOPE_API_KEY="your_api_key" > ~/.zshrc
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a company branding design wizard."),
		llms.TextParts(llms.ChatMessageTypeHuman, "What would be a good company name a company that makes colorful socks?"),
	}

	// if _, err := llm.GenerateContent(ctx, content,
	// 	llms.WithMaxTokens(1024),
	// 	llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
	// 		fmt.Print(string(chunk))
	// 		return nil
	// 	})); err != nil {
	// 	log.Fatal(err)
	// }
	r, err := llm.GenerateContent(ctx, content, llms.WithMaxTokens(1024))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(r.Choices[0].Content)

}