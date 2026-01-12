package langchain

import (
	"context"
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func TestPromptFewshotTemplate(t *testing.T) {
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

	// Simple but counterintuitive: convert letters to alphabet positions
	// Rule: Each letter becomes its position in the alphabet (a=1, b=2, c=3, etc.)
	// This is counterintuitive because why would letters become numbers?
	// Without examples, the AI has no reason to know this mapping!

	examplePrompt := prompts.NewPromptTemplate(
		"Word: {{.word}}\nNumbers: {{.numbers}}",
		[]string{"word", "numbers"},
	)

	// Two simple examples
	examples := []map[string]string{
		{
			"word":    "cat", // c=3, a=1, t=20
			"numbers": "3-1-20",
		},
		{
			"word":    "dog", // d=4, o=15, g=7
			"numbers": "4-15-7",
		},
	}

	fewshotPrompt, err := prompts.NewFewShotPrompt(
		examplePrompt, // [1] Template to format a single example (defines "Word: ... Numbers: ..." structure)
		examples,      // [2] Few-shot example dataset (2 samples: cat→3-1-20, dog→4-15-7)
		nil,           // [3] ExampleSelector: nil = use all examples (no dynamic filtering)
		"Convert words to alphabet position numbers:\n",     // [4] Prefix: Task instruction shown before all examples
		"\n\nConvert this word:\nWord: {{.word}}\nNumbers:", // [5] Suffix: Instruction to process the input word, with template variable {{.word}}
		[]string{"word"},                 // [6] Input variables: Only "word" is needed to render the final prompt
		nil,                              // [7] Example separator: nil = use default "\n" to split multiple examples
		"\n",                             // [8] Human message prefix: Use a newline to keep the prompt clean (no extra role labels)
		prompts.TemplateFormatGoTemplate, // [9] Template format: Use Go's native template syntax ({{.var}})
		true,                             // [10] Validate template: Enable syntax & variable check to catch errors early
	)
	if err != nil {
		t.Fatal(err)
	}

	// Test with "bird" - should become "2-9-18-4" (b=2, i=9, r=18, d=4)
	promptText, err := fewshotPrompt.Format(map[string]any{
		"word": "bird",
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Fewshot prompt (letter to number conversion):")
	t.Log(promptText)

	// Call LLM - without examples, it might guess wrong or refuse!
	resp, err := llm.Call(ctx, promptText)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("LLM response (should be '2-9-18-4'):")
	t.Log(resp)
}
