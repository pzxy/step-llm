package langchain

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/outputparser"
	"github.com/tmc/langchaingo/prompts"
)

// TestOutputParser demonstrates all output parsers in langchaingo:
// 1. Simple - returns raw text as-is
// 2. CommaSeparatedList - parses comma-separated values into a string slice
// 3. BooleanParser - parses boolean values (YES/NO, TRUE/FALSE)
// 4. Structured - parses JSON output into map[string]string with schema validation
// 5. Defined - parses JSON output into typed Go structs
// 6. RegexParser - extracts values using regex named groups
func TestOutputParser(t *testing.T) {
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

	// ============================================================
	// 1. Simple Parser - returns raw text as-is (no processing)
	// ============================================================
	t.Log("\n--- 1. Simple Parser ---")
	simpleParser := outputparser.NewSimple()
	t.Logf("Format Instructions: %q", simpleParser.GetFormatInstructions())

	simpleResult, err := simpleParser.Parse("  Hello, World!  ")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Simple parser result: %q", simpleResult)

	// ============================================================
	// 2. CommaSeparatedList Parser - parses CSV into []string
	// ============================================================
	t.Log("\n--- 2. CommaSeparatedList Parser ---")
	csvParser := outputparser.NewCommaSeparatedList()
	t.Logf("Format Instructions: %s", csvParser.GetFormatInstructions())

	// Create prompt with format instructions
	csvTemplate := prompts.NewPromptTemplate(
		"List 5 popular programming languages. {{.format_instructions}}",
		[]string{"format_instructions"},
	)
	csvPrompt, err := csvTemplate.Format(map[string]any{
		"format_instructions": csvParser.GetFormatInstructions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Prompt: %s", csvPrompt)

	// Call LLM
	csvContent := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, csvPrompt),
	}
	csvResp, err := llm.GenerateContent(ctx, csvContent)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LLM raw response: %s", csvResp.Choices[0].Content)

	// Parse the response
	csvResult, err := csvParser.Parse(csvResp.Choices[0].Content)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Parsed result ([]string): %v", csvResult)
	t.Logf("Number of items: %d", len(csvResult))

	// ============================================================
	// 3. BooleanParser - parses YES/NO, TRUE/FALSE
	// ============================================================
	t.Log("\n--- 3. BooleanParser ---")
	boolParser := outputparser.NewBooleanParser()
	t.Logf("Format Instructions: %s", boolParser.GetFormatInstructions())

	// Create prompt asking for a yes/no answer
	boolTemplate := prompts.NewPromptTemplate(
		"Is Python a programming language? Answer with only YES or NO. {{.format_instructions}}",
		[]string{"format_instructions"},
	)
	boolPrompt, err := boolTemplate.Format(map[string]any{
		"format_instructions": boolParser.GetFormatInstructions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Prompt: %s", boolPrompt)

	// Call LLM
	boolContent := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, boolPrompt),
	}
	boolResp, err := llm.GenerateContent(ctx, boolContent)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LLM raw response: %s", boolResp.Choices[0].Content)

	// Parse the response
	boolResult, err := boolParser.Parse(boolResp.Choices[0].Content)
	if err != nil {
		t.Logf("Parse error (may happen if LLM adds extra text): %v", err)
	} else {
		t.Logf("Parsed result (bool): %v", boolResult)
	}

	// ============================================================
	// 4. Structured Parser - parses JSON with schema validation
	// ============================================================
	t.Log("\n--- 4. Structured Parser ---")
	structuredParser := outputparser.NewStructured([]outputparser.ResponseSchema{
		{Name: "name", Description: "The name of the person"},
		{Name: "age", Description: "The age of the person"},
		{Name: "occupation", Description: "The person's job or profession"},
	})
	t.Logf("Format Instructions:\n%s", structuredParser.GetFormatInstructions())

	// Create prompt with format instructions
	structuredTemplate := prompts.NewPromptTemplate(
		"Generate information about a fictional software engineer. {{.format_instructions}}",
		[]string{"format_instructions"},
	)
	structuredPrompt, err := structuredTemplate.Format(map[string]any{
		"format_instructions": structuredParser.GetFormatInstructions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Prompt: %s", structuredPrompt)

	// Call LLM
	structuredContent := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, structuredPrompt),
	}
	structuredResp, err := llm.GenerateContent(ctx, structuredContent)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LLM raw response:\n%s", structuredResp.Choices[0].Content)

	// Parse the response
	structuredResult, err := structuredParser.Parse(structuredResp.Choices[0].Content)
	if err != nil {
		t.Logf("Parse error: %v", err)
	} else {
		t.Logf("Parsed result (map[string]string): %v", structuredResult)
	}

	// ============================================================
	// 5. Defined Parser - parses JSON into typed Go structs
	// ============================================================
	t.Log("\n--- 5. Defined Parser (Typed Struct) ---")

	// Define the struct for the expected output
	type MovieReview struct {
		Title  string `json:"title" describe:"movie title"`
		Rating int    `json:"rating" describe:"rating from 1 to 10"`
		Genre  string `json:"genre" describe:"movie genre"`
		Review string `json:"review" describe:"brief review in one sentence"`
	}

	definedParser, err := outputparser.NewDefined(MovieReview{})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Format Instructions:\n%s", definedParser.GetFormatInstructions())

	// Create prompt with format instructions
	definedTemplate := prompts.NewPromptTemplate(
		"Generate a movie review for a sci-fi movie. {{.format_instructions}}",
		[]string{"format_instructions"},
	)
	definedPrompt, err := definedTemplate.Format(map[string]any{
		"format_instructions": definedParser.GetFormatInstructions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Prompt: %s", definedPrompt)

	// Call LLM
	definedContent := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, definedPrompt),
	}
	definedResp, err := llm.GenerateContent(ctx, definedContent)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LLM raw response:\n%s", definedResp.Choices[0].Content)

	// Parse the response into typed struct
	definedResult, err := definedParser.Parse(definedResp.Choices[0].Content)
	if err != nil {
		t.Logf("Parse error: %v", err)
	} else {
		t.Logf("Parsed result (MovieReview struct):")
		t.Logf("  Title: %s", definedResult.Title)
		t.Logf("  Rating: %d", definedResult.Rating)
		t.Logf("  Genre: %s", definedResult.Genre)
		t.Logf("  Review: %s", definedResult.Review)
	}

	// ============================================================
	// 6. RegexParser - extracts values using regex named groups
	// ============================================================
	t.Log("\n--- 6. RegexParser ---")

	// Create regex parser to extract name and score
	regexParser := outputparser.NewRegexParser(`Name: (?P<name>\w+), Score: (?P<score>\d+)`)
	t.Logf("Format Instructions: %s", regexParser.GetFormatInstructions())

	// Create prompt asking for specific format
	regexTemplate := prompts.NewPromptTemplate(
		"Generate a player name and their score. Format your response EXACTLY as: Name: [name], Score: [score]",
		nil,
	)
	regexPrompt, err := regexTemplate.Format(nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Prompt: %s", regexPrompt)

	// Call LLM
	regexContent := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, regexPrompt),
	}
	regexResp, err := llm.GenerateContent(ctx, regexContent)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LLM raw response: %s", regexResp.Choices[0].Content)

	// Parse the response
	regexResult, err := regexParser.Parse(regexResp.Choices[0].Content)
	if err != nil {
		t.Logf("Parse error: %v", err)
	} else {
		t.Logf("Parsed result (map[string]string): %v", regexResult)
		if name, ok := regexResult.(map[string]string)["name"]; ok {
			t.Logf("  Extracted name: %s", name)
		}
		if score, ok := regexResult.(map[string]string)["score"]; ok {
			t.Logf("  Extracted score: %s", score)
		}
	}
}

// TestOutputParserUnit demonstrates output parsers without LLM calls
// This is useful for unit testing parser logic
func TestOutputParserUnit(t *testing.T) {
	// ============================================================
	// Unit tests for parsers (no LLM required)
	// ============================================================

	t.Run("Simple", func(t *testing.T) {
		parser := outputparser.NewSimple()
		result, err := parser.Parse("  trimmed  ")
		if err != nil {
			t.Fatal(err)
		}
		if result != "trimmed" {
			t.Errorf("expected 'trimmed', got '%v'", result)
		}
	})

	t.Run("CommaSeparatedList", func(t *testing.T) {
		parser := outputparser.NewCommaSeparatedList()
		result, err := parser.Parse("apple, banana, cherry")
		if err != nil {
			t.Fatal(err)
		}
		expected := []string{"apple", "banana", "cherry"}
		if len(result) != len(expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
		for i, v := range result {
			if v != expected[i] {
				t.Errorf("at index %d: expected '%s', got '%s'", i, expected[i], v)
			}
		}
	})

	t.Run("BooleanParser", func(t *testing.T) {
		parser := outputparser.NewBooleanParser()

		// Test TRUE values
		for _, input := range []string{"YES", "yes", "Yes", "TRUE", "true", "True"} {
			result, err := parser.Parse(input)
			if err != nil {
				t.Errorf("unexpected error for '%s': %v", input, err)
			}
			if result != true {
				t.Errorf("expected true for '%s', got %v", input, result)
			}
		}

		// Test FALSE values
		for _, input := range []string{"NO", "no", "No", "FALSE", "false", "False"} {
			result, err := parser.Parse(input)
			if err != nil {
				t.Errorf("unexpected error for '%s': %v", input, err)
			}
			if result != false {
				t.Errorf("expected false for '%s', got %v", input, result)
			}
		}
	})

	t.Run("Structured", func(t *testing.T) {
		parser := outputparser.NewStructured([]outputparser.ResponseSchema{
			{Name: "city", Description: "The name of the city"},
			{Name: "country", Description: "The country where the city is located"},
		})

		input := "```json\n{\"city\": \"Tokyo\", \"country\": \"Japan\"}\n```"
		result, err := parser.Parse(input)
		if err != nil {
			t.Fatal(err)
		}

		parsed, ok := result.(map[string]string)
		if !ok {
			t.Fatalf("expected map[string]string, got %T", result)
		}
		if parsed["city"] != "Tokyo" {
			t.Errorf("expected city 'Tokyo', got '%s'", parsed["city"])
		}
		if parsed["country"] != "Japan" {
			t.Errorf("expected country 'Japan', got '%s'", parsed["country"])
		}
	})

	t.Run("Defined", func(t *testing.T) {
		type Person struct {
			Name string `json:"name" describe:"person's name"`
			Age  int    `json:"age" describe:"person's age"`
		}

		parser, err := outputparser.NewDefined(Person{})
		if err != nil {
			t.Fatal(err)
		}

		input := "```json\n{\"name\": \"Alice\", \"age\": 30}\n```"
		result, err := parser.Parse(input)
		if err != nil {
			t.Fatal(err)
		}

		if result.Name != "Alice" {
			t.Errorf("expected name 'Alice', got '%s'", result.Name)
		}
		if result.Age != 30 {
			t.Errorf("expected age 30, got %d", result.Age)
		}
	})

	t.Run("RegexParser", func(t *testing.T) {
		parser := outputparser.NewRegexParser(`Temperature: (?P<temp>\d+)°C, Humidity: (?P<humidity>\d+)%`)

		input := "Temperature: 25°C, Humidity: 60%"
		result, err := parser.Parse(input)
		if err != nil {
			t.Fatal(err)
		}

		parsed, ok := result.(map[string]string)
		if !ok {
			t.Fatalf("expected map[string]string, got %T", result)
		}
		if parsed["temp"] != "25" {
			t.Errorf("expected temp '25', got '%s'", parsed["temp"])
		}
		if parsed["humidity"] != "60" {
			t.Errorf("expected humidity '60', got '%s'", parsed["humidity"])
		}
	})
}

// TestOutputParserWithChain demonstrates using output parsers in a chain pattern
// combining prompts, LLM calls, and parsing
func TestOutputParserWithChain(t *testing.T) {
	// Set up API connection
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))
	llm, err := openai.New(
		openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
		openai.WithModel("qwen-long-latest"),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Example: Recipe Generator with Structured Output
	t.Log("\n--- Recipe Generator with Structured Output ---")

	// Define the output schema
	type Recipe struct {
		Name        string   `json:"name" describe:"name of the dish"`
		Cuisine     string   `json:"cuisine" describe:"type of cuisine (e.g., Italian, Chinese)"`
		PrepTime    string   `json:"prep_time" describe:"preparation time in minutes"`
		Ingredients []string `json:"ingredients" describe:"list of ingredients"`
		Steps       []string `json:"steps" describe:"cooking steps"`
	}

	parser, err := outputparser.NewDefined(Recipe{})
	if err != nil {
		t.Fatal(err)
	}

	// Create a chat prompt template
	chatTemplate := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate(
			"You are a professional chef. Generate recipes in the exact JSON format requested.",
			nil,
		),
		prompts.NewHumanMessagePromptTemplate(
			"Create a simple {{.cuisine}} recipe. {{.format_instructions}}",
			[]string{"cuisine", "format_instructions"},
		),
	})

	messages, err := chatTemplate.FormatMessages(map[string]any{
		"cuisine":             "Italian",
		"format_instructions": parser.GetFormatInstructions(),
	})
	if err != nil {
		t.Fatal(err)
	}

	// Convert to MessageContent
	content := make([]llms.MessageContent, len(messages))
	for i, msg := range messages {
		content[i] = llms.TextParts(msg.GetType(), msg.GetContent())
	}

	// Call LLM
	resp, err := llm.GenerateContent(ctx, content)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("LLM Response:")
	t.Log(resp.Choices[0].Content)

	// Parse the response
	// Note: LLM might include extra text, so we need to extract the JSON block
	responseText := resp.Choices[0].Content
	if !strings.HasPrefix(responseText, "```json") {
		// Try to find the JSON block in the response
		if idx := strings.Index(responseText, "```json"); idx != -1 {
			endIdx := strings.LastIndex(responseText, "```")
			if endIdx > idx {
				responseText = responseText[idx : endIdx+3]
			}
		}
	}

	recipe, err := parser.Parse(responseText)
	if err != nil {
		t.Logf("Parse error (LLM response may not match expected format): %v", err)
	} else {
		t.Log("\nParsed Recipe:")
		t.Logf("  Name: %s", recipe.Name)
		t.Logf("  Cuisine: %s", recipe.Cuisine)
		t.Logf("  Prep Time: %s", recipe.PrepTime)
		t.Logf("  Ingredients: %v", recipe.Ingredients)
		t.Logf("  Steps: %v", recipe.Steps)
	}
}
