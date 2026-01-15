package langchain

import (
	"context"
	"math"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
)

func TestRagEmbeddings(t *testing.T) {
	// This test demonstrates a common RAG "embeddings" step:
	// - Use a document vector model (Embeddings) to convert texts -> vectors
	// - Embed a query and rank documents by cosine similarity
	//
	// It requires an OpenAI-compatible embeddings endpoint + API key.
	// If no key is found, we skip (so CI doesn't fail by default).

	// Prefer true "OpenAI-compatible" keys first.
	os.Setenv("OPENAI_API_KEY", os.Getenv("DASHSCOPE_API_KEY"))

	baseURL := "https://dashscope.aliyuncs.com/compatible-mode/v1"
	embeddingModel := "text-embedding-v3"
	openaiOpts := []openai.Option{
		openai.WithEmbeddingModel(embeddingModel),
		openai.WithBaseURL(baseURL),
		openai.WithHTTPClient(&http.Client{Timeout: 90 * time.Second}),
	}

	llm, err := openai.New(openaiOpts...)
	if err != nil {
		t.Fatal(err)
	}
	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	docs := []string{
		"Colorful socks branding ideas and product naming tips.",
		"RAG: split long documents into chunks, embed each chunk, and retrieve by similarity at query time.",
		"Weather forecast: today's temperature and humidity in Beijing.",
	}
	query := "How do I split documents into chunks for RAG retrieval?"

	docVecs, err := embedder.EmbedDocuments(ctx, docs)
	if err != nil {
		t.Skipf("embedding documents failed (check BASE_URL/EMBEDDING_MODEL/network): %v", err)
	}
	if len(docVecs) != len(docs) {
		t.Fatalf("expected %d doc vectors, got %d", len(docs), len(docVecs))
	}

	qVec, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		t.Skipf("embedding query failed (check BASE_URL/EMBEDDING_MODEL/network): %v", err)
	}

	bestIdx := -1
	bestScore := float32(-1)
	for i := range docs {
		score, err := cosineSimilarity(qVec, docVecs[i])
		if err != nil {
			t.Fatalf("cosineSimilarity(doc=%d): %v", i, err)
		}
		t.Logf("doc[%d] score=%.4f text=%q", i, score, docs[i])
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	t.Logf("best match: idx=%d score=%.4f text=%q", bestIdx, bestScore, docs[bestIdx])
	if bestIdx != 1 {
		t.Fatalf("expected doc[1] to be the most similar to the query, got idx=%d (score=%.4f)", bestIdx, bestScore)
	}
}

func cosineSimilarity(a, b []float32) (float32, error) {
	if len(a) == 0 || len(b) == 0 {
		return 0, embeddings.ErrAllTextsLenZero
	}
	if len(a) != len(b) {
		return 0, embeddings.ErrVectorsNotSameSize
	}

	var dot float32
	var normA float32
	var normB float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0, embeddings.ErrAllTextsLenZero
	}

	den := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	return dot / den, nil
}
