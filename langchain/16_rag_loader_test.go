package langchain

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/textsplitter"
)

func TestRagLoader(t *testing.T) {
	// This test demonstrates a common RAG "loader" step:
	// - Load local documents from disk
	// - Filter by extension
	// - Preserve provenance via metadata["source"]
	// - Optionally split long documents into chunks for embedding / retrieval

	ctx := context.Background()

	root := ragFixtureRoot(t)
	if _, err := os.Stat(root); err != nil {
		t.Fatalf("fixture directory does not exist: %s: %v", root, err)
	}

	loader := documentloaders.NewRecursiveDirLoader(
		documentloaders.WithRoot(root),
		documentloaders.WithMaxDepth(2),
		documentloaders.WithAllowExts("txt", "md"), // ignore 16_ignored.bin
	)

	// ------------------------------------------------------------
	// 1) Load documents (one doc per file for Text/Markdown files)
	// ------------------------------------------------------------
	docs, err := loader.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 docs (16_a.txt + 16_b.md), got %d", len(docs))
	}

	seen := map[string]bool{}
	for _, d := range docs {
		if strings.TrimSpace(d.PageContent) == "" {
			t.Fatalf("unexpected empty PageContent: %+v", d)
		}
		src, ok := d.Metadata["source"].(string)
		if !ok || src == "" {
			t.Fatalf("expected Metadata[\"source\"] to be set, got: %+v", d.Metadata)
		}
		if strings.HasSuffix(strings.ToLower(src), ".bin") {
			t.Fatalf("expected .bin to be filtered out, got source=%q", src)
		}
		seen[filepath.Base(src)] = true
	}
	if !seen["16_a.txt"] || !seen["16_b.md"] {
		t.Fatalf("expected sources {16_a.txt, 16_b.md}, got: %+v", seen)
	}

	// ------------------------------------------------------------
	// 2) LoadAndSplit: turn long files into smaller chunks (RAG-ready)
	// ------------------------------------------------------------
	// Keep chunk size small so our fixture definitely splits.
	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(120),
		textsplitter.WithChunkOverlap(0),
	)

	chunks, err := loader.LoadAndSplit(ctx, splitter)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) <= len(docs) {
		t.Fatalf("expected more chunks than original docs after splitting; docs=%d chunks=%d", len(docs), len(chunks))
	}
	for _, c := range chunks {
		if strings.TrimSpace(c.PageContent) == "" {
			t.Fatalf("unexpected empty chunk PageContent: %+v", c)
		}
		src, ok := c.Metadata["source"].(string)
		if !ok || src == "" {
			t.Fatalf("expected chunk Metadata[\"source\"] to be set, got: %+v", c.Metadata)
		}
	}
}

func ragFixtureRoot(t *testing.T) string {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to locate test file path via runtime.Caller")
	}
	// .../step-llm/langchain -> .../step-llm/data/arg_load
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "data", "arg_load"))
}
