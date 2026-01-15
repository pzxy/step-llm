package langchain

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

func TestRagSplitter(t *testing.T) {
	// This test demonstrates "document splitting" for RAG:
	// - Split long documents into chunks for embedding & retrieval
	// - Preserve metadata (especially provenance like metadata["source"])

	root := ragSplitFixtureRoot(t)
	txtPath := filepath.Join(root, "16_a.txt")
	mdPath := filepath.Join(root, "16_b.md")

	txt := mustReadFile(t, txtPath)
	md := mustReadFile(t, mdPath)
	mdHeading := firstMarkdownHeadingLine(string(md))
	if mdHeading == "" {
		t.Fatalf("expected markdown fixture to contain a heading line starting with '#': %s", mdPath)
	}

	// ------------------------------------------------------------
	// 1) Plain text splitter (RecursiveCharacter)
	// ------------------------------------------------------------
	txtDocs := []schema.Document{{
		PageContent: string(txt),
		Metadata: map[string]any{
			"source": txtPath,
		},
	}}
	txtSplitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(120),
		textsplitter.WithChunkOverlap(0),
	)

	txtChunks, err := textsplitter.SplitDocuments(txtSplitter, txtDocs)
	if err != nil {
		t.Fatal(err)
	}
	if len(txtChunks) <= 1 {
		t.Fatalf("expected text splitter to create multiple chunks, got %d", len(txtChunks))
	}
	for _, c := range txtChunks {
		if strings.TrimSpace(c.PageContent) == "" {
			t.Fatalf("unexpected empty txt chunk: %+v", c)
		}
		if got, _ := c.Metadata["source"].(string); got != txtPath {
			t.Fatalf("expected metadata[source]=%q, got %q", txtPath, got)
		}
		if textsplitter.DefaultOptions().LenFunc(c.PageContent) > 120 {
			t.Fatalf("expected chunk length <= 120, got %d", textsplitter.DefaultOptions().LenFunc(c.PageContent))
		}
	}

	// ------------------------------------------------------------
	// 2) Markdown-aware splitter (MarkdownTextSplitter)
	// ------------------------------------------------------------
	mdDocs := []schema.Document{{
		PageContent: string(md),
		Metadata: map[string]any{
			"source": mdPath,
		},
	}}
	mdSplitter := textsplitter.NewMarkdownTextSplitter(
		textsplitter.WithChunkSize(40),
		textsplitter.WithChunkOverlap(0),
		textsplitter.WithHeadingHierarchy(true),
	)

	mdChunks, err := textsplitter.SplitDocuments(mdSplitter, mdDocs)
	if err != nil {
		t.Fatal(err)
	}
	if len(mdChunks) <= 1 {
		t.Fatalf("expected markdown splitter to create multiple chunks, got %d", len(mdChunks))
	}

	// With HeadingHierarchy enabled, chunks should generally include the current header context.
	seenHeading := false
	for _, c := range mdChunks {
		if strings.TrimSpace(c.PageContent) == "" {
			t.Fatalf("unexpected empty md chunk: %+v", c)
		}
		if got, _ := c.Metadata["source"].(string); got != mdPath {
			t.Fatalf("expected metadata[source]=%q, got %q", mdPath, got)
		}
		trimmed := strings.TrimSpace(c.PageContent)
		// We expect the chunk to be prefixed with the (possibly hierarchical) heading context.
		// In our fixture there is only one heading, so it should match the first heading line.
		if strings.HasPrefix(trimmed, mdHeading) {
			seenHeading = true
		}
	}
	if !seenHeading {
		for i, c := range mdChunks {
			if i >= 5 {
				break
			}
			t.Logf("mdChunks[%d]=%q", i, strings.TrimSpace(c.PageContent))
		}
		t.Fatalf("expected at least one chunk to contain markdown heading context, but none did")
	}
}

func ragSplitFixtureRoot(t *testing.T) string {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to locate test file path via runtime.Caller")
	}
	// .../step-llm/langchain -> .../step-llm/data/arg_load
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "data", "arg_load"))
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file %s: %v", path, err)
	}
	return b
}

func firstMarkdownHeadingLine(markdown string) string {
	for _, line := range strings.Split(markdown, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			return trimmed
		}
	}
	return ""
}
