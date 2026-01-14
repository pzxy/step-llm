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

func TestArgLoad(t *testing.T) {
	ctx := context.Background()
	dataDir := filepath.Join(projectRoot(t), "data", "arg_load")

	// ------------------------------------------------------------
	// 1) Load from an io.Reader (e.g. a file, bytes.Buffer, network stream)
	// ------------------------------------------------------------
	filePath := filepath.Join(dataDir, "16_a.txt")
	f, err := os.Open(filePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	textLoader := documentloaders.NewText(f)
	docs, err := textLoader.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if !strings.Contains(docs[0].PageContent, "hello langchaingo loaders") {
		t.Fatalf("unexpected content: %q", docs[0].PageContent[:min(60, len(docs[0].PageContent))])
	}
	t.Logf("Text loader doc chars=%d", len(docs[0].PageContent))

	// Split the loaded document into chunks (useful before embeddings / retrieval).
	// IMPORTANT: Load() consumes the io.Reader, so we must re-open for LoadAndSplit().
	f2, err := os.Open(filePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	textLoader2 := documentloaders.NewText(f2)
	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(200),
		textsplitter.WithChunkOverlap(50),
	)
	chunks, err := textLoader2.LoadAndSplit(ctx, splitter)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	t.Logf("Text loader chunks=%d (first chunk chars=%d)", len(chunks), len(chunks[0].PageContent))

	// ------------------------------------------------------------
	// 2) Load a directory recursively (auto-chooses loader by extension)
	// ------------------------------------------------------------
	dirLoader := documentloaders.NewRecursiveDirLoader(
		documentloaders.WithRoot(dataDir),
		documentloaders.WithMaxDepth(2),
		documentloaders.WithAllowExts("txt", "md"),
	)
	dirDocs, err := dirLoader.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(dirDocs) != 2 {
		t.Fatalf("expected 2 docs (txt+md), got %d", len(dirDocs))
	}
	for _, d := range dirDocs {
		src, _ := d.Metadata["source"].(string)
		if src == "" {
			t.Fatalf("expected source metadata, got: %#v", d.Metadata)
		}
		t.Logf("Dir doc source=%s chars=%d", filepath.Base(src), len(d.PageContent))
	}

}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func projectRoot(t *testing.T) string {
	t.Helper()
	// Resolve repo root based on this test file location:
	// .../step-llm/langchain/16_arg_load_test.go -> .../step-llm
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Dir(filepath.Dir(thisFile))
}
