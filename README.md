# step-llm demo

This is a small demo showing how to provide an API key and server address to the program.

Usage options (either set environment variables or pass flags):

- Environment variables:

  - `API_KEY` – your API key
  - `SERVER_ADDR` – your server address (include host and optional path; scheme optional)

- Flags:

  - `-api-key` – API key (overrides `API_KEY`)
  - `-server` – server address (overrides `SERVER_ADDR`)

Example using environment variables:

```bash
export API_KEY="sk-..."
export SERVER_ADDR="api.example.com/health"
go run main.go
```

Example using flags:

```bash
go run main.go -api-key "sk-..." -server "https://api.example.com/health"
```

Notes:

- The demo sends a GET request to the provided server address with header `Authorization: Bearer <API_KEY>`.
- If the provided server address doesn't include a scheme, `http://` is prefixed automatically.
# step-llm
learning for llm
