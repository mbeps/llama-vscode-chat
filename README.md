# Llama.cpp Provider for GitHub Copilot

This extension integrates Llama.cpp models into GitHub Copilot in VS Code.

> NOTE: My computer is not powerful enough to run decent models locally so this extension wasn't tested fully. Feel free to contribute if this is something that is useful.

## Features

- Integrates Llama.cpp server into VS Code's language model chat.
- Supports streaming responses.
- Handles tool calling for function invocations.
- Manages multiple models from the Llama.cpp server.

## Requirements

- VS Code version 1.104.0 or higher.
- A running Llama.cpp server with OpenAI-compatible API.

## Stack

- [**TypeScript**](https://www.typescriptlang.org/): A typed superset of JavaScript.
- [**VS Code API**](https://code.visualstudio.com/api): APIs for building extensions.

## Design

The extension uses a base provider class for OpenAI-compatible chat APIs. The Llama.cpp provider extends this base to connect to a local Llama.cpp server. It handles model fetching, message conversion, and streaming responses. Tool calling is supported through OpenAI-compatible formats.

## Setting Up Project Locally

1. Clone the repository.
```sh
git clone https://github.com/mbeps/llama-vscode-chat.git
```

2. Install dependencies.
```sh
npm install
```

3. Compile the extension.
```sh
npm run compile
```

4. Open in VS Code and run the extension.

## Configuration

1. Open the command palette.
2. Select "Llama.cpp: Set Server URL"
3. Configure the URL of your Llama.cpp server (e.g., `http://localhost:8000`).

## References

- [Llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [VS Code Extension API](https://code.visualstudio.com/api)