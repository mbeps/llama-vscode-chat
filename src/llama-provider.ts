
import * as vscode from "vscode";
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelChatRequestMessage,
    PrepareLanguageModelChatModelOptions,
    ProvideLanguageModelChatResponseOptions,
    LanguageModelResponsePart,
    Progress,
} from "vscode";
import { BaseChatModelProvider, DEFAULT_CONTEXT_LENGTH, DEFAULT_MAX_OUTPUT_TOKENS } from "./base-provider";
import { convertMessages, convertTools, validateRequest } from "./utils";
import { ExtendedLanguageModelChatInformation, LlamaCppServerProps } from "./types";

/**
 * Chat model provider for Llama.cpp servers.
 * Implements the VS Code language model chat provider interface for Llama.cpp compatible APIs.
 * Handles model discovery, chat responses, and streaming from local Llama.cpp instances.
 *
 */
export class LlamaCppChatModelProvider extends BaseChatModelProvider {
    /**
     * Cached context size from the server.
     */
    private _cachedContextSize: number | undefined;

    /**
     * Creates a new Llama.cpp chat model provider.
     * Initializes the provider with secret storage and user agent for API requests.
     *
     * @param secrets - VS Code secret storage for storing server URL and API key.
     * @param userAgent - User agent string to include in HTTP requests.
     */
    constructor(secrets: vscode.SecretStorage, private readonly userAgent: string) {
        super(secrets);
    }

    /**
     * Provides information about available Llama.cpp models.
     * Fetches model list from the configured server and returns model information.
     *
     * @param options - Options for the request, including error suppression.
     * @param token - Cancellation token to abort the operation.
     * @returns Promise resolving to an array of available models.
     */
    async provideLanguageModelChatInformation(
        options: PrepareLanguageModelChatModelOptions,
        _token: CancellationToken
    ): Promise<ExtendedLanguageModelChatInformation[]> {
        const serverUrl = await this.getServerUrl();
        const apiKey = await this.getApiKey(); // Optional

        try {
            // Parallelize model and property fetching
            const [n_ctx, models] = await Promise.all([
                this.fetchServerProps(serverUrl, apiKey),
                this.fetchModels(serverUrl, apiKey),
            ]);

            const contextSize = n_ctx ?? DEFAULT_CONTEXT_LENGTH;
            this._cachedContextSize = contextSize;

            const maxOutput = Math.min(DEFAULT_MAX_OUTPUT_TOKENS, Math.floor(contextSize / 2));
            const maxInput = Math.max(1, contextSize - maxOutput);

            return models.map(model => ({
                id: model.id,
                name: model.id, // Llama.cpp usually returns filename as ID
                tooltip: `Llama.cpp model: ${model.id}`,
                family: "llama-cpp",
                version: "1.0.0",
                maxInputTokens: maxInput,
                maxOutputTokens: maxOutput,
                capabilities: {
                    toolCalling: true, // Assuming modern models support it
                    imageInput: false, // Could be true for vision models, but safe default is false
                },
                isUserSelectable: true,
                metadata: {},
            }));
        } catch (err) {
            if (!options.silent) {
                console.error("[Llama.cpp Provider] Failed to fetch models", err);
            }
            return []; // Return empty if failed or server not running
        }
    }

    /**
     * Fetches server properties from the Llama.cpp `/props` endpoint.
     * Extracts the default context size (n_ctx) if available.
     *
     * @param serverUrl - The base URL of the Llama.cpp server.
     * @param apiKey - Optional API key for authentication.
     * @returns Promise resolving to the context size, or undefined if fetch fails.
     */
    private async fetchServerProps(serverUrl: string, apiKey?: string): Promise<number | undefined> {
        try {
            const headers: Record<string, string> = {
                "Content-Type": "application/json",
                "User-Agent": this.userAgent,
            };
            if (apiKey) {
                headers["Authorization"] = `Bearer ${apiKey}`;
            }

            const response = await fetch(`${serverUrl}/props`, {
                method: "GET",
                headers,
            });

            if (!response.ok) {
                return undefined;
            }

            const data = (await response.json()) as LlamaCppServerProps;
            const contextSize = data.default_generation_settings?.n_ctx;

            // Sanity check: ignore n_ctx values below 1000
            return (contextSize && contextSize >= 1000) ? contextSize : undefined;
        } catch (err) {
            console.warn("[Llama.cpp Provider] Failed to fetch server props", err);
            return undefined;
        }
    }

    /**
     * Provides a chat response from the Llama.cpp model.
     * Sends a chat completion request to the server and processes the streaming response.
     *
     * @param model - Information about the selected model.
     * @param messages - Array of chat messages for the conversation.
     * @param options - Options for the response generation.
     * @param progress - Progress callback to report response parts.
     * @param token - Cancellation token to abort the operation.
     * @returns Promise that resolves when the response is complete.
     */
    async provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatRequestMessage[],
        options: ProvideLanguageModelChatResponseOptions,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken
    ): Promise<void> {
        const serverUrl = await this.getServerUrl();
        const apiKey = await this.getApiKey();

        validateRequest(messages);
        const openaiMessages = convertMessages(messages);
        const toolConfig = convertTools(options);

        // Check token limits roughly
        const inputTokenCount = this.estimateMessagesTokens(messages);
        const toolTokenCount = this.estimateToolTokens(toolConfig.tools);
        const totalInput = inputTokenCount + toolTokenCount;
        const contextSize = this._cachedContextSize ?? DEFAULT_CONTEXT_LENGTH;
        const remainingContext = Math.max(1, contextSize - totalInput);
        const requestedMaxTokens = options.modelOptions?.max_tokens || 4096;

        const requestBody: Record<string, unknown> = {
            model: model.id,
            messages: openaiMessages,
            stream: true,
            max_tokens: Math.min(requestedMaxTokens, remainingContext),
            temperature: options.modelOptions?.temperature ?? 0.7,
        };

        if (toolConfig.tools) {
            requestBody.tools = toolConfig.tools;
        }
        if (toolConfig.tool_choice) {
            requestBody.tool_choice = toolConfig.tool_choice;
        }

        const headers: Record<string, string> = {
            "Content-Type": "application/json",
            "User-Agent": this.userAgent,
        };
        if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
        }

        const controller = new AbortController();
        const disposable = token.onCancellationRequested(() => controller.abort());

        try {
            const response = await fetch(`${serverUrl}/v1/chat/completions`, {
                method: "POST",
                headers,
                body: JSON.stringify(requestBody),
                signal: controller.signal,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Llama.cpp API error: ${response.status} ${response.statusText}\n${errorText}`);
            }

            if (!response.body) {
                throw new Error("No response body from Llama.cpp API");
            }

            await this.processStreamingResponse(response.body, progress, token);
        } catch (err) {
            if (token.isCancellationRequested) {
                return;
            }
            console.error("[Llama.cpp Provider] Chat request failed", err);
            throw err;
        } finally {
            disposable.dispose();
        }
    }

    /**
     * Retrieves the configured server URL from secrets.
     * Falls back to default localhost URL if not configured.
     *
     * @returns Promise resolving to the server URL.
     */
    private async getServerUrl(): Promise<string> {
        // Default to localhost:8080 if not configured
        return (await this.secrets.get("llamacpp.serverUrl")) || "http://localhost:8080";
    }

    /**
     * Retrieves the optional API key from secrets.
     * Returns undefined if no API key is configured.
     *
     * @returns Promise resolving to the API key or undefined.
     */
    private async getApiKey(): Promise<string | undefined> {
        return await this.secrets.get("llamacpp.apiKey");
    }

    /**
     * Fetches the list of available models from the Llama.cpp server.
     * Makes a GET request to the /v1/models endpoint.
     *
     * @param serverUrl - The base URL of the Llama.cpp server.
     * @param apiKey - Optional API key for authentication.
     * @returns Promise resolving to an array of model objects.
     */
    private async fetchModels(serverUrl: string, apiKey?: string): Promise<{ id: string }[]> {
        const headers: Record<string, string> = {
             "User-Agent": this.userAgent
        };
        if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
        }

        const response = await fetch(`${serverUrl}/v1/models`, {
            method: "GET",
            headers,
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
        }

        const data = (await response.json()) as { data: { id: string }[] };
        return data.data || [];
    }
}
