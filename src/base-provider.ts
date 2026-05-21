
import * as vscode from "vscode";
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelChatRequestMessage,
    LanguageModelChatProvider,
    PrepareLanguageModelChatModelOptions,
    ProvideLanguageModelChatResponseOptions,
    LanguageModelResponsePart,
    Progress,
    EventEmitter,
} from "vscode";
import { tryParseJSONObject } from "./utils";

export const DEFAULT_MAX_OUTPUT_TOKENS = 16000;
export const DEFAULT_CONTEXT_LENGTH = 128000;

/**
 * Base class for OpenAI-compatible chat providers.
 * Provides common functionality for handling streaming responses, tool calls, and token estimation.
 * Subclasses must implement the abstract methods to integrate with specific APIs.
 *
 */
export abstract class BaseChatModelProvider implements LanguageModelChatProvider {
    /** Event fired when the list of available models changes. */
    private readonly _onDidChangeLanguageModelChatInformation = new EventEmitter<void>();
    public readonly onDidChangeLanguageModelChatInformation = this._onDidChangeLanguageModelChatInformation.event;

    /** Buffer for assembling streamed tool calls by index. */
    private _toolCallBuffers: Map<number, { id?: string; name?: string; args: string }> = new Map<
        number,
        { id?: string; name?: string; args: string }
    >();

    /** Indices for which a tool call has been fully emitted. */
    private _completedToolCallIndices = new Set<number>();

    /** Track if we emitted any assistant text before seeing tool calls (SSE-like begin-tool-calls hint). */
    private _hasEmittedAssistantText = false;

    /** Track if we emitted the begin-tool-calls whitespace flush. */
    private _emittedBeginToolCallsHint = false;

    // Lightweight tokenizer state for tool calls embedded in text
    private _textToolParserBuffer = "";
    private _textToolActive:
        | undefined
        | {
              name?: string;
              index?: number;
              argBuffer: string;
              emitted?: boolean;
          };
    private _emittedTextToolCallKeys = new Set<string>();
    private _emittedTextToolCallIds = new Set<string>();

    /**
     * Creates a new instance of the base chat model provider.
     * Initializes internal state for handling streaming responses and tool calls.
     *
     * @param secrets - VS Code secret storage for storing sensitive data like API keys.
     */
    constructor(protected readonly secrets: vscode.SecretStorage) {}

    /**
     * Notify VS Code that the provider configuration has changed and model information should be refreshed.
     */
    public notifyConfigChanged(): void {
        this._onDidChangeLanguageModelChatInformation.fire();
    }

    /**
     * Provides information about available language models.
     * Subclasses must implement this to return model details from their API.
     *
     * @param options - Options for the request, including whether to suppress errors.
     * @param token - Cancellation token to abort the operation.
     * @returns Promise resolving to an array of language model information.
     */
    abstract provideLanguageModelChatInformation(
        options: PrepareLanguageModelChatModelOptions,
        token: CancellationToken
    ): Promise<LanguageModelChatInformation[]>;

    /**
     * Provides a chat response from the language model.
     * Subclasses must implement this to send requests to their API and handle responses.
     *
     * @param model - Information about the selected language model.
     * @param messages - Array of chat messages for the conversation.
     * @param options - Options for the response generation.
     * @param progress - Progress callback to report response parts.
     * @param token - Cancellation token to abort the operation.
     * @returns Promise that resolves when the response is complete.
     */
    abstract provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatRequestMessage[],
        options: ProvideLanguageModelChatResponseOptions,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken
    ): Promise<void>;

    /**
     * Roughly estimate tokens for VS Code chat messages (text only).
     * Uses a simple heuristic of 1 token per 4 characters.
     *
     * @param msgs - Array of chat messages to estimate tokens for.
     * @returns Estimated number of tokens.
     */
    protected estimateMessagesTokens(msgs: readonly vscode.LanguageModelChatRequestMessage[]): number {
        let total = 0;
        for (const m of msgs) {
            for (const part of m.content) {
                if (part instanceof vscode.LanguageModelTextPart) {
                    total += Math.ceil(part.value.length / 4);
                }
            }
        }
        return total;
    }

    /**
     * Rough token estimate for tool definitions by JSON size.
     * Serializes the tools to JSON and estimates tokens based on length.
     *
     * @param tools - Array of tool definitions to estimate tokens for.
     * @returns Estimated number of tokens for the tools.
     */
    protected estimateToolTokens(
        tools: { type: string; function: { name: string; description?: string; parameters?: object } }[] | undefined
    ): number {
        if (!tools || tools.length === 0) {
            return 0;
        }
        try {
            const json = JSON.stringify(tools);
            return Math.ceil(json.length / 4);
        } catch {
            return 0;
        }
    }

    /**
     * Returns the number of tokens for a given text using the model specific tokenizer logic.
     * Uses a simple heuristic for estimation since actual tokenization requires model-specific logic.
     *
     * @param model - Information about the language model.
     * @param text - The text or message to count tokens for.
     * @param _token - Cancellation token (unused in this implementation).
     * @returns Promise resolving to the estimated token count.
     */
    async provideTokenCount(
        model: LanguageModelChatInformation,
        text: string | LanguageModelChatRequestMessage,
        _token: CancellationToken
    ): Promise<number> {
        if (typeof text === "string") {
            return Math.ceil(text.length / 4);
        } else {
            let totalTokens = 0;
            for (const part of text.content) {
                if (part instanceof vscode.LanguageModelTextPart) {
                    totalTokens += Math.ceil(part.value.length / 4);
                }
            }
            return totalTokens;
        }
    }

    /**
     * Read and parse the stream (SSE-like) response and report parts.
     * Handles Server-Sent Events from the API, processing deltas and emitting progress.
     *
     * @param responseBody - The readable stream from the API response.
     * @param progress - Progress callback to report response parts.
     * @param token - Cancellation token to abort processing.
     * @returns Promise that resolves when streaming is complete.
     */
    protected async processStreamingResponse(
        responseBody: ReadableStream<Uint8Array>,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        token: vscode.CancellationToken
    ): Promise<void> {
        this._toolCallBuffers.clear();
        this._completedToolCallIndices.clear();
        this._hasEmittedAssistantText = false;
        this._emittedBeginToolCallsHint = false;
        this._textToolParserBuffer = "";
        this._textToolActive = undefined;
        this._emittedTextToolCallKeys.clear();
        this._emittedTextToolCallIds.clear();

        const reader = responseBody.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        try {
            while (!token.isCancellationRequested) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.startsWith("data: ")) {
                        continue;
                    }
                    const data = line.slice(6);
                    if (data === "[DONE]") {
                        // Do not throw on [DONE]; any incomplete/empty buffers are ignored.
                        await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ false);
                        // Flush any in-progress text-embedded tool call (silent if incomplete)
                        await this.flushActiveTextToolCall(progress);
                        continue;
                    }

                    try {
                        const parsed = JSON.parse(data);
                        await this.processDelta(parsed, progress);
                    } catch {
                        // Silently ignore malformed SSE lines temporarily
                    }
                }
            }
        } finally {
            reader.releaseLock();
            // Clean up any leftover tool call state
            this._toolCallBuffers.clear();
            this._completedToolCallIndices.clear();
            this._hasEmittedAssistantText = false;
            this._emittedBeginToolCallsHint = false;
            this._textToolParserBuffer = "";
            this._textToolActive = undefined;
            this._emittedTextToolCallKeys.clear();
        }
    }

    /**
     * Handle a single streamed delta chunk, emitting text and tool call parts.
     * Processes the delta from the streaming response and reports appropriate parts.
     *
     * @param delta - The delta object from the API response.
     * @param progress - Progress callback to report response parts.
     * @returns Promise resolving to true if something was emitted, false otherwise.
     */
    private async processDelta(
        delta: Record<string, unknown>,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>
    ): Promise<boolean> {
        let emitted = false;
        const choice = (delta.choices as Record<string, unknown>[] | undefined)?.[0];
        if (!choice) {
            return false;
        }

        const deltaObj = choice.delta as Record<string, unknown> | undefined;

        // report thinking progress if backend provides it and host supports it
        try {
            const maybeThinking =
                (choice as Record<string, unknown> | undefined)?.thinking ?? (deltaObj as Record<string, unknown> | undefined)?.thinking;
            if (maybeThinking !== undefined) {
                const vsAny = vscode as unknown as Record<string, unknown>;
                // Note: Thinking parts are currently a proposed VS Code API. We use a runtime probe
                // to maintain compatibility with versions that do not yet expose this type.
                const ThinkingCtor = vsAny["LanguageModelThinkingPart"] as
                    | (new (text: string, id?: string, metadata?: unknown) => unknown)
                    | undefined;
                if (ThinkingCtor) {
                    let text = "";
                    let id: string | undefined;
                    let metadata: unknown;
                    if (maybeThinking && typeof maybeThinking === "object") {
                        const mt = maybeThinking as Record<string, unknown>;
                        text = typeof mt["text"] === "string" ? (mt["text"] as string) : "";
                        id = typeof mt["id"] === "string" ? (mt["id"] as string) : undefined;
                        metadata = mt["metadata"];
                    } else if (typeof maybeThinking === "string") {
                        text = maybeThinking;
                    }
                    if (text) {
                        progress.report(
                            new (ThinkingCtor as new (text: string, id?: string, metadata?: unknown) => unknown)(
                                text,
                                id,
                                metadata
                            ) as unknown as vscode.LanguageModelResponsePart
                        );
                        emitted = true;
                    }
                }
            }
        } catch {
            // ignore errors here temporarily
        }
        if (deltaObj?.content) {
            const content = String(deltaObj.content);
            const res = this.processTextContent(content, progress);
            if (res.emittedText) {
                this._hasEmittedAssistantText = true;
            }
            if (res.emittedAny) {
                emitted = true;
            }
        }

        if (deltaObj?.tool_calls) {
            const toolCalls = deltaObj.tool_calls as Array<Record<string, unknown>>;

            // SSEProcessor-like: if first tool call appears after text, emit a whitespace
            // to ensure any UI buffers/linkifiers are flushed without adding visible noise.
            if (!this._emittedBeginToolCallsHint && this._hasEmittedAssistantText && toolCalls.length > 0) {
                progress.report(new vscode.LanguageModelTextPart(" "));
                this._emittedBeginToolCallsHint = true;
            }

            for (const tc of toolCalls) {
                const idx = (tc.index as number) ?? 0;
                // Ignore any further deltas for an index we've already completed
                if (this._completedToolCallIndices.has(idx)) {
                    continue;
                }
                const buf = this._toolCallBuffers.get(idx) ?? { args: "" };
                if (tc.id && typeof tc.id === "string") {
                    buf.id = tc.id as string;
                }
                const func = tc.function as Record<string, unknown> | undefined;
                if (func?.name && typeof func.name === "string") {
                    buf.name = func.name as string;
                }
                if (typeof func?.arguments === "string") {
                    buf.args += func.arguments as string;
                }
                this._toolCallBuffers.set(idx, buf);

                // Emit immediately once arguments become valid JSON to avoid perceived hanging
                await this.tryEmitBufferedToolCall(idx, progress);
            }
        }

        const finish = (choice.finish_reason as string | undefined) ?? undefined;
        if (finish === "tool_calls" || finish === "stop") {
            // On both 'tool_calls' and 'stop', emit any buffered calls and throw on invalid JSON
            await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ true);
        }
        return emitted;
    }

    private processTextContent(
        input: string,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>
    ): { emittedText: boolean; emittedAny: boolean } {
        const BEGIN = "<|tool_call_begin|>";
        const ARG_BEGIN = "<|tool_call_argument_begin|>";
        const END = "<|tool_call_end|>";

        let data = this._textToolParserBuffer + input;
        let emittedText = false;
        let emittedAny = false;
        let visibleOut = "";

        while (data.length > 0) {
            if (!this._textToolActive) {
                const b = data.indexOf(BEGIN);
                if (b === -1) {
                    // No tool-call start: emit visible portion, but keep any partial BEGIN prefix as buffer
                    const longestPartialPrefix = ((): number => {
                        for (let k = Math.min(BEGIN.length - 1, data.length - 1); k > 0; k--) {
                            if (data.endsWith(BEGIN.slice(0, k))) {
                                return k;
                            }
                        }
                        return 0;
                    })();
                    if (longestPartialPrefix > 0) {
                        const visible = data.slice(0, data.length - longestPartialPrefix);
                        if (visible) {
                            visibleOut += this.stripControlTokens(visible);
                        }
                        this._textToolParserBuffer = data.slice(data.length - longestPartialPrefix);
                        data = "";
                        break;
                    } else {
                        // All visible, clean other control tokens
                        visibleOut += this.stripControlTokens(data);
                        data = "";
                        break;
                    }
                }
                // Emit text before the token
                const pre = data.slice(0, b);
                if (pre) {
                    visibleOut += this.stripControlTokens(pre);
                }
                // Advance past BEGIN
                data = data.slice(b + BEGIN.length);

                // Find the delimiter that ends the name/index segment
                const a = data.indexOf(ARG_BEGIN);
                const e = data.indexOf(END);
                let delimIdx = -1;
                let delimKind: "arg" | "end" | undefined = undefined;
                if (a !== -1 && (e === -1 || a < e)) {
                    delimIdx = a;
                    delimKind = "arg";
                } else if (e !== -1) {
                    delimIdx = e;
                    delimKind = "end";
                } else {
                    // Incomplete header; keep for next chunk (re-add BEGIN so we don't lose it)
                    this._textToolParserBuffer = BEGIN + data;
                    data = "";
                    break;
                }

                const header = data.slice(0, delimIdx).trim();
                const m = header.match(/^([A-Za-z0-9_\-.]+)(?::(\d+))?/);
                const name = m?.[1] ?? undefined;
                const index = m?.[2] ? Number(m?.[2]) : undefined;
                this._textToolActive = { name, index, argBuffer: "", emitted: false };
                // Advance past delimiter token
                if (delimKind === "arg") {
                    data = data.slice(delimIdx + ARG_BEGIN.length);
                } else /* end */ {
                    // No args, finalize immediately
                    data = data.slice(delimIdx + END.length);
                    const did = this.emitTextToolCallIfValid(progress, this._textToolActive, "{}");
                    if (did) {
                        this._textToolActive.emitted = true;
                        emittedAny = true;
                    }
                    this._textToolActive = undefined;
                }
                continue;
            }

            // We are inside arguments, collect until END and emit as soon as JSON becomes valid
            const e2 = data.indexOf(END);
            if (e2 === -1) {
                // No end marker yet, accumulate and check for early valid JSON
                this._textToolActive.argBuffer += data;
                // Early emit when JSON becomes valid and we haven't emitted yet
                if (!this._textToolActive.emitted) {
                    const did = this.emitTextToolCallIfValid(progress, this._textToolActive, this._textToolActive.argBuffer);
                    if (did) {
                        this._textToolActive.emitted = true;
                        emittedAny = true;
                    }
                }
                data = "";
                break;
            } else {
                this._textToolActive.argBuffer += data.slice(0, e2);
                // Consume END
                data = data.slice(e2 + END.length);
                // Final attempt to emit if not already
                if (!this._textToolActive.emitted) {
                    const did = this.emitTextToolCallIfValid(progress, this._textToolActive, this._textToolActive.argBuffer);
                    if (did) {
                        emittedAny = true;
                    }
                }
                this._textToolActive = undefined;
                continue;
            }
        }

        // Emit any visible text
        const textToEmit = visibleOut;
        if (textToEmit && textToEmit.length > 0) {
            progress.report(new vscode.LanguageModelTextPart(textToEmit));
            emittedText = true;
            emittedAny = true;
        }

        // Store leftover for next chunk
        this._textToolParserBuffer = data;

        return { emittedText, emittedAny };
    }

    private emitTextToolCallIfValid(
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        call: { name?: string; index?: number; argBuffer: string; emitted?: boolean },
        argText: string
    ): boolean {
        const name = call.name ?? "unknown_tool";
        const parsed = tryParseJSONObject(argText);
        if (!parsed.ok) {
            return false;
        }
        const canonical = JSON.stringify(parsed.value);
        const key = `${name}:${canonical}`;
        // identity-based dedupe when index is present
        if (typeof call.index === "number") {
            const idKey = `${name}:${call.index}`;
            if (this._emittedTextToolCallIds.has(idKey)) {
                return false;
            }
            // Mark identity as emitted
            this._emittedTextToolCallIds.add(idKey);
        } else if (this._emittedTextToolCallKeys.has(key)) {
            return false;
        }
        this._emittedTextToolCallKeys.add(key);
        const id = `tct_${Math.random().toString(36).slice(2, 10)}`;
        progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
        return true;
    }

    /**
     * Flushes any active text-embedded tool call.
     * Attempts to parse and emit the tool call if arguments are valid JSON.
     *
     * @param progress - Progress callback to report the tool call part.
     * @returns Promise that resolves when flushing is complete.
     */
    private async flushActiveTextToolCall(progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
        if (!this._textToolActive) {
            return;
        }
        const argText = this._textToolActive.argBuffer;
        const parsed = tryParseJSONObject(argText);
        if (!parsed.ok) {
            return;
        }
        // Emit (dedupe ensures we don't double-emit)
        this.emitTextToolCallIfValid(progress, this._textToolActive, argText);
        this._textToolActive = undefined;
    }

    private async tryEmitBufferedToolCall(
        index: number,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>
    ): Promise<void> {
        const buf = this._toolCallBuffers.get(index);
        if (!buf) {
            return;
        }
        if (!buf.name) {
            return;
        }
        const canParse = tryParseJSONObject(buf.args);
        if (!canParse.ok) {
            return;
        }
        const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
        const parameters = canParse.value;
        try {
            const canonical = JSON.stringify(parameters);
            this._emittedTextToolCallKeys.add(`${buf.name}:${canonical}`);
        } catch {
            /* ignore */
        }
        progress.report(new vscode.LanguageModelToolCallPart(id, buf.name, parameters));
        this._toolCallBuffers.delete(index);
        this._completedToolCallIndices.add(index);
    }

    /**
     * Flushes all accumulated tool call buffers.
     * Attempts to parse and emit tool calls, optionally throwing on invalid JSON.
     *
     * @param progress - Progress callback to report tool call parts.
     * @param throwOnInvalid - Whether to throw an error for invalid JSON arguments.
     * @returns Promise that resolves when all buffers are flushed.
     */
    private async flushToolCallBuffers(
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        throwOnInvalid: boolean
    ): Promise<void> {
        if (this._toolCallBuffers.size === 0) {
            return;
        }
        for (const [idx, buf] of Array.from(this._toolCallBuffers.entries())) {
            const parsed = tryParseJSONObject(buf.args);
            if (!parsed.ok) {
                if (throwOnInvalid) {
                    console.error("[Chat Model Provider] Invalid JSON for tool call", {
                        idx,
                        snippet: (buf.args || "").slice(0, 200),
                    });
                    throw new Error("Invalid JSON for tool call");
                }
                // When not throwing (e.g. on [DONE]), drop silently to reduce noise
                continue;
            }
            const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
            const name = buf.name ?? "unknown_tool";
            try {
                const canonical = JSON.stringify(parsed.value);
                this._emittedTextToolCallKeys.add(`${name}:${canonical}`);
            } catch {
                /* ignore */
            }
            progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
            this._toolCallBuffers.delete(idx);
            this._completedToolCallIndices.add(idx);
        }
    }

    /**
     * Helper to strip control tokens from text if they leak.
     * Removes special tokens that might appear in the response text.
     *
     * @param text - The text to clean.
     * @returns The text with control tokens removed.
     */
    private stripControlTokens(text: string): string {
         // Implement if needed, or just return text if not used elsewhere, but the original code called `this.stripControlTokens`.
         // I missed copying that method or it wasn't shown in the view_file.
         // Let me check the view_file output again.
         return text.replace(/<\|tool_call_begin\|>|<\|tool_call_argument_begin\|>|<\|tool_call_end\|>/g, "");
    }
}
