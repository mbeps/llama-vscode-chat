import * as vscode from "vscode";
import type { OpenAIChatMessage, OpenAIChatRole, OpenAIFunctionToolDef, OpenAIToolCall } from "./types";

// Tool calling sanitization helpers

/**
 * Checks if a property name is likely to represent an integer value.
 * Uses heuristics based on common integer-related keywords.
 *
 * @param propertyName - The property name to check.
 * @returns True if the property name suggests an integer, false otherwise.
 */
function isIntegerLikePropertyName(propertyName: string | undefined): boolean {
    if (!propertyName){
		return false;
	}
    const lowered = propertyName.toLowerCase();
    const integerMarkers = [
        "id",
        "limit",
        "count",
        "index",
        "size",
        "offset",
        "length",
        "results_limit",
        "maxresults",
        "debugsessionid",
        "cellid",
    ];
    return integerMarkers.some((m) => lowered.includes(m)) || lowered.endsWith("_id");
}

/**
 * Sanitizes a function name to make it safe for use.
 * Replaces invalid characters and ensures it starts with a letter.
 *
 * @param name - The original function name.
 * @returns The sanitized function name.
 */
function sanitizeFunctionName(name: unknown): string {
    if (typeof name !== "string" || !name){
		return "tool";
	}
    let sanitized = name.replace(/[^a-zA-Z0-9_-]/g, "_");
    if (!/^[a-zA-Z]/.test(sanitized)) {
        sanitized = `tool_${sanitized}`;
    }
    sanitized = sanitized.replace(/_+/g, "_");
    return sanitized.slice(0, 64);
}

/**
 * Prunes unknown or unsupported keywords from a JSON schema.
 * Keeps only allowed schema properties for compatibility.
 *
 * @param schema - The schema object to prune.
 * @returns The pruned schema object.
 */
function pruneUnknownSchemaKeywords(schema: unknown): Record<string, unknown> {
    if (!schema || typeof schema !== "object" || Array.isArray(schema)){
		return {};
	}
    const allow = new Set([
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "enum",
        "default",
        "items",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "pattern",
        "format",
    ]);
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(schema as Record<string, unknown>)) {
        if (allow.has(k)){
			out[k] = v as unknown;
		}
    }
    return out;
}

/**
 * Sanitizes a JSON schema by pruning unknown keywords and processing properties.
 * Recursively cleans the schema for safe use in tool definitions.
 *
 * @param input - The schema to sanitize.
 * @param propName - Optional property name for context.
 * @returns The sanitized schema.
 */
function sanitizeSchema(input: unknown, propName?: string): Record<string, unknown> {
    if (!input || typeof input !== "object" || Array.isArray(input)) {
        return { type: "object", properties: {} } as Record<string, unknown>;
    }

    let schema = input as Record<string, unknown>;

    for (const composite of ["anyOf", "oneOf", "allOf"]) {
        const branch = (schema as Record<string, unknown>)[composite] as unknown;
        if (Array.isArray(branch) && branch.length > 0) {
            let preferred: Record<string, unknown> | undefined;
            for (const b of branch) {
                if (b && typeof b === "object" && (b as Record<string, unknown>).type === "string") {
                    preferred = b as Record<string, unknown>;
                    break;
                }
            }
            schema = { ...(preferred ?? (branch[0] as Record<string, unknown>)) };
            break;
        }
    }

    schema = pruneUnknownSchemaKeywords(schema);

    let t = schema.type as string | undefined;
    if (t == null) {
        t = "object";
        schema.type = t;
    }

    if (t === "number" && propName && isIntegerLikePropertyName(propName)) {
        schema.type = "integer";
        t = "integer";
    }

    if (t === "object") {
        const props = (schema.properties as Record<string, unknown> | undefined) ?? {};
        const newProps: Record<string, unknown> = {};
        if (props && typeof props === "object") {
            for (const [k, v] of Object.entries(props)) {
                newProps[k] = sanitizeSchema(v, k);
            }
        }
        schema.properties = newProps;

        const req = schema.required as unknown;
        if (Array.isArray(req)) {
            schema.required = req.filter((r) => typeof r === "string");
        } else if (req !== undefined) {
            schema.required = [];
        }

        const ap = schema.additionalProperties as unknown;
        if (ap !== undefined && typeof ap !== "boolean") {
            delete schema.additionalProperties;
        }
    } else if (t === "array") {
        const items = schema.items as unknown;
        if (Array.isArray(items) && items.length > 0) {
            schema.items = sanitizeSchema(items[0]);
        } else if (items && typeof items === "object") {
            schema.items = sanitizeSchema(items);
        } else {
            schema.items = { type: "string" } as Record<string, unknown>;
        }
    }

    return schema;
}

/**
 * Convert VS Code chat request messages into OpenAI-compatible message objects.
 * @param messages The VS Code chat messages to convert.
 * @returns OpenAI-compatible messages array.
 */
/**
 * Converts VS Code language model chat messages to OpenAI-compatible format.
 * Transforms message roles and content to match OpenAI's chat completion API.
 *
 * @param messages - Array of VS Code chat messages to convert.
 * @returns Array of OpenAI-compatible chat messages.
 */
export function convertMessages(messages: readonly vscode.LanguageModelChatRequestMessage[]): OpenAIChatMessage[] {
	const raw: OpenAIChatMessage[] = [];
	for (const m of messages) {
		const role = mapRole(m);
		const textParts: string[] = [];
		const toolCalls: OpenAIToolCall[] = [];
		const toolResults: { callId: string; content: string }[] = [];

		for (const part of m.content ?? []) {
			if (part instanceof vscode.LanguageModelTextPart) {
				textParts.push(part.value);
			} else if (part instanceof vscode.LanguageModelToolCallPart) {
				const id = part.callId || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
				let args = "{}";
				try {
					args = JSON.stringify(part.input ?? {});
				} catch {
					args = "{}";
				}
				toolCalls.push({ id, type: "function", function: { name: part.name, arguments: args } });
			} else if (isToolResultPart(part)) {
				const callId = (part as { callId?: string }).callId ?? "";
				const content = collectToolResultText(part as { content?: ReadonlyArray<unknown> });
				toolResults.push({ callId, content });
			}
		}

		let emittedAssistantToolCall = false;
		if (toolCalls.length > 0) {
			raw.push({ role: "assistant", content: textParts.join("") || undefined, tool_calls: toolCalls });
			emittedAssistantToolCall = true;
		}

		for (const tr of toolResults) {
			raw.push({ role: "user", content: tr.content || "" });
		}

		const text = textParts.join("");
		if (text && (role === "system" || role === "user" || (role === "assistant" && !emittedAssistantToolCall))) {
			raw.push({ role, content: text });
		}
	}

	// Post-process to merge consecutive messages of the same role (User/System/Assistant)
	// Post-process: Hoist all System messages to the very top and merge them.
	// This prevents System messages from appearing in the middle of conversation (e.g. User -> System -> User),
	// which causes Jinja template errors in many Llama.cpp models.
	const systemMessages = raw.filter((m) => m.role === "system");
	const nonSystemMessages = raw.filter((m) => m.role !== "system");

	if (systemMessages.length > 0) {
		const mergedSystemContent = systemMessages
			.map((m) => m.content)
			.filter((c) => typeof c === "string" && c.trim().length > 0)
			.join("\n\n");

		if (mergedSystemContent) {
			nonSystemMessages.unshift({ role: "system", content: mergedSystemContent });
		}
	}

	// Post-process to merge consecutive messages of the same role (User/System/Assistant)
	const merged: OpenAIChatMessage[] = [];
	for (const msg of nonSystemMessages) {
		if (merged.length === 0) {
			merged.push(msg);
			continue;
		}
		const last = merged[merged.length - 1];

		// Case 1: Merge consecutive Assistant messages (text and/or tool calls)
		if (msg.role === "assistant" && last.role === "assistant") {
			if (msg.content) {
				last.content = last.content ? last.content + "\n\n" + msg.content : msg.content;
			}
			if (msg.tool_calls) {
				last.tool_calls = [...(last.tool_calls ?? []), ...msg.tool_calls];
			}
			continue;
		}


		// Case 2: Merge consecutive "User-side" messages (User text or Tool results)
		// Strict templates often require strict alternation [User, Assistant, User, Assistant]
		// So we merge all [User, Tool, User, Tool...] sequences into a single User message.

		const isLastUserSide =
			(last.role === "user" && typeof last.content === "string" && !last.tool_calls) ||
			last.role === "tool";

		const isMsgUserSide =
			(msg.role === "user" && typeof msg.content === "string" && !msg.tool_calls) ||
			msg.role === "tool";

		if (isLastUserSide && isMsgUserSide) {
			// Ensure target is a Text User message
			if (last.role === "tool") {
				last.role = "user";
				delete last.tool_call_id;
			}

			const nextContent = typeof msg.content === "string" ? msg.content : "";
			last.content = (last.content || "") + "\n\n" + nextContent;
			continue;
		}

		merged.push(msg);
	}
	return merged;
}

/**
 * Convert VS Code tool definitions to OpenAI function tool definitions.
 * @param options Request options containing tools and toolMode.
 */
/**
 * Converts VS Code language model chat options to OpenAI-compatible tool format.
 * Extracts and transforms tool definitions for API requests.
 *
 * @param options - VS Code chat response options containing tools.
 * @returns Object with tools array and tool_choice configuration.
 */
export function convertTools(options: vscode.ProvideLanguageModelChatResponseOptions): {
	tools?: OpenAIFunctionToolDef[];
	tool_choice?: "auto" | "required" | { type: "function"; function: { name: string } };
} {
	const tools = options.tools ?? [];
	if (!tools || tools.length === 0) {
		return {};
	}

	const toolDefs: OpenAIFunctionToolDef[] = tools
		.filter((t) => t && typeof t === "object")
		.map((t) => {
			const name = sanitizeFunctionName(t.name);
			const description = typeof t.description === "string" ? t.description : "";
			const params = sanitizeSchema(t.inputSchema ?? { type: "object", properties: {} });
			return {
				type: "function" as const,
				function: {
					name,
					description,
					parameters: params,
				},
			} satisfies OpenAIFunctionToolDef;
		});

	let tool_choice: "auto" | "required" | { type: "function"; function: { name: string } } = "auto";
	if (options.toolMode === vscode.LanguageModelChatToolMode.Required) {
		if (tools.length === 1) {
			tool_choice = { type: "function", function: { name: sanitizeFunctionName(tools[0].name) } };
		} else {
			tool_choice = "required";
		}
	}

	return { tools: toolDefs, tool_choice };
}

/**
 * Validate tool names to ensure they contain only word chars, hyphens, or underscores.
 * @param tools Tools to validate.
 */
/**
 * Validates an array of VS Code language model chat tools.
 * Ensures tool definitions are properly structured before use.
 *
 * @param tools - Array of tools to validate.
 */
export function validateTools(tools: readonly vscode.LanguageModelChatTool[]): void {
	for (const tool of tools) {
		if (!tool.name.match(/^[\w-]+$/)) {
            console.error("[Llama.cpp Chat Provider] Invalid tool name detected:", tool.name);
            throw new Error(
                `Invalid tool name "${tool.name}": only alphanumeric characters, hyphens, and underscores are allowed.`
            );
		}
	}
}

/**
 * Validate the request message sequence for correct tool call/result pairing.
 * @param messages The full request message list.
 */
/**
 * Validates an array of VS Code language model chat request messages.
 * Checks for proper message structure and content.
 *
 * @param messages - Array of messages to validate.
 */
export function validateRequest(messages: readonly vscode.LanguageModelChatRequestMessage[]): void {
	const lastMessage = messages[messages.length - 1];
	if (!lastMessage) {
    console.error("[Llama.cpp Chat Provider] No messages in request");
    throw new Error("Invalid request: no messages.");
	}

	messages.forEach((message, i) => {
		if (message.role === vscode.LanguageModelChatMessageRole.Assistant) {
			const toolCallIds = new Set(
				message.content
					.filter((part) => part instanceof vscode.LanguageModelToolCallPart)
					.map((part) => (part as unknown as vscode.LanguageModelToolCallPart).callId)
			);
			if (toolCallIds.size === 0) {
				return;
			}

			let nextMessageIdx = i + 1;
			const errMsg =
				"Invalid request: Tool call part must be followed by a User message with a LanguageModelToolResultPart with a matching callId.";
			while (toolCallIds.size > 0) {
				const nextMessage = messages[nextMessageIdx++];
				if (!nextMessage || nextMessage.role !== vscode.LanguageModelChatMessageRole.User) {
                    console.error("[Llama.cpp Chat Provider] Validation failed: missing tool result for call IDs:", Array.from(toolCallIds));
                    throw new Error(errMsg);
				}

				nextMessage.content.forEach((part) => {
					if (!isToolResultPart(part)) {
						const ctorName =
							(Object.getPrototypeOf(part as object) as { constructor?: { name?: string } } | undefined)?.constructor
								?.name ?? typeof part;
                        console.error("[Llama.cpp Chat Provider] Validation failed: expected tool result part, got:", ctorName);
                        throw new Error(errMsg);
					}
					const callId = (part as { callId: string }).callId;
					toolCallIds.delete(callId);
				});
			}
		}
	});
}

/**
 * Type guard for LanguageModelToolResultPart-like values.
 * @param value Unknown value to test.
 */
/**
 * Type guard to check if a value is a tool result part.
 * Determines if the value represents a tool call result with callId and content.
 *
 * @param value - The value to check.
 * @returns True if the value is a tool result part, false otherwise.
 */
export function isToolResultPart(value: unknown): value is { callId: string; content?: ReadonlyArray<unknown> } {
	if (!value || typeof value !== "object") {
		return false;
	}
	const obj = value as Record<string, unknown>;
	const hasCallId = typeof obj.callId === "string";
	const hasContent = "content" in obj;
	return hasCallId && hasContent;
}

/**
 * Map VS Code message role to OpenAI message role string.
 * @param message The message whose role is mapped.
 */
/**
 * Maps a VS Code chat message to an OpenAI-compatible role.
 * Converts VS Code message types to OpenAI roles, excluding tool role.
 *
 * @param message - The VS Code chat message.
 * @returns The corresponding OpenAI role.
 * @author Maruf Bepary
 */
function mapRole(message: vscode.LanguageModelChatRequestMessage): Exclude<OpenAIChatRole, "tool"> {
	const USER = vscode.LanguageModelChatMessageRole.User as unknown as number;
	const ASSISTANT = vscode.LanguageModelChatMessageRole.Assistant as unknown as number;
	const r = message.role as unknown as number;
	if (r === USER) {
		return "user";
	}
	if (r === ASSISTANT) {
		return "assistant";
	}
	return "system";
}

/**
 * Concatenate tool result content into a single text string.
 * @param pr Tool result-like object with content array.
 */
/**
 * Collects text content from a tool result part.
 * Extracts and concatenates text from the content array.
 *
 * @param pr - The tool result part with content.
 * @returns The concatenated text content.
 */
function collectToolResultText(pr: { content?: ReadonlyArray<unknown> }): string {
	let text = "";
	for (const c of pr.content ?? []) {
		if (c instanceof vscode.LanguageModelTextPart) {
			text += c.value;
		} else if (typeof c === "string") {
			text += c;
		} else {
			try {
				text += JSON.stringify(c);
			} catch {
				/* ignore */
			}
		}
	}
	return text;
}

/**
 * Try to parse a JSON object from a string.
 * @param text The input string.
 * @returns Parsed object or ok:false.
 */
/**
 * Attempts to parse a string as JSON object.
 * Safely parses JSON and returns success/failure result.
 *
 * @param text - The string to parse as JSON.
 * @returns Object with ok flag and parsed value if successful.
 */
export function tryParseJSONObject(text: string): { ok: true; value: Record<string, unknown> } | { ok: false } {
	try {
		if (!text || !/[{]/.test(text)) {
			return { ok: false };
		}
		const value = JSON.parse(text);
		if (value && typeof value === "object" && !Array.isArray(value)) {
			return { ok: true, value };
		}
		return { ok: false };
	} catch {
		return { ok: false };
	}
}
