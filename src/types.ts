import * as vscode from "vscode";

/**
 * OpenAI function-call entry emitted by assistant messages.
 * Represents a tool call initiated by the assistant in a chat response.
 *
 */
export interface OpenAIToolCall {
	/**
	 * Unique identifier for the tool call.
	 */
	id: string;
	/**
	 * Type of the tool call, always "function".
	 */
	type: "function";
	/**
	 * Details of the function to call, including name and arguments.
	 */
	function: { name: string; arguments: string };
}

/**
 * Extended VS Code LanguageModelChatInformation to include custom properties.
 */
export interface ExtendedLanguageModelChatInformation extends vscode.LanguageModelChatInformation {
	/**
	 * Whether the model is user selectable in the model picker.
	 */
	isUserSelectable?: boolean;
}

/**
 * OpenAI function tool definition used to advertise tools.
 * Defines a tool that can be called by the model, including its name, description, and parameters.
 *
 */
export interface OpenAIFunctionToolDef {
	/**
	 * Type of the tool, always "function".
	 */
	type: "function";
	/**
	 * Function details including name, description, and parameter schema.
	 */
	function: { name: string; description?: string; parameters?: object };
}

/**
 * OpenAI-style chat message used for router requests.
 * Represents a message in a chat conversation, compatible with OpenAI's API format.
 *
 */
export interface OpenAIChatMessage {
	/**
	 * Role of the message sender (system, user, assistant, or tool).
	 */
	role: OpenAIChatRole;
	/**
	 * Content of the message, optional for some roles.
	 */
	content?: string;
	/**
	 * Name of the sender, optional.
	 */
	name?: string;
	/**
	 * Tool calls made by the assistant, if any.
	 */
	tool_calls?: OpenAIToolCall[];
	/**
	 * ID of the tool call this message is responding to, if applicable.
	 */
	tool_call_id?: string;
}

/**
 * Llama.cpp server properties response from the `/props` endpoint.
 * Contains default generation settings including context size.
 */
export interface LlamaCppServerProps {
	/**
	 * Default generation settings for the server.
	 */
	default_generation_settings: {
		/**
		 * The default context size (number of tokens).
		 */
		n_ctx: number;
		[key: string]: unknown;
	};
	[key: string]: unknown;
}

/**
 * Buffer used to accumulate streamed tool call parts until arguments are valid JSON.
 * Helps in assembling incomplete tool call data from streaming responses.
 *
 */
export interface ToolCallBuffer {
	/**
	 * Optional unique identifier for the tool call.
	 */
	id?: string;
	/**
	 * Optional name of the tool being called.
	 */
	name?: string;
	/**
	 * Accumulated arguments string for the tool call.
	 */
	args: string;
}

/**
 * OpenAI-style chat roles.
 * Defines the possible roles for messages in a chat conversation.
 *
 */
export type OpenAIChatRole = "system" | "user" | "assistant" | "tool";
