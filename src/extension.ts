
import * as vscode from "vscode";

import { LlamaCppChatModelProvider } from "./llama-provider";

/**
 * Activates the VS Code extension.
 * Registers the Llama.cpp chat model provider and management commands.
 * Called when the extension is activated by VS Code.
 *
 * @param context - The extension context provided by VS Code.
 */
export function activate(context: vscode.ExtensionContext) {
	// Build a descriptive User-Agent to help quantify API usage
	const ext = vscode.extensions.getExtension("Llama.llama-vscode-chat");
	const extVersion = ext?.packageJSON?.version ?? "unknown";
	const vscodeVersion = vscode.version;
	// Keep UA minimal: only extension version and VS Code version
	const ua = `llama-vscode-chat/${extVersion} VSCode/${vscodeVersion}`;



	// Llama.cpp Provider
	const llamaProvider = new LlamaCppChatModelProvider(context.secrets, ua);
	context.subscriptions.push(vscode.lm.registerLanguageModelChatProvider("llamacpp", llamaProvider));

	context.subscriptions.push(
		vscode.commands.registerCommand("llamacpp.manage", async () => {
			const existingUrl = await context.secrets.get("llamacpp.serverUrl");
			const serverUrl = await vscode.window.showInputBox({
				title: "Llama.cpp Server URL",
				prompt: "Enter the URL of your Llama.cpp server",
				value: existingUrl || "http://localhost:8080",
				ignoreFocusOut: true,
			});

			if (serverUrl === undefined) {
				return; // User canceled
			}

			if (serverUrl.trim()) {
				await context.secrets.store("llamacpp.serverUrl", serverUrl.trim());
			} else {
				await context.secrets.delete("llamacpp.serverUrl");
			}

			llamaProvider.notifyConfigChanged();

			vscode.window.showInformationMessage("Llama.cpp configuration saved.");
		})
	);
}

/**
 * Deactivates the VS Code extension.
 * Performs cleanup when the extension is deactivated.
 */
export function deactivate() {}
