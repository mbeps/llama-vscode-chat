import * as assert from "assert";
import * as vscode from "vscode";
import { LlamaCppChatModelProvider } from "../llama-provider";
import { OpenAIChatMessage } from "../types";
import { convertMessages, convertTools, validateRequest } from "../utils";

// Mock SecretStorage
class MockSecretStorage implements vscode.SecretStorage {
    private secrets = new Map<string, string>();
    get(key: string): Thenable<string | undefined> {
        return Promise.resolve(this.secrets.get(key));
    }
    store(key: string, value: string): Thenable<void> {
        this.secrets.set(key, value);
        return Promise.resolve();
    }
    delete(key: string): Thenable<void> {
        this.secrets.delete(key);
        return Promise.resolve();
    }
    keys(): Thenable<string[]> {
        return Promise.resolve(Array.from(this.secrets.keys()));
    }
    onDidChange: vscode.Event<vscode.SecretStorageChangeEvent> = new vscode.EventEmitter<vscode.SecretStorageChangeEvent>().event;
}

suite("Llama.cpp Chat Provider Extension", () => {
    suite("provider", () => {
        const secretStorage = new MockSecretStorage();
        const provider = new LlamaCppChatModelProvider(secretStorage, "test-user-agent");

        test("provideLanguageModelChatInformation returns array (defaults)", async () => {
            const infos = await provider.provideLanguageModelChatInformation(
                { silent: true },
                new vscode.CancellationTokenSource().token
            );
            // It might fail if no server running, but it returns array (empty or populated)
            assert.ok(Array.isArray(infos));
        });

        test("provideTokenCount calculation for text", async () => {
            const count = await provider.provideTokenCount(
                {} as vscode.LanguageModelChatInformation,
                "hello world",
                new vscode.CancellationTokenSource().token
            );
            assert.strictEqual(count, 3); // "hello world".length / 4 ceil = 11/4 = 2.75 -> 3
        });
    });

    suite("utils/convertMessages", () => {
        test("maps user/assistant text", () => {
            const messages: vscode.LanguageModelChatMessage[] = [
                {
                    role: vscode.LanguageModelChatMessageRole.User,
                    content: [new vscode.LanguageModelTextPart("hi")],
                    name: undefined,
                },
                {
                    role: vscode.LanguageModelChatMessageRole.Assistant,
                    content: [new vscode.LanguageModelTextPart("hello")],
                    name: undefined,
                },
            ];
            const out: OpenAIChatMessage[] = convertMessages(messages);
            assert.deepEqual(out, [
                { role: "user", content: "hi" },
                { role: "assistant", content: "hello" },
            ]);
        });

        test("merges consecutive user messages", () => {
            const messages: vscode.LanguageModelChatMessage[] = [
                {
                    role: vscode.LanguageModelChatMessageRole.User,
                    content: [new vscode.LanguageModelTextPart("context")],
                    name: undefined,
                },
                {
                    role: vscode.LanguageModelChatMessageRole.User,
                    content: [new vscode.LanguageModelTextPart("query")],
                    name: undefined,
                },
            ];
            const out: OpenAIChatMessage[] = convertMessages(messages);
            // Expectation: merged into one message
            assert.strictEqual(out.length, 1);
            assert.strictEqual(out[0].role, "user");
            assert.ok(out[0].content!.includes("context"));
            assert.ok(out[0].content!.includes("query"));
        });

        test("merges consecutive assistant messages (text + tool call)", () => {
            const messages: vscode.LanguageModelChatMessage[] = [
                {
                    role: vscode.LanguageModelChatMessageRole.Assistant,
                    content: [new vscode.LanguageModelTextPart("thinking...")],
                    name: undefined,
                },
                {
                    role: vscode.LanguageModelChatMessageRole.Assistant,
                    content: [new vscode.LanguageModelToolCallPart("call1", "my_tool", { a: 1 })],
                    name: undefined,
                },
            ];
            const out: OpenAIChatMessage[] = convertMessages(messages);
            assert.strictEqual(out.length, 1);
            assert.strictEqual(out[0].role, "assistant");
            assert.strictEqual(out[0].content, "thinking...");
            assert.ok(out[0].tool_calls && out[0].tool_calls.length === 1);
            assert.strictEqual(out[0].tool_calls[0].function.name, "my_tool");
        });


        test("merges consecutive tool messages", () => {
             const messages: vscode.LanguageModelChatMessage[] = [
                {
                    role: vscode.LanguageModelChatMessageRole.User,
                    content: [new vscode.LanguageModelToolResultPart("id1", [new vscode.LanguageModelTextPart("res1")])],
                    name: undefined,
                },
                {
                    role: vscode.LanguageModelChatMessageRole.User,
                    content: [new vscode.LanguageModelToolResultPart("id2", [new vscode.LanguageModelTextPart("res2")])],
                    name: undefined,
                },
            ];
            const out: OpenAIChatMessage[] = convertMessages(messages);
            // Expectation: merged into single User message with combined text
            assert.strictEqual(out.length, 1);
            assert.strictEqual(out[0].role, "user");
            assert.ok(out[0].content!.includes("res1"));
            assert.ok(out[0].content!.includes("res2"));
        });
        test("merges user (text) into tool message", () => {
            const messages: vscode.LanguageModelChatMessage[] = [
               {
                   role: vscode.LanguageModelChatMessageRole.User,
                   content: [new vscode.LanguageModelTextPart("context")],
                   name: undefined,
               },
               {
                   role: vscode.LanguageModelChatMessageRole.User,
                   content: [new vscode.LanguageModelToolResultPart("id1", [new vscode.LanguageModelTextPart("res1")])],
                   name: undefined,
               },
           ];
           const out: OpenAIChatMessage[] = convertMessages(messages);
           assert.strictEqual(out.length, 1);
           assert.strictEqual(out[0].role, "user");
           assert.ok(out[0].content!.includes("context"));
           assert.ok(out[0].content!.includes("res1"));
       });

       test("merges tool message and user (text)", () => {
           const messages: vscode.LanguageModelChatMessage[] = [
              {
                  role: vscode.LanguageModelChatMessageRole.User,
                  content: [new vscode.LanguageModelToolResultPart("id1", [new vscode.LanguageModelTextPart("res1")])],
                  name: undefined,
              },
              {
                  role: vscode.LanguageModelChatMessageRole.User,
                  content: [new vscode.LanguageModelTextPart("followup")],
                  name: undefined,
              },
          ];
          const out: OpenAIChatMessage[] = convertMessages(messages);
          assert.strictEqual(out.length, 1);
          assert.strictEqual(out[0].role, "user");
          assert.ok(out[0].content!.includes("res1"));
          assert.ok(out[0].content!.includes("followup"));
      });

      test("hoists system messages to the top", () => {
          // The type is 'readonly vscode.LanguageModelChatMessage[]'.
          const sysMsg = { role: 3, content: [new vscode.LanguageModelTextPart("sys instruction")] } as unknown as vscode.LanguageModelChatMessage;
          const userMsg = { role: vscode.LanguageModelChatMessageRole.User, content: [new vscode.LanguageModelTextPart("hi")] } as vscode.LanguageModelChatMessage;

          const msgs = [userMsg, sysMsg];
          const out: OpenAIChatMessage[] = convertMessages(msgs);

          // Expect: [System, User]
          assert.strictEqual(out.length, 2);
          assert.strictEqual(out[0].role, "system");
          assert.strictEqual(out[0].content, "sys instruction");
          assert.strictEqual(out[1].role, "user");
          assert.strictEqual(out[1].content, "hi");
      });


    });

    suite("utils/tools", () => {
        test("convertTools returns function tool definitions", () => {
			const out = convertTools({
				tools: [
					{
						name: "do_something",
						description: "Does something",
						inputSchema: { type: "object", properties: { x: { type: "number" } }, additionalProperties: false },
					},
				],
				toolMode: vscode.LanguageModelChatToolMode.Auto,
			} satisfies vscode.ProvideLanguageModelChatResponseOptions);

			assert.ok(out);
			assert.equal(out.tool_choice, "auto");
			assert.ok(Array.isArray(out.tools) && out.tools[0].type === "function");
			assert.equal(out.tools[0].function.name, "do_something");
		});

		test("convertTools respects ToolMode.Required for single tool", () => {
			const out = convertTools({
				toolMode: vscode.LanguageModelChatToolMode.Required,
				tools: [
					{
						name: "only_tool",
						description: "Only tool",
						inputSchema: {},
					},
				],
			} satisfies vscode.ProvideLanguageModelChatResponseOptions);
			assert.deepEqual(out.tool_choice, { type: "function", function: { name: "only_tool" } });
		});
    });

    suite("utils/validation", () => {
        test("validateRequest enforces tool result pairing", () => {
            const callId = "xyz";
            const toolCall = new vscode.LanguageModelToolCallPart(callId, "toolA", { q: 1 });
            const toolRes = new vscode.LanguageModelToolResultPart(callId, [new vscode.LanguageModelTextPart("ok")]);
            const valid = [
                { role: vscode.LanguageModelChatMessageRole.Assistant, content: [toolCall], name: undefined },
                { role: vscode.LanguageModelChatMessageRole.User, content: [toolRes], name: undefined },
            ];
            assert.doesNotThrow(() => validateRequest(valid));

            const invalid = [
                { role: vscode.LanguageModelChatMessageRole.Assistant, content: [toolCall], name: undefined },
                { role: vscode.LanguageModelChatMessageRole.User, content: [new vscode.LanguageModelTextPart("missing")], name: undefined },
            ];
            assert.throws(() => validateRequest(invalid));
        });
    });
});
