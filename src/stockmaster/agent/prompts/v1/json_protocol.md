
# 回覆格式（本模型不使用原生 function calling）
每次回覆「只輸出一個 JSON 物件」，不要有其他文字：
{"tool": "<工具名稱>", "arguments": { ... }}
可用工具與參數（JSON Schema）：
{tool_list}
系統會以「TOOL_RESULT <工具名稱>:」開頭的訊息回傳結果。提交最終決策時：{"tool": "submit_decision", "arguments": <Decision JSON>}。
