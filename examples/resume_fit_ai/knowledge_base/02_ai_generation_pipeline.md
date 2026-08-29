# Resume Fit AI generation pipeline

## Provider abstraction

The service supports DeepSeek, Qwen, OpenAI, and a custom OpenAI-compatible endpoint. Provider configuration normalizes the display name, base URL, API key, model name, timeout, and token usage. Compatibility is limited to the common chat-completions request shape; provider-specific tool calling would require a dedicated adapter.

## Structured output as untrusted input

The model is instructed to return a JSON report with a fixed schema. A response is not trusted merely because it parses as JSON. The server normalizes the report before rendering: scores are clamped to 0–100, arrays and strings have length limits, nested objects are checked field by field, invalid types are replaced or removed, and unknown properties are discarded.

This runtime normalization prevents unexpected model fields or oversized content from flowing directly into the browser and export functions. The frontend uses text nodes or escaped HTML and applies a Content Security Policy.

## Factual fidelity repair

A long, structured rewrite can omit resume sections, technologies, or project details. After full generation, the server performs deterministic coverage checks against the original resume. If too much supported information disappeared, it makes at most one targeted fidelity-repair model request.

The repair result is compared with the first draft. It is accepted only when the coverage penalty improves; otherwise the original draft remains. This two-stage design reduces omissions without creating an unbounded self-correction loop. It lowers hallucination risk but does not remove the need for user review.

## Fast and full prompt budgets

Fast diagnosis and full generation use separate prompts and output-token limits. The fast path deliberately asks for fewer detailed fields, while the full path pays the latency and cost of a complete rewrite and optional fidelity repair. This is both a product-tier decision and a resource-control mechanism.
