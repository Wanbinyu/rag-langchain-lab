# Resume Fit AI security and privacy boundaries

Resumes and job descriptions are untrusted model input. They can contain prompt-injection text that tries to override instructions, reveal the system prompt, force a score, or pollute the JSON structure.

## Layered input protection

Before a model request, text is normalized with Unicode NFKC. Zero-width characters, bidirectional control characters, and unsafe control characters are removed. Detection covers Chinese and English override phrases, forged system/developer/assistant/tool role labels, spaced-character variants, URL-encoded text, and a bounded Base64 decoding pass. A high-confidence match is rejected before consuming model tokens.

Regular expressions are only one layer and cannot prove prompt-injection safety. The prompt clearly labels resume and JD fields as data, repeats the allowed task after the data boundary, and gives the model no file, database, messaging, browsing, or code-execution tools.

## Leak and output protection

Each model request includes a random guard token visible only in the system message. If the model output contains that token, or a distinctive system-prompt fragment, the response is discarded as a possible leak. Upstream error bodies are not returned verbatim to users.

Model JSON passes through an allowlist and size limits before it reaches the frontend. Unknown fields are discarded. Browser rendering escapes model text and uses a Content Security Policy, reducing the impact of an unexpected payload.

## Access and retention

Uploaded resumes are parsed in memory and are not written to disk. API keys remain in server environment variables. Task reports require the random task access token. The administrative statistics endpoint is disabled unless a separate token of at least 24 characters is configured.

The privacy notice explains that resume data and the JD are sent over an encrypted connection to the configured AI provider. Users must explicitly accept the privacy policy and terms before analysis, and should not submit identity numbers, bank details, passwords, home addresses, or unrelated sensitive information.
