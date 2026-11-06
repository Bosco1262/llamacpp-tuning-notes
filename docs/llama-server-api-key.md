# Adding an API Key to `llama-server`

[English](llama-server-api-key.md) | [简体中文](llama-server-api-key.zh-CN.md)

`llama-server` can take an API key directly via the `--api-key` or `--api-key-file` flag, so that your local inference service isn't available to anything that can reach the port. Pick whichever of the two options below matches your security needs.

---

## Option 1: Pass the key on the command line (simplest)

When starting `llama-server`, just append the `--api-key` flag with the secret you want to set:

```bat
llama-server -m your_model.gguf --api-key "your-secret-key"
```

## Option 2: Load the key from a file (safer)

If the key is very long, or you'd rather not have it exposed on the process command line (where `ps` or Task Manager can read it), put the key in a file:

1. Create a text file named `api_key.txt` and write your key into it.
2. Start the server with `--api-key-file` pointing at that file:

```bat
llama-server -m your_model.gguf --api-key-file /path/to/api_key.txt
```

> **Tip**: on Linux/macOS, `chmod 600 api_key.txt` first to restrict permissions, and make sure the file is never committed to Git (add it to `.gitignore`).

---

## Calling the server from a client

Once a key is set, every request sent to `llama-server` must carry it in an HTTP header — otherwise the server returns `401 Unauthorized`.

### Example 1: testing with cURL

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Example 2: calling from Python (OpenAI SDK)

Because `llama-server` is fully OpenAI-API-compatible, you only need to pass your key when constructing the client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-secret-key",  # the key you configured
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

Point `base_url` at your server and swap in the real key, and the same snippet works for third-party front ends such as Open WebUI or Dify, since they all speak the same OpenAI-compatible API.

---

## References

1. <https://unsloth.ai/docs/zh/ji-cheng/connections/jiang-llama.cpp-lian-jie-dao-unsloth-shi-yong-llamaserver-yun-xing-gguf>
2. <https://adg.csdn.net/697090fa437a6b40336abebd.html>
3. <https://trilium.atibm.com/share/softwareaillamaserverparam>
4. <https://zhuanlan.zhihu.com/p/2046232859321619837>
5. <https://openwebui-doc-zh.pages.dev/getting-started/quick-start/starting-with-llama-cpp/>
