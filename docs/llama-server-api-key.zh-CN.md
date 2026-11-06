# 为 llama-server 添加 API 密钥

[English](llama-server-api-key.md) | [简体中文](llama-server-api-key.zh-CN.md)

`llama-server` 可以通过 `--api-key` 或 `--api-key-file` 参数直接设置 API 密钥（API Key），以保护你的本地推理服务不被滥用。根据你的安全需求，选择以下两种方式之一。

---

## 方式一：直接在命令行设置密钥（最简单）

启动 `llama-server` 时，直接在命令后面追加 `--api-key` 参数并输入你想设置的密码：

```bat
llama-server -m your_model.gguf --api-key "your-secret-key"
```

## 方式二：通过文件加载密钥（更安全）

如果密钥非常长，或者不希望密钥暴露在系统进程的命令行里（`ps`、任务管理器都能看到），可以把密钥写入文件：

1. 创建一个名为 `api_key.txt` 的文本文件，在里面写入你的密钥。
2. 启动服务时，使用 `--api-key-file` 指向该文件：

```bat
llama-server -m your_model.gguf --api-key-file /path/to/api_key.txt
```

> **提示**：Linux/macOS 下建议先 `chmod 600 api_key.txt` 限制权限，并确认该文件没有被提交进 Git 仓库（加入 `.gitignore`）。

---

## 客户端如何调用

设置好密钥后，所有向 `llama-server` 发送的请求都必须在 HTTP 请求头（Header）中携带该密钥，否则服务器会返回 `401 Unauthorized` 拒绝访问。

### 示例 1：使用 cURL 测试

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 示例 2：在 Python（OpenAI SDK）中调用

因为 `llama-server` 完美兼容 OpenAI API 格式，你只需在初始化时传入你的密钥即可：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-secret-key",  # 填入你设置的密钥
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

把 `base_url` 指向你的服务地址、`api_key` 换成实际密钥后，这段代码可以直接对接 Open WebUI、Dify 等第三方前端所依赖的同一套 OpenAI 兼容接口。

---

## 参考资料

1. <https://unsloth.ai/docs/zh/ji-cheng/connections/jiang-llama.cpp-lian-jie-dao-unsloth-shi-yong-llamaserver-yun-xing-gguf>
2. <https://adg.csdn.net/697090fa437a6b40336abebd.html>
3. <https://trilium.atibm.com/share/softwareaillamaserverparam>
4. <https://zhuanlan.zhihu.com/p/2046232859321619837>
5. <https://openwebui-doc-zh.pages.dev/getting-started/quick-start/starting-with-llama-cpp/>
