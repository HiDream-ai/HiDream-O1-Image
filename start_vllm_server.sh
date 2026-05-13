vllm serve HiDream-ai/gemma-4-31B-it-Prompt-Refine \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --chat-template HiDream-ai/gemma-4-31B-it-Prompt-Refine/tool_chat_template_gemma4.jinja \
    --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
    --port 8000
