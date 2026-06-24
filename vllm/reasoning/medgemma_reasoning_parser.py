# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module("medgemma")
class MedGemmaReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MedGemma-style hidden reasoning. This adapts the DeepSeekR1 reasoning parser to work with
    MedGemma's reasoning tokens. 

    MedGemma uses <unused94> to denote the start of reasoning and finishes reasoning with <unused95>. This is different
    from how Gemma3 style models typically work, as they do not use CoT reasoning. Google however specifically trained
    MedGemma to use reasoning tokens. 
    
    The standard chat template included with MedGemma Models on huggingface does not support enabling reasoning, but the
    model sometimes still choses to generate reasoning tokens. This parser will handle this case correctly. There is 
    also a custom template in examples/reasoning/reasoning_chat_template_medgemma that understands the Qwen / Gemma4 
    style "enable_thinking" template argument. This forces the model into reasoning mode.

    To use both this parser and the reasoning template use
    ```
        --reasoning-parser medgemma 
        --default-chat-template-kwargs '{"enable_thinking":true}
        --chat-template examples/reasoning/reasoning_chat_template_medgemma.jinja
    ```

    The chat template is not required if you still want to let the model choose deliberately when to use reasoning.
    """

    @property
    def start_token(self) -> str:
        return "<unused94>"

    @property
    def end_token(self) -> str:
        return "<unused95>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        
        # The chat template primed the assistant turn with <unused94>.
        # Therefore the generated delta stream may never contain start_token.
        # Until <unused95> appears, generated text should still be treated
        # as reasoning.
        if (
            ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )

            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text or None)

            return DeltaMessage(reasoning=delta_text or None)

        return ret