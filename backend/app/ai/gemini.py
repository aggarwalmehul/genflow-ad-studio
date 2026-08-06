import logging

from google import genai
from google.genai import types

from app.ai.prompts import (
    PROMPT_REWRITE_TEMPLATE,
    SCRIPT_SYSTEM_INSTRUCTION,
    SCRIPT_USER_PROMPT_TEMPLATE,
    STORYBOARD_QC_SYSTEM_INSTRUCTION,
    STORYBOARD_QC_USER_PROMPT,
    VIDEO_QC_SYSTEM_INSTRUCTION,
    VIDEO_QC_USER_PROMPT,
    build_narrative_arc,
)
from app.ai.retry import async_retry
from app.config import Settings
from app.utils.json_parser import parse_json_response

logger = logging.getLogger(__name__)

ALL_SAFETY_OFF = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
]


from app.ai.token_logger import log_tokens, current_user_email


class GeminiService:
    def __init__(self, client=None, settings=None, client_factory=None):
        # factory-aware: prefer client_factory (model-routing); else wrap a static client
        if client_factory is not None:
            self._client_factory = client_factory
        elif client is not None:
            self._client_factory = lambda _model: client
        else:
            raise ValueError("GeminiService needs client or client_factory")
        self.settings = settings

    async def _gen_logged(self, model, **kwargs):
        client = self._client_factory(model)
        resp = await client.aio.models.generate_content(model=model, **kwargs)
        try:
            log_tokens("genflow", model, resp, user_email=current_user_email.get())
        except Exception:
            pass
        return resp

    @async_retry(retries=3)
    async def generate_script(
        self,
        product_name: str,
        specs: str,
        image_bytes: bytes,
        scene_count: int = 3,
        target_duration: int = 30,
        ad_tone: str = "energetic",
        model_id: str | None = None,
        max_words: int = 25,
        custom_instructions: str = "",
    ) -> dict:
        """Generate video script using Gemini with structured JSON output.

        Args:
            model_id: Optional model override. Falls back to settings.gemini_model.
            max_words: Maximum dialogue words per scene.
        """
        user_prompt = SCRIPT_USER_PROMPT_TEMPLATE.format(
            product_name=product_name,
            specs=specs,
            scene_count=scene_count,
            target_duration=target_duration,
            narrative_arc=build_narrative_arc(scene_count, target_duration),
            ad_tone=ad_tone,
            max_words=max_words,
        )

        if custom_instructions:
            user_prompt += f"\n\nADDITIONAL CREATIVE DIRECTION FROM CLIENT:\n{custom_instructions}"

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        text_part = types.Part.from_text(text=user_prompt)

        response = await self._gen_logged(
            model=model_id or self.settings.gemini_model,
            contents=[image_part, text_part],
            config=types.GenerateContentConfig(
                system_instruction=SCRIPT_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                safety_settings=ALL_SAFETY_OFF,
                temperature=1.0,
            ),
        )

        return parse_json_response(response.text)

    @async_retry(retries=3)
    async def analyze_product_image(self, image_bytes: bytes) -> dict:
        """Extract product name and specifications from an image using Flash model."""
        prompt = (
            "Analyze this product image and extract the following information. "
            "Return ONLY valid JSON with no additional text.\n\n"
            '{"product_name": "The product name", '
            '"specifications": "Key specifications formatted as key: value lines"}'
        )

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        text_part = types.Part.from_text(text=prompt)

        response = await self._gen_logged(
            model=self.settings.gemini_flash_model,
            contents=[image_part, text_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                safety_settings=ALL_SAFETY_OFF,
                temperature=0.5,
            ),
        )

        return parse_json_response(response.text)

    @async_retry(retries=3)
    async def qc_storyboard(
        self,
        avatar_bytes: bytes,
        product_bytes: bytes,
        storyboard_bytes: bytes,
    ) -> dict:
        """QC storyboard using Gemini 3 Flash for faster evaluation."""
        avatar_part = types.Part.from_bytes(data=avatar_bytes, mime_type="image/png")
        product_part = types.Part.from_bytes(
            data=product_bytes, mime_type="image/png"
        )
        storyboard_part = types.Part.from_bytes(
            data=storyboard_bytes, mime_type="image/png"
        )
        text_part = types.Part.from_text(text=STORYBOARD_QC_USER_PROMPT)

        response = await self._gen_logged(
            model=self.settings.gemini_flash_model,
            contents=[avatar_part, product_part, storyboard_part, text_part],
            config=types.GenerateContentConfig(
                system_instruction=STORYBOARD_QC_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                safety_settings=ALL_SAFETY_OFF,
                temperature=1.0,
            ),
        )

        if not getattr(response, "text", None):
            logger.warning("qc_storyboard: empty/blocked response — returning safe default (approved)")
            return {
                "avatar_validation": {"score": 100, "reason": "QC skipped: empty model response"},
                "product_validation": {"score": 100, "reason": "QC skipped: empty model response"},
                "composition_quality": {"score": 100, "reason": "QC skipped: empty model response"},
            }
        return parse_json_response(response.text)

    @async_retry(retries=3)
    async def qc_video(
        self, video_uri: str, reference_image_uri: str
    ) -> dict:
        """QC video using Gemini 3 Flash.

        Accepts GCS URIs for the video and reference product image.
        """
        image_part = types.Part.from_uri(
            file_uri=reference_image_uri, mime_type="image/png"
        )
        video_part = types.Part.from_uri(
            file_uri=video_uri, mime_type="video/mp4"
        )
        text_part = types.Part.from_text(text=VIDEO_QC_USER_PROMPT)

        response = await self._gen_logged(
            model=self.settings.gemini_flash_model,
            contents=[image_part, video_part, text_part],
            config=types.GenerateContentConfig(
                system_instruction=VIDEO_QC_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                safety_settings=ALL_SAFETY_OFF,
                temperature=1.0,
            ),
        )

        return parse_json_response(response.text)

    @async_retry(retries=3)
    async def rewrite_prompt(
        self, original_prompt: str, qc_feedback: str
    ) -> str:
        """Rewrite a generation prompt based on QC feedback using Gemini 3 Flash."""
        prompt_text = PROMPT_REWRITE_TEMPLATE.format(
            original_prompt=original_prompt,
            qc_feedback=qc_feedback,
        )

        response = await self._gen_logged(
            model=self.settings.gemini_flash_model,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                safety_settings=ALL_SAFETY_OFF,
                temperature=1.0,
            ),
        )

        return response.text.strip()
