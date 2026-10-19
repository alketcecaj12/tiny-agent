"""Chapter 9 - Multimodal understanding: send an image along with the task.

    python examples/ch09_multimodal.py                     # book cover from the web
    python examples/ch09_multimodal.py path/to/image.png "What is in this image?"
"""

import base64
import mimetypes
import sys
import urllib.request
from pathlib import Path

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.memory import MultimodalMemory
from tinyagent.planning import NativeReAct
from tinyagent.tools import NativeTools

IMAGE_URL = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png"
QUESTION = "Which animal is on the cover of 'Hands-On Large Language Models'?"


def load_image(source: str) -> tuple[str, str]:
    """Return (base64 data, mime type) for a URL or local path."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            raw = response.read()
            mime = response.headers.get_content_type()
    else:
        raw = Path(source).read_bytes()
        mime = mimetypes.guess_type(source)[0] or "image/png"
    return base64.b64encode(raw).decode("utf-8"), mime


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else IMAGE_URL
    question = sys.argv[2] if len(sys.argv) > 2 else QUESTION
    image_data, mime = load_image(source)

    agent = TinyAgent(
        llm=LLM(model=config.model("vision"), think=True),
        tools=NativeTools(),
        memory=MultimodalMemory(mime_type=mime),
        planner=NativeReAct(),
    )
    print(agent.run(question, image_data=image_data))


if __name__ == "__main__":
    main()
