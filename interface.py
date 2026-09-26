import re
import base64
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage


class ChatInterface:
    """
    A chat interface for interacting with a medical AI agent through Gradio.

    Handles file uploads, message processing, and chat history management.
    Supports both regular image files and DICOM medical imaging files.
    """

    def __init__(self, agent, tools_dict, send_image=True):
        """
        Initialize the chat interface.

        Args:
            agent: The medical AI agent to handle requests
            tools_dict (dict): Dictionary of available tools for image processing
            send_image (bool): Whether the director LLM sees the image itself. If False it
                only gets the image path, so all perception comes from the tools.
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.send_image = send_image
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        # Separate storage for original and display paths
        self.original_file_path = None  # For tools (.dcm or other)
        self.viewable_file_path = None  # Viewable version of the original, sent to the LLM
        self.display_file_path = None  # For UI (may be a tool's annotated image)
        self.last_sent_file_path = None  # Image last sent to the LLM in this thread

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path)

        # Handle DICOM conversion for display only
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = output["image_path"]
        else:
            self.display_file_path = str(saved_path)
        self.viewable_file_path = self.display_file_path

        return self.display_file_path

    def _current_image(self, display_image: Optional[str]) -> Optional[str]:
        """
        Return the original image to analyze, registering the displayed image if it was
        never uploaded through handle_upload.

        Args:
            display_image (Optional[str]): Path to image being displayed

        Returns:
            Optional[str]: Path to the original image, or None if there is no image
        """
        if self.original_file_path is None and display_image:
            self.handle_upload(display_image)
        return self.original_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history.

        Args:
            message (str): Text message to add
            display_image (str): Path to image being displayed
            history (List[dict]): Current chat history

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        image_path = self._current_image(display_image)
        # Only show the image in the chat when it is new
        if image_path is not None and image_path != self.last_sent_file_path:
            history.append({"role": "user", "content": {"path": self.viewable_file_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history

        Yields:
            Tuple[List[ChatMessage], Optional[str], str]: Updated chat history, display path, and empty string
        """
        chat_history = chat_history or []

        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        image_path = self._current_image(display_image)

        if image_path is not None:
            # Send path for tools; flag a changed image so the LLM stops using earlier ones
            if self.last_sent_file_path is not None and image_path != self.last_sent_file_path:
                path_text = (
                    f"A NEW image was uploaded and replaces all earlier images in this "
                    f"conversation. Earlier findings and tool results do not apply to it. "
                    f"image_path: {image_path}"
                )
            else:
                path_text = f"image_path: {image_path}"
            messages.append({"role": "user", "content": path_text})
            self.last_sent_file_path = image_path

            if self.send_image:
                # Load and encode the viewable original (never a tool's annotated image)
                with open(self.viewable_file_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            }
                        ],
                    }
                )

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

        shown_images = set()  # Images already added to the chat in this reply
        validated = False  # Whether EDV critiqued a tool result in this reply
        try:
            for event in self.agent.workflow.stream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        ai_message = event["process"]["messages"][-1]
                        content = ai_message.content
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content)
                            metadata = None
                            # With EDV, the director's closing answer is replaced by the
                            # final responder's; keep it visible but collapsed
                            if (
                                validated
                                and self.agent.edv_respond
                                and not ai_message.tool_calls
                            ):
                                metadata = {
                                    "title": "📝 Director draft (superseded by the EDV final responder)"
                                }
                            chat_history.append(
                                ChatMessage(role="assistant", content=content, metadata=metadata)
                            )
                            yield chat_history, self.display_file_path, ""

                    elif "respond" in event:
                        content = re.sub(r"temp/[^\s]*", "", event["respond"]["messages"][-1].content)
                        chat_history.append(ChatMessage(role="assistant", content=content))
                        yield chat_history, self.display_file_path, ""

                    elif "execute" in event:
                        notes = {n["tool_call_id"]: n for n in event["execute"].get("edv_notes", [])}
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            tool_result = eval(message.content)[0]

                            # image_visualizer only re-displays an image; skip it if this
                            # reply already showed that image (e.g. a grounding visualization)
                            if tool_name == "image_visualizer" and isinstance(tool_result, dict):
                                shown_path = tool_result.get("image_path")
                                if shown_path and Path(shown_path).as_posix() in shown_images:
                                    continue

                            if tool_result:
                                metadata = {"title": f"🖼️ Image from tool: {tool_name}"}
                                formatted_result = " ".join(
                                    line.strip() for line in str(tool_result).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result,
                                        metadata=metadata,
                                    )
                                )

                            # For image_visualizer, use display path
                            if tool_name == "image_visualizer":
                                self.display_file_path = tool_result["image_path"]
                                shown_images.add(Path(self.display_file_path).as_posix())
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        # content=gr.Image(value=self.display_file_path),
                                        content={"path": self.display_file_path},
                                    )
                                )
                            # For tools that return an annotated image (e.g. grounding boxes)
                            elif isinstance(tool_result, dict) and tool_result.get(
                                "visualization_path"
                            ):
                                self.display_file_path = tool_result["visualization_path"]
                                shown_images.add(Path(self.display_file_path).as_posix())
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content={"path": self.display_file_path},
                                    )
                                )

                            # EDV critique of this tool result, if any
                            note = notes.get(message.tool_call_id)
                            if note:
                                validated = True
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=note["explanation"],
                                        metadata={"title": f"🔎 Evidence validation: {tool_name}"},
                                    )
                                )

                            yield chat_history, self.display_file_path, ""

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_file_path, ""


def create_demo(agent, tools_dict, send_image=True, director_label=None):
    """
    Create a Gradio demo interface for the medical AI agent.

    Args:
        agent: The medical AI agent to handle requests
        tools_dict (dict): Dictionary of available tools for image processing
        send_image (bool): Whether the director LLM sees the image itself
        director_label (str, optional): Director model name to show in the header

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    interface = ChatInterface(agent, tools_dict, send_image=send_image)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
            # 🏥 MedRAX
            Medical Reasoning Agent for Chest X-ray
            """
            )
            if director_label:
                gr.Markdown(
                    f"Director: **{director_label}** · "
                    f"{'sees the image' if send_image else 'tools only (does not see the image)'}"
                )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Agent",
                        avatar_images=(
                            None,
                            "assets/medrax_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Ask about the X-ray...",
                                container=False,
                            )

                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image", type="filepath", height=700, container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload X-Ray",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload DICOM",
                            file_types=["file"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread")

        # Event handlers
        def clear_image():
            interface.original_file_path = None
            interface.viewable_file_path = None
            interface.display_file_path = None

        def clear_chat():
            clear_image()
            return [], None

        def new_thread():
            interface.current_thread_id = str(time.time())
            interface.last_sent_file_path = None
            # Show the original image again rather than a previous tool's annotation
            interface.display_file_path = interface.viewable_file_path
            return [], interface.display_file_path

        def handle_file_upload(file):
            return interface.handle_upload(file.name)

        def handle_panel_upload(file_path):
            # Images dropped or uploaded directly onto the image panel
            return interface.handle_upload(file_path)

        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)

        dicom_upload.upload(handle_file_upload, inputs=dicom_upload, outputs=image_display)

        image_display.upload(handle_panel_upload, inputs=image_display, outputs=image_display)
        image_display.clear(clear_image)

        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
