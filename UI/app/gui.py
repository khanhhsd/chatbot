
"""
GUI setup and event handlers for the chatbot application.
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import scrolledtext
import torch

from core.config import (
    GUI_WINDOW_TITLE, GUI_WINDOW_GEOMETRY, GUI_BG_COLOR, GUI_FG_COLOR,
    GUI_FONT_NAME, GUI_FONT_SIZE, INPUT_FRAME_HEIGHT, HISTORY_MAX_DISPLAY_LENGTH,
    SYSTEM_PROMPT, DEVICE, GENERATION_CONFIG
)
from core.utils import build_prompt, clean_response
from core.conversation import ConversationManager


class ChatbotGUI:
    """Main GUI class for the chatbot application."""
    
    def __init__(self, tokenizer, model):
        """
        Initialize the GUI with model and tokenizer.
        
        Args:
            tokenizer: Transformers tokenizer
            model: Transformers model for generation
        """
        self.tokenizer = tokenizer
        self.model = model
        self.conv_manager = ConversationManager()
        
        # Setup main window
        self.root = tk.Tk()
        self.root.title(GUI_WINDOW_TITLE)
        self.root.geometry(GUI_WINDOW_GEOMETRY)
        self.root.configure(bg=GUI_BG_COLOR)
        
        # Setup font
        self.my_font = tkfont.Font(size=GUI_FONT_SIZE, family=GUI_FONT_NAME)
        
        # Initialize GUI components
        self._setup_layout()
        self._update_history()
        self._load_conversation(0)
    
    def _setup_layout(self) -> None:
        """Setup all GUI components and layout."""
        # Create paned window for resizable left/right sections
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=GUI_BG_COLOR)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left frame: Chat History
        self._setup_left_frame(paned)
        
        # Right frame: Chat Box
        self._setup_right_frame(paned)
    
    def _setup_left_frame(self, paned) -> None:
        """Setup the left frame with chat history."""
        left_frame = tk.Frame(paned, bg=GUI_BG_COLOR)
        paned.add(left_frame, minsize=369)
        
        # Header
        history_label = tk.Label(left_frame, text="Chat History", bg=GUI_BG_COLOR, 
                                fg=GUI_FG_COLOR, font=self.my_font)
        history_label.pack()
        
        # New Chat button
        new_chat_button = tk.Button(left_frame, text="New Chat", command=self._on_new_chat,
                                   bg=GUI_BG_COLOR, fg=GUI_FG_COLOR, font=self.my_font)
        new_chat_button.pack()
        
        # History listbox
        self.history_listbox = tk.Listbox(left_frame, bg=GUI_BG_COLOR, 
                                         fg=GUI_FG_COLOR, font=self.my_font)
        history_scrollbar = tk.Scrollbar(left_frame, orient=tk.VERTICAL,
                                        command=self.history_listbox.yview,
                                        bg=GUI_BG_COLOR, troughcolor=GUI_BG_COLOR)
        self.history_listbox.config(yscrollcommand=history_scrollbar.set)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_listbox.bind('<<ListboxSelect>>', self._on_select)
    
    def _setup_right_frame(self, paned) -> None:
        """Setup the right frame with chat display and input."""
        right_frame = tk.Frame(paned, bg=GUI_BG_COLOR)
        paned.add(right_frame, minsize=900)
        
        # Chat header
        chat_label = tk.Label(right_frame, text="Chat", bg=GUI_BG_COLOR,
                             fg=GUI_FG_COLOR, font=self.my_font)
        chat_label.pack()
        
        # Chat display area
        self.chat_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD,
                                                   bg=GUI_BG_COLOR, fg=GUI_FG_COLOR,
                                                   insertbackground=GUI_FG_COLOR,
                                                   font=self.my_font)
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        
        # Input frame
        input_frame = tk.Frame(right_frame, bg=GUI_BG_COLOR)
        input_frame.pack(fill=tk.X)
        
        # Text input
        self.input_text = tk.Text(input_frame, height=INPUT_FRAME_HEIGHT,
                                 bg=GUI_BG_COLOR, fg=GUI_FG_COLOR,
                                 insertbackground=GUI_FG_COLOR, font=self.my_font)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Send button
        send_button = tk.Button(input_frame, text="Send", command=self._on_send_message,
                               bg=GUI_BG_COLOR, fg=GUI_FG_COLOR, font=self.my_font)
        send_button.pack(side=tk.RIGHT)
    
    def _on_new_chat(self) -> None:
        """Handle new chat button click."""
        self.conv_manager.new_conversation()
        self._update_history()
        self._load_conversation(self.conv_manager.current_conv_index)
    
    def _on_select(self, event) -> None:
        """Handle history listbox selection."""
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            self.conv_manager.set_current_conversation(index)
            self._load_conversation(index)
    
    def _on_send_message(self) -> None:
        """Handle send message button click."""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Add user message
        self.conv_manager.add_user_message(message)
        self.chat_text.insert(tk.END, f"You: {message}\n")
        self.input_text.delete("1.0", tk.END)
        
        # Generate response
        response = self._generate_response()
        
        # Add AI response
        self.conv_manager.add_ai_message(response)
        self.chat_text.insert(tk.END, f"AI: {response}\n")
        self._update_history()
    
    def _generate_response(self) -> str:
        """
        Generate AI response for the current conversation.
        
        Returns:
            str: Generated response
        """
        # Build prompt from conversation history
        conv = self.conv_manager.get_current_conversation()
        prompt = build_prompt(SYSTEM_PROMPT, conv)
        
        # Tokenize and prepare inputs
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **GENERATION_CONFIG
            )
        
        # Decode and clean response
        raw = self.tokenizer.decode(output[0][input_ids.shape[-1]:], 
                                   skip_special_tokens=True)
        response = clean_response(raw)
        return response
    
    def _load_conversation(self, index: int) -> None:
        """
        Load and display a conversation.
        
        Args:
            index: Index of the conversation to load
        """
        self.chat_text.delete("1.0", tk.END)
        conv = self.conv_manager.conversations[index]
        for i, msg in enumerate(conv):
            if i % 2 == 0:
                self.chat_text.insert(tk.END, f"You: {msg}\n")
            else:
                self.chat_text.insert(tk.END, f"AI: {msg}\n")
    
    def _update_history(self) -> None:
        """Update the history listbox with all conversations."""
        self.history_listbox.delete(0, tk.END)
        for i, conv in enumerate(self.conv_manager.get_all_conversations()):
            summary = self.conv_manager.get_conversation_summary(i, HISTORY_MAX_DISPLAY_LENGTH)
            self.history_listbox.insert(tk.END, summary)
    
    def run(self) -> None:
        """Start the GUI application."""
        self.root.mainloop()
