"""
Updated GUI with RAG integration.
Shows retrieved documents and uses them to augment prompts.
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import scrolledtext, messagebox
import torch

from core.config import (
    GUI_WINDOW_TITLE, GUI_WINDOW_GEOMETRY, GUI_BG_COLOR, GUI_FG_COLOR,
    GUI_FONT_NAME, GUI_FONT_SIZE, INPUT_FRAME_HEIGHT, HISTORY_MAX_DISPLAY_LENGTH,
    SYSTEM_PROMPT, DEVICE, GENERATION_CONFIG, USE_RAG, RAG_SIMILARITY_THRESHOLD
)
from core.utils import build_prompt, clean_response
from core.conversation import ConversationManager

# Import RAG manager if enabled
rag_manager_available = False
RAGManager = None
if USE_RAG:
    try:
        from rag.rag_manager import RAGManager
        rag_manager_available = True
    except ImportError:
        print("Warning: RAGManager not available. Install sentence-transformers and scikit-learn")

# Import nutrition calculator
try:
    from core.nutrition_calculator import calculate_nutrition_portions, NutritionCalculator
    nutrition_available = True
    nutrition_calculator_available = True
except ImportError:
    nutrition_available = False
    nutrition_calculator_available = False
    print("Warning: Nutrition calculator not available")


class ChatbotGUI:
    """Main GUI class for the chatbot application with RAG support."""
    
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
        
        # Initialize RAG manager if enabled
        self.rag_manager = None
        self.rag_toggle = None
        self.use_rag = USE_RAG and rag_manager_available
        if self.use_rag:
            try:
                print("Initializing RAG manager...")
                self.rag_manager = RAGManager()
                print("RAG manager initialized successfully")
            except Exception as e:
                print(f"Error initializing RAG: {e}")
                self.use_rag = False
        
        # Initialize nutrition calculator
        self.nutrition_calculator = None
        if nutrition_available:
            try:
                print("Initializing nutrition calculator...")
                self.nutrition_calculator = NutritionCalculator()
                print("Nutrition calculator initialized successfully")
            except Exception as e:
                print(f"Error initializing nutrition calculator: {e}")
                self.nutrition_calculator = None
        
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
        
        # Right frame: Chat Box + RAG Context
        if self.use_rag:
            self._setup_right_frame_with_rag(paned)
        else:
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
        
        # Right-click menu for clearing history
        self.context_menu = tk.Menu(left_frame, tearoff=0)
        self.context_menu.add_command(label="Clear All", command=self._clear_all_history)
        
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
        """Setup the right frame with chat display and input (without RAG)."""
        right_frame = tk.Frame(paned, bg=GUI_BG_COLOR)
        paned.add(right_frame, minsize=900)
        
        # Chat header
        chat_label = tk.Label(right_frame, text="Chat", bg=GUI_BG_COLOR,
                             fg=GUI_FG_COLOR, font=self.my_font)
        chat_label.pack(padx=5, pady=5)
        
        # Chat display area
        self.chat_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD,
                                                   bg=GUI_BG_COLOR, fg=GUI_FG_COLOR,
                                                   insertbackground=GUI_FG_COLOR,
                                                   font=self.my_font)
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input frame
        input_frame = tk.Frame(right_frame, bg=GUI_BG_COLOR)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Text input - single line with Enter to send
        self.input_text = tk.Text(input_frame, height=1,
                                 bg=GUI_BG_COLOR, fg=GUI_FG_COLOR,
                                 insertbackground=GUI_FG_COLOR, font=self.my_font)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_text.bind("<Return>", lambda e: self._on_send_message() if not e.state & 0x1 else None)
        self.input_text.bind("<Shift-Return>", lambda e: self.input_text.insert(tk.INSERT, "\n") or "break")
        
        # Send button
        send_button = tk.Button(input_frame, text="Send", command=self._on_send_message,
                               bg=GUI_BG_COLOR, fg=GUI_FG_COLOR, font=self.my_font)
        send_button.pack(side=tk.RIGHT, padx=5)
    
    def _setup_right_frame_with_rag(self, paned) -> None:
        """Setup the right frame with RAG context display inline."""
        right_frame = tk.Frame(paned, bg=GUI_BG_COLOR)
        paned.add(right_frame, minsize=900)
        
        # Header with RAG toggle
        header_frame = tk.Frame(right_frame, bg=GUI_BG_COLOR)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        chat_label = tk.Label(header_frame, text="Chat (with RAG)", bg=GUI_BG_COLOR,
                             fg=GUI_FG_COLOR, font=self.my_font)
        chat_label.pack(side=tk.LEFT)
        
        self.rag_toggle = tk.BooleanVar(value=True)
        rag_checkbox = tk.Checkbutton(header_frame, text="Use RAG", variable=self.rag_toggle,
                                     bg=GUI_BG_COLOR, fg=GUI_FG_COLOR, font=self.my_font,
                                     selectcolor=GUI_BG_COLOR)
        rag_checkbox.pack(side=tk.RIGHT)
        
        # Chat display area
        self.chat_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD,
                                                   bg=GUI_BG_COLOR, fg=GUI_FG_COLOR,
                                                   insertbackground=GUI_FG_COLOR,
                                                   font=self.my_font)
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input frame
        input_frame = tk.Frame(right_frame, bg=GUI_BG_COLOR)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Text input - single line with Enter to send
        self.input_text = tk.Text(input_frame, height=1,
                                 bg=GUI_BG_COLOR, fg=GUI_FG_COLOR,
                                 insertbackground=GUI_FG_COLOR, font=self.my_font)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_text.bind("<Return>", lambda e: self._on_send_message() if not e.state & 0x1 else None)
        self.input_text.bind("<Shift-Return>", lambda e: self.input_text.insert(tk.INSERT, "\n") or "break")
        
        # Send button
        send_button = tk.Button(input_frame, text="Send", command=self._on_send_message,
                               bg=GUI_BG_COLOR, fg=GUI_FG_COLOR, font=self.my_font)
        send_button.pack(side=tk.RIGHT, padx=5)
    
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
        self.root.update()  # Update UI
        
        # Show retrieved context if RAG enabled
        # Retrieve and show context before generating response (when RAG is enabled)
        metadata = []
        enhanced_query = message
        if self.rag_manager and self.rag_toggle and self.rag_toggle.get():
            # Parse nutrition needs to enhance query
            if self.nutrition_calculator:
                needs = self.nutrition_calculator.parse_nutrition_needs(message)
                if needs:
                    # Enhance query with nutrition keywords for better retrieval
                    nutrient_keywords = ' '.join(f'high {nutrient}' for nutrient in needs.keys())
                    enhanced_query = f"{message} {nutrient_keywords}"
                    print(f"Enhanced query for nutrition: {enhanced_query}")
            
            try:
                augmented_prompt_text, metadata = self.rag_manager.get_augmented_response(enhanced_query, threshold=RAG_SIMILARITY_THRESHOLD)
                if metadata:
                    self._display_rag_context_inline(message, metadata)
                context_for_generation = augmented_prompt_text
            except Exception as e:
                print(f"RAG context error: {e}")
                context_for_generation = None
        else:
            context_for_generation = None

        # Calculate nutrition if applicable (AFTER user message AND RAG context received, BEFORE prompt building)
        # This ensures nutrition calculation happens with actual food data from RAG
        # but before the complete prompt is built and passed to the AI
        nutrition_info = None
        nutrition_context = None
        if self._should_calculate_nutrition(message, None):
            # Format RAG metadata as food context for the calculator
            rag_food_context = self._format_metadata_as_food_context(metadata)
            nutrition_info = self._calculate_nutrition(message, rag_food_context)
            if nutrition_info:
                nutrition_context = self._format_nutrition_for_prompt(nutrition_info)
                if context_for_generation:
                    context_for_generation += "\n\n" + nutrition_context
                else:
                    context_for_generation = nutrition_context
        
        # Generate response with nutrition context
        response = self._generate_response(message, augmented_context=context_for_generation)
        
        # Add AI response
        self.conv_manager.add_ai_message(response)
        self.chat_text.insert(tk.END, f"AI: {response}\n\n")
        
        # Display nutrition results if they were calculated
        if nutrition_info:
            self._display_nutrition_info(nutrition_info)
        
        self._update_history()
        
        # Auto-scroll to bottom
        self.chat_text.see(tk.END)
    
    def _should_calculate_nutrition(self, user_message: str, ai_response: str = None) -> bool:
        """Check if nutrition calculation should be performed."""
        if not nutrition_calculator_available:
            return False
        
        # Check if user message contains nutrition requirements
        user_lower = user_message.lower()
        nutrition_keywords = ['protein', 'carbs', 'carbohydrates', 'grams', 'need', 'recommendation', 'lunch', 'dinner', 'breakfast', 'meal']
        
        return any(keyword in user_lower for keyword in nutrition_keywords)
    
    def _calculate_nutrition(self, user_message: str, ai_response: str = None) -> dict:
        """Calculate nutrition portions."""
        try:
            return calculate_nutrition_portions(user_message, ai_response or "")
        except Exception as e:
            print(f"Nutrition calculation error: {e}")
            return None
    
    def _format_metadata_as_food_context(self, metadata: list) -> str:
        """
        Format RAG metadata into food context string for nutrition calculator.
        
        Args:
            metadata: List of metadata dicts from RAG retrieval
            
        Returns:
            Formatted string with food names and nutrition info
        """
        if not metadata:
            return ""
        
        food_items = []
        for meta in metadata:
            data = meta.get('data', {})
            food_name = data.get('food', 'Unknown')
            if food_name and food_name.lower() != 'unknown':
                # Use the exact food name from database for better matching
                food_items.append(food_name)
        
        # Return as comma-separated list
        return ", ".join(food_items) if food_items else ""

    
    def _format_nutrition_for_prompt(self, nutrition_info: dict) -> str:
        """Format nutrition calculation results as context for the AI prompt."""
        if not nutrition_info:
            return ""
        
        lines = ["[CALCULATED NUTRITION DATA - USE THESE EXACT VALUES IN RESPONSE]:"]
        lines.append("")
        
        portions = nutrition_info.get('portions', {})
        if portions:
            lines.append("RECOMMENDED FOOD PORTIONS:")
            for food, portion in portions.items():
                if portion > 0:
                    lines.append(f"  • {food}: {portion:.1f} grams")
        
        actual = nutrition_info.get('actual_nutrition', {})
        target = nutrition_info.get('target_needs', {})
        
        if actual or target:
            lines.append("")
            lines.append("NUTRITION TOTALS FOR RECOMMENDED PORTIONS:")
            for nutrient in ['protein', 'carbs', 'fat', 'calories']:
                actual_val = actual.get(nutrient, 0) if actual else 0
                target_val = target.get(nutrient, 0) if target else 0
                
                if target_val > 0 or actual_val > 0:
                    unit = 'kcal' if nutrient == 'calories' else 'g'
                    status = "✓ MET" if actual_val >= target_val else "⚠ PARTIAL"
                    lines.append(f"  • {nutrient.capitalize()}: {actual_val:.1f}{unit} / {target_val:.1f}{unit} {status}")
        
        lines.append("")
        lines.append("IMPORTANT: Show these exact values in your response. Do not recalculate or estimate.")
        
        return "\n".join(lines)
    
    def _display_nutrition_info(self, nutrition_info: dict) -> None:
        """Display nutrition calculation results."""
        self.chat_text.insert(tk.END, "[Nutrition Calculation]:\n")
        
        portions = nutrition_info.get('portions', {})
        if portions:
            self.chat_text.insert(tk.END, "Recommended Portions:\n")
            for food, portion in portions.items():
                if portion > 0:
                    self.chat_text.insert(tk.END, f"  - {food}: {portion:.1f}g\n")
        
        actual = nutrition_info.get('actual_nutrition', {})
        target = nutrition_info.get('target_needs', {})
        
        if actual and target:
            self.chat_text.insert(tk.END, "\nNutrition Summary:\n")
            for nutrient in ['protein', 'carbs', 'fat', 'calories']:
                if nutrient in target:
                    actual_val = actual.get(nutrient, 0)
                    target_val = target[nutrient]
                    self.chat_text.insert(tk.END, f"  - {nutrient.capitalize()}: {actual_val:.1f}g / {target_val:.1f}g\n")
        
        self.chat_text.insert(tk.END, "\n")
    
    def _generate_response(self, user_message: str = None, augmented_context: str = None) -> str:
        """
        Generate AI response with optional RAG augmentation.
        
        Args:
            user_message: Current user message (for RAG)
            augmented_context: Text context for model generation
        
        Returns:
            str: Generated response
        """
        # Get conversation history
        conv = self.conv_manager.get_current_conversation()

        # Build base prompt from conversation (system prompt + user prompt prepared here)
        prompt = build_prompt(SYSTEM_PROMPT, conv)

        # Append augmented context if available (nutrition calculation results are in here)
        # At this point: system prompt + user prompt are prepared, nutrition is calculated, now passing to AI
        if augmented_context:
            prompt = f"{prompt}\n\nRetrieved context:\n{augmented_context}\n"
        
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
    
    def _display_rag_context_inline(self, query: str, metadata: list) -> None:
        """
        Display retrieved context inline in the chat.
        
        Args:
            query: The user query
            metadata: List of retrieved document metadata
        """
        context_text = "\n[Retrieved Context]:\n"
        context_text += "─" * 40 + "\n"
        
        display_count = min(len(metadata), 10)
        context_text += f"Showing top {display_count} of {len(metadata)} retrieved foods:\n"
        for i, meta in enumerate(metadata[:display_count], 1):
            data = meta.get('data', {})
            food_name = data.get('food', 'Unknown food')
            caloric_value = data.get('Caloric Value', 'N/A')
            
            context_text += f"• {food_name} ({caloric_value} kcal)\n"
            # Show key nutritional info
            key_nutrients = ['Protein', 'Fat', 'Carbohydrates', 'Sugars']
            for nutrient in key_nutrients:
                value = data.get(nutrient)
                if value is not None and str(value) != 'nan':
                    context_text += f"    {nutrient}: {value}g\n"
        
        context_text += "─" * 40 + "\n\n"
        self.chat_text.insert(tk.END, context_text)
    
    def _update_rag_context(self, query: str, metadata: list) -> None:
        """
        Legacy method - kept for compatibility.
        
        Args:
            query: The user query
            metadata: List of retrieved document metadata
        """
        # Now handled inline in chat
        pass
    
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
                self.chat_text.insert(tk.END, f"AI: {msg}\n\n")
    
    def _update_history(self) -> None:
        """Update the history listbox with all conversations."""
        self.history_listbox.delete(0, tk.END)
        convs = self.conv_manager.get_all_conversations()
        for i, conv in enumerate(convs):
            summary = self.conv_manager.get_conversation_summary(i, HISTORY_MAX_DISPLAY_LENGTH)
            self.history_listbox.insert(tk.END, summary)
        # Highlight current conversation
        self.history_listbox.selection_set(self.conv_manager.current_conv_index)
    
    def _clear_all_history(self) -> None:
        """Clear all conversation history."""
        if messagebox.askyesno("Clear History", "Clear all conversations?"):
            self.conv_manager.conversations = [[]]
            self.conv_manager.current_conv_index = 0
            self._update_history()
            self._load_conversation(0)
    
    def run(self) -> None:
        """Start the GUI application."""
        self.root.mainloop()
