"""
Conversation state management.
Displays conversation history and manages multiple conversations.
"""


class ConversationManager:
    """Manages multiple conversations and current conversation tracking."""
    
    def __init__(self):
        """Initialize conversation manager with one empty conversation."""
        self.conversations = [[]]
        self.current_conv_index = 0
    
    def new_conversation(self) -> int:
        """
        Create a new conversation.
        
        Returns:
            int: Index of the newly created conversation
        """
        self.conversations.append([])
        self.current_conv_index = len(self.conversations) - 1
        return self.current_conv_index
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the current conversation.
        
        Args:
            message: The user's message
        """
        self.conversations[self.current_conv_index].append(message)
    
    def add_ai_message(self, message: str) -> None:
        """
        Add an AI response to the current conversation.
        
        Args:
            message: The AI's response
        """
        self.conversations[self.current_conv_index].append(message)
    
    def get_current_conversation(self) -> list:
        """
        Get the current conversation.
        
        Returns:
            list: List of messages in the current conversation
        """
        return self.conversations[self.current_conv_index]
    
    def set_current_conversation(self, index: int) -> None:
        """
        Switch to a different conversation.
        
        Args:
            index: Index of the conversation to switch to
        """
        if 0 <= index < len(self.conversations):
            self.current_conv_index = index
    
    def get_all_conversations(self) -> list:
        """
        Get all conversations.
        
        Returns:
            list: List of all conversations
        """
        return self.conversations
    
    def get_conversation_summary(self, index: int, max_length: int = 30) -> str:
        """
        Get a summary of a conversation for display.
        
        Args:
            index: Index of the conversation
            max_length: Maximum length of the summary
            
        Returns:
            str: Summary string truncated to max_length
        """
        conv = self.conversations[index]
        if not conv:
            return "(new chat)"
        
        # Get the last user message (even indices are user turns)
        user_msgs = [conv[j] for j in range(0, len(conv), 2)]
        if user_msgs:
            latest_prompt = user_msgs[-1]
            if len(latest_prompt) > max_length:
                return latest_prompt[:max_length] + "..."
            return latest_prompt
        return "(no user prompt)"
