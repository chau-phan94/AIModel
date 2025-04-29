class ChatService:
    @staticmethod
    def clear_history(session_state):
        session_state.chat_history = []
