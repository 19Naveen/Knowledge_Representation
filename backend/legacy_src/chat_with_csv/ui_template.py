CSS = '''
<style>
    .chat-container {
        margin: 0 auto;
        padding: 20px;
    }
    .chat-message {
        display: flex;
        margin-bottom: 20px;
        animation: fadeIn 0.5s;
    }
    .chat-message.user {
        justify-content: flex-end;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        overflow: hidden;
        margin: 0 10px;
    }
    .chat-message .avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .chat-message .message {
        padding: 15px;
        border-radius: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.bot .message {
        background-color: #f0f0f0;
        color: #333;
    }
    .chat-message.user .message {
        background-color: #007bff;
        color: white;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
'''

message_template = '''
<div class="chat-container">
    <div class="chat-message {role}">
        <div class="avatar">
            <img src="{avatar_url}" alt="{role}">
        </div>
        <div class="message">
            {message}
        </div>
    </div>
</div>
'''

def format_message(role, message):
    """
    Formats a message with the given role and message content.

    Args:
        role (str): The role of the message sender. Can be either "bot" or "user".
        message (str): The content of the message.

    Returns:
        str: The formatted message with the role, avatar URL, and message content.
    """
    avatar_url = "https://i.ibb.co/vvPvZcv/robot-croped.gif" if role == "bot" else "https://i.ibb.co/c8s2Mmb/image.png"
    return message_template.format(role=role, avatar_url=avatar_url, message=message)

bot_template = lambda message: format_message("bot", message)
user_template = lambda message: format_message("user", message)