/**
 * Gemini Chat - Main JavaScript
 * Script untuk menangani fungsionalitas chat
 */

document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const clearHistoryButton = document.getElementById('clear-history');

    const API_URL = '/api/chat';
    const chatHistory = [];

    messageInput.addEventListener('input', handleInputChange);
    messageInput.addEventListener('keydown', handleKeyDown);
    sendButton.addEventListener('click', sendMessage);
    clearHistoryButton.addEventListener('click', clearChatHistory);

    loadHistory();

    function handleInputChange() {
        sendButton.disabled = !messageInput.value.trim();
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight > 120 ? 120 : messageInput.scrollHeight) + 'px';
    }

    function handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!sendButton.disabled) sendMessage();
        }
    }

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();

        addMessageToUI('user', message);
        chatHistory.push({ sender: 'user', message });
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;

        showTypingIndicator();
        sendToBackend(message);
    }

    function addMessageToUI(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const messageHeader = document.createElement('div');
        messageHeader.className = 'message-header';

        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender}-avatar`;
        avatar.textContent = sender === 'user' ? 'U' : 'G';

        const senderName = document.createElement('div');
        senderName.className = 'sender';
        senderName.textContent = sender === 'user' ? 'Anda' : 'Gemini';

        messageHeader.appendChild(avatar);
        messageHeader.appendChild(senderName);

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = sender === 'bot' ? processMarkdown(content) : escapeHTML(content);

        messageDiv.appendChild(messageHeader);
        messageDiv.appendChild(messageContent);
        messagesContainer.appendChild(messageDiv);

        scrollToBottom();
    }

    function processMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    function escapeHTML(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator-container';
        typingDiv.id = 'typing-indicator';

        const typingHeader = document.createElement('div');
        typingHeader.className = 'message-header';

        const avatar = document.createElement('div');
        avatar.className = 'avatar bot-avatar';
        avatar.textContent = 'G';

        const senderName = document.createElement('div');
        senderName.className = 'sender';
        senderName.textContent = 'Gemini';

        typingHeader.appendChild(avatar);
        typingHeader.appendChild(senderName);

        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';

        typingDiv.appendChild(typingHeader);
        typingDiv.appendChild(typingIndicator);
        messagesContainer.appendChild(typingDiv);

        scrollToBottom();
    }

    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) typingIndicator.remove();
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async function sendToBackend(message) {
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();

            hideTypingIndicator();
            addMessageToUI('bot', data.response);
            chatHistory.push({ sender: 'bot', message: data.response });
            saveHistory();

        } catch (error) {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessageToUI('bot', 'Maaf, terjadi kesalahan saat menghubungi server. Silakan coba lagi nanti.');
        }
    }

    function saveHistory() {
        localStorage.setItem('gemini-chat-history', JSON.stringify(chatHistory));
    }

    function loadHistory() {
        const history = localStorage.getItem('gemini-chat-history');
        if (history) {
            const parsed = JSON.parse(history);
            parsed.forEach(entry => {
                addMessageToUI(entry.sender, entry.message);
                chatHistory.push(entry);
            });
        }
    }

    function clearChatHistory() {
        localStorage.removeItem('gemini-chat-history');
        messagesContainer.innerHTML = '';
        chatHistory.length = 0;
    }
});
